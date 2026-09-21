"""Detector diagnostics: is the forward-consistency metric channel-biased?

Runs the three diagnostics requested to tell apart "action attacks are
intrinsically stealthy" from "the one-step forward metric is blind to the action
channel":

  A. Controlled channel sensitivity: apply an isolated normalized perturbation
     of magnitude m to the observation vs. the action and measure the resulting
     one-step forward residual (true dynamics). obs gain is ~1 by construction;
     the act gain = ||(f(s,a+dа) - f(s,a))/scale|| / m is the empirical J_a.

  B. Forward detector at horizon k (k=1,2,4,8): the k-step version rolls the
     model forward WITHOUT re-anchoring on the real observation, so accumulated
     trajectory drift can re-enter the residual. AUROC(clean vs attacked) is
     reported per channel.

  C. Inverse-dynamics detector: g(o_{t-1}, o_t) -> a_hat trained on clean data;
     score = ||a_hat - a_v||. AUROC per channel.

Forward models use the TRUE MuJoCo dynamics to isolate channel sensitivity /
re-anchoring from PEDM estimation error. A learned detector can only be worse.

Example:
    python examples/diagnose_detectors.py --n_episodes 15 --episode_length 200
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.attacks.adversary_ac import AdversaryActorCritic, orthogonal_init
from examples.eval_robust_detector import Victim, auroc
from examples.train_obs_attacker import (
    AntTrueDynamics, l2_project, victim_act_batch, estimate_scales, resolve_victim_run,
)


def load_attacker(path, obs_dim, act_dim, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    channel = ck.get("channel", "obs")
    mode = ck.get("mode", "ppo")
    out_dim = ck.get("out_dim", obs_dim if channel == "obs" else act_dim)
    # infer in_dim from the first layer weight
    in_dim = ck["state_dict"]["actor.0.weight"].shape[1]
    net = AdversaryActorCritic(in_dim, out_dim, hidden_size=256, activation="tanh").to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    net.illusory_input = ck.get("illusory_input", "privileged")  # carried for rollout input building
    return net, channel, mode, float(ck["budget"])


def build_adv_input(channel, mode, obs_n, pert_prev_n, expected_n, a_v, illusory_input="privileged"):
    if channel == "obs":
        if mode == "illusory":
            if illusory_input == "history":
                return np.concatenate([obs_n, pert_prev_n, a_v], axis=1).astype(np.float32)  # a_v = a_prev
            return np.concatenate([obs_n, pert_prev_n, expected_n], axis=1).astype(np.float32)
        return obs_n.astype(np.float32)
    if mode == "illusory" and illusory_input != "history":
        return np.concatenate([obs_n, a_v, expected_n], axis=1).astype(np.float32)
    return np.concatenate([obs_n, a_v], axis=1).astype(np.float32)


@torch.no_grad()
def rollout(net, channel, mode, budget, victim, dyn, scenario, scale, ascale,
            act_low, act_high, n_episodes, ep_len, device, seed0):
    """Return list of episodes: dict(o_seq[T+1,od], a_seq[T,ad], dnorm[T], ret)."""
    env = gym.make(scenario)
    obs_dim = scale.shape[1]
    act_dim = ascale.shape[1]
    scale1, ascale1 = scale[0], ascale[0]  # 1-D per-dim scales
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        obs = np.asarray(obs, np.float32)
        victim.reset()
        last_pert = obs.copy()
        last_av = np.zeros(act_dim, np.float32)
        last_done = 0.0
        o_seq, a_seq, dnorm = [], [], []
        ret = 0.0
        for t in range(ep_len):
            s_n = obs / scale1
            expected = dyn.next_obs(last_pert, last_av) if last_done == 0 else obs
            exp_n = (expected / scale1)[None]
            obs_n = s_n[None]
            if net is None:  # clean
                a_v = np.clip(victim.act(obs), act_low, act_high)
                o_vis, env_action, phi = obs, a_v, np.zeros(obs_dim, np.float32)
                new_last_pert, new_last_av = obs, a_v
            elif channel == "obs":
                adv_in = build_adv_input("obs", mode, obs_n, (last_pert / scale1)[None], exp_n,
                                         last_av[None], getattr(net, "illusory_input", "privileged"))
                a_t, _, _ = net.act(torch.as_tensor(adv_in, device=device), deterministic=True)
                phi = l2_project(a_t.cpu().numpy()[0], budget)
                o_vis = obs + phi * scale1
                a_v = np.clip(victim.act(o_vis), act_low, act_high)
                env_action = a_v
                new_last_pert, new_last_av = o_vis, a_v
            else:  # act
                a_v = np.clip(victim.act(obs), act_low, act_high)
                adv_in = build_adv_input("act", mode, obs_n, None, exp_n, a_v[None],
                                         getattr(net, "illusory_input", "privileged"))
                a_t, _, _ = net.act(torch.as_tensor(adv_in, device=device), deterministic=True)
                phi = l2_project(a_t.cpu().numpy()[0], budget)
                env_action = np.clip(a_v + phi * ascale1, act_low, act_high)
                o_vis = obs  # detector sees true obs
                new_last_pert, new_last_av = obs, a_v

            o_seq.append(o_vis.astype(np.float32))
            a_seq.append(a_v.astype(np.float32))
            dnorm.append(float(np.linalg.norm(phi)))
            no, r, term, trunc, _ = env.step(env_action)
            ret += float(r)
            obs = np.asarray(no, np.float32)
            last_pert, last_av, last_done = new_last_pert, new_last_av, float(term or trunc)
            if term or trunc:
                break
        # drop the last action: its reported next-obs was never recorded, so
        # padding it (previously with a duplicate obs) created a bogus transition.
        episodes.append({"o": np.array(o_seq), "a": np.array(a_seq[:-1]),
                         "dnorm": np.array(dnorm[:-1]), "ret": ret})
    env.close()
    return episodes


def forward_kstep_scores(ep, dyn, scale, k):
    """Per-transition normalized residual of a k-step forward rollout (no re-anchor)."""
    o, a = ep["o"], ep["a"]
    n = len(a)
    scores = []
    for t in range(n - k):
        pred = o[t]
        for j in range(k):
            pred = dyn.next_obs(pred, a[t + j])
        scores.append(np.linalg.norm((pred - o[t + k]) / scale[0]))
    return np.array(scores)


class InvDyn(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(2 * obs_dim, hidden)), nn.ReLU(),
            orthogonal_init(nn.Linear(hidden, hidden)), nn.ReLU(),
            orthogonal_init(nn.Linear(hidden, act_dim)),
        )

    def forward(self, o0, o1):
        return self.net(torch.cat([o0, o1], -1))


def train_invdyn(clean_eps, obs_dim, act_dim, scale, device, epochs=60):
    O0, O1, A = [], [], []
    for ep in clean_eps:
        o, a = ep["o"], ep["a"]
        for t in range(len(a)):
            O0.append(o[t] / scale[0]); O1.append(o[t + 1] / scale[0]); A.append(a[t])
    O0 = torch.tensor(np.array(O0), dtype=torch.float32, device=device)
    O1 = torch.tensor(np.array(O1), dtype=torch.float32, device=device)
    A = torch.tensor(np.array(A), dtype=torch.float32, device=device)
    g = InvDyn(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    n = len(A)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 512):
            idx = perm[s:s + 512]
            loss = (g(O0[idx], O1[idx]) - A[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return g


@torch.no_grad()
def invdyn_scores(ep, g, scale, ascale, device):
    o, a = ep["o"], ep["a"]
    o0 = torch.tensor(o[:-1] / scale[0], dtype=torch.float32, device=device)
    o1 = torch.tensor(o[1:] / scale[0], dtype=torch.float32, device=device)
    a_hat = g(o0, o1).cpu().numpy()
    return np.linalg.norm((a_hat - a) / ascale[0], axis=1)


def pooled_auroc(clean_eps, att_eps, score_fn):
    c = np.concatenate([score_fn(e) for e in clean_eps])
    a = np.concatenate([score_fn(e) for e in att_eps])
    return auroc(c, a)


def experiment_A(victim, dyn, scenario, scale, ascale, act_low, act_high, device,
                 n_samples=500, mags=(0.05, 0.1, 0.2, 0.4)):
    """Controlled per-unit channel sensitivity of the one-step forward residual."""
    env = gym.make(scenario)
    obs, _ = env.reset(seed=123)
    victim.reset()
    S, AV = [], []
    for _ in range(n_samples):
        s = np.asarray(obs, np.float32)
        a_v = np.clip(victim.act(s), act_low, act_high)
        S.append(s); AV.append(a_v)
        obs, _, term, trunc, _ = env.step(a_v)
        if term or trunc:
            obs, _ = env.reset(); victim.reset()
    env.close()
    S, AV = np.array(S), np.array(AV)
    obs_dim, act_dim = scale.shape[1], ascale.shape[1]
    print("\n[A] controlled one-step channel sensitivity (true dynamics)")
    print(f"  {'magnitude':>10}{'obs_resid':>12}{'act_resid':>12}{'act_gain':>10}")
    rng = np.random.default_rng(0)
    for m in mags:
        obs_res, act_res = [], []
        for i in range(n_samples):
            s, a_v = S[i], AV[i]
            # obs one-shot: residual = ||delta_o||_normalized = m (by construction)
            do = rng.standard_normal(obs_dim); do = do / np.linalg.norm(do) * m
            obs_res.append(np.linalg.norm(do))
            # act one-shot: residual = ||(f(s,a+da) - f(s,a))/scale||, ||da||_norm = m
            da = rng.standard_normal(act_dim); da = da / np.linalg.norm(da) * m
            f0 = dyn.next_obs(s, a_v)
            f1 = dyn.next_obs(s, np.clip(a_v + da * ascale[0], act_low, act_high))
            act_res.append(np.linalg.norm((f1 - f0) / scale[0]))
        omean, amean = np.mean(obs_res), np.mean(act_res)
        print(f"  {m:>10.2f}{omean:>12.4f}{amean:>12.4f}{amean / m:>10.3f}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", default="Ant-v4")
    p.add_argument("--victim_run", default="")
    p.add_argument("--obs_ppo", default="results/obs_attackers/Ant-v4/attacker_ppo.pt")
    p.add_argument("--act_ppo", default="results/obs_attackers/Ant-v4/attacker_act_ppo.pt")
    p.add_argument("--n_episodes", type=int, default=15)
    p.add_argument("--episode_length", type=int, default=200)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--pedm_epochs", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    from gymnasium.spaces import Box
    victim_run = resolve_victim_run(args.victim_run or
                                    "results/robust_victim/Ant-v4/mappo/*/seed-00001-*")
    probe = gym.make(args.scenario)
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    act_space = Box(probe.action_space.low, probe.action_space.high,
                    probe.action_space.shape, np.float32)
    act_low, act_high = act_space.low.astype(np.float32), act_space.high.astype(np.float32)
    probe.close()

    victim = Victim(victim_run, obs_dim, act_space, device)
    dyn = AntTrueDynamics(args.scenario)
    print("estimating scales...")
    obs_scale, act_scale = estimate_scales(args.scenario, victim, act_low, act_high)
    scale, ascale = obs_scale[None], act_scale[None]

    # ---- Experiment A ----
    experiment_A(victim, dyn, args.scenario, scale, ascale, act_low, act_high, device)

    # ---- rollouts: clean, obs_ppo, act_ppo ----
    print("\ncollecting rollouts (clean / obs_ppo / act_ppo)...")
    clean = rollout(None, "obs", "ppo", 0.0, victim, dyn, args.scenario, scale, ascale,
                    act_low, act_high, args.n_episodes, args.episode_length, device, 5000)
    obs_net, obs_ch, obs_mode, obs_b = load_attacker(args.obs_ppo, obs_dim, act_dim, device)
    act_net, act_ch, act_mode, act_b = load_attacker(args.act_ppo, obs_dim, act_dim, device)
    obs_eps = rollout(obs_net, obs_ch, obs_mode, obs_b, victim, dyn, args.scenario, scale,
                      ascale, act_low, act_high, args.n_episodes, args.episode_length, device, 7000)
    act_eps = rollout(act_net, act_ch, act_mode, act_b, victim, dyn, args.scenario, scale,
                      ascale, act_low, act_high, args.n_episodes, args.episode_length, device, 8000)

    clean_ret = np.mean([e["ret"] for e in clean])
    print(f"\nvictim return: clean {clean_ret:.0f} | "
          f"obs_ppo {np.mean([e['ret'] for e in obs_eps]):.0f} "
          f"(drop {clean_ret - np.mean([e['ret'] for e in obs_eps]):.0f}) | "
          f"act_ppo {np.mean([e['ret'] for e in act_eps]):.0f} "
          f"(drop {clean_ret - np.mean([e['ret'] for e in act_eps]):.0f})")
    print("mean applied perturbation norm (normalized): "
          f"obs {np.mean([e['dnorm'].mean() for e in obs_eps]):.3f} | "
          f"act {np.mean([e['dnorm'].mean() for e in act_eps]):.3f}")

    # ---- inverse-dynamics detector ----
    print("\ntraining inverse-dynamics model on clean data...")
    g = train_invdyn(clean, obs_dim, act_dim, scale, device)

    # ---- learned PEDM forward detector (realistic noise floor) ----
    from harl.detectors.pedm_detector import PEDMDetector
    print("fitting learned PEDM detector on clean data...")
    fo = np.concatenate([e["o"][:-1] for e in clean])
    fa = np.concatenate([e["a"] for e in clean])
    fn = np.concatenate([e["o"][1:] for e in clean])
    pedm = PEDMDetector(obs_dim, act_dim, n_part=100, device=device)
    pedm.fit(fo, fa, fn, n_train_epochs=args.pedm_epochs)

    def pedm_scores(e):
        return pedm.predict_scores(e["o"], e["a"])

    def mean_score(eps, fn):
        return float(np.mean(np.concatenate([fn(e) for e in eps])))

    # ---- detector x channel AUROC matrix ----
    print("\n[B/C] AUROC (clean vs attacked)")
    header = f"  {'detector':<18}{'obs_ppo':>10}{'act_ppo':>10}"
    print(header); print("  " + "-" * (len(header) - 2))
    for k in args.ks:
        fn = lambda e, k=k: forward_kstep_scores(e, dyn, scale, k)
        print(f"  true_fwd_{k}step{'':<4}{pooled_auroc(clean, obs_eps, fn):>10.3f}"
              f"{pooled_auroc(clean, act_eps, fn):>10.3f}")
    print(f"  {'pedm_1step':<18}{pooled_auroc(clean, obs_eps, pedm_scores):>10.3f}"
          f"{pooled_auroc(clean, act_eps, pedm_scores):>10.3f}")
    fn_inv = lambda e: invdyn_scores(e, g, scale, ascale, device)
    print(f"  {'inverse_dyn':<18}{pooled_auroc(clean, obs_eps, fn_inv):>10.3f}"
          f"{pooled_auroc(clean, act_eps, fn_inv):>10.3f}")

    print("\nmean per-transition score (noise floor vs signal):")
    print(f"  {'detector':<18}{'clean':>10}{'obs_ppo':>10}{'act_ppo':>10}")
    tf1 = lambda e: forward_kstep_scores(e, dyn, scale, 1)
    print(f"  {'true_fwd_1step':<18}{mean_score(clean, tf1):>10.4f}"
          f"{mean_score(obs_eps, tf1):>10.4f}{mean_score(act_eps, tf1):>10.4f}")
    print(f"  {'pedm_1step':<18}{mean_score(clean, pedm_scores):>10.4f}"
          f"{mean_score(obs_eps, pedm_scores):>10.4f}{mean_score(act_eps, pedm_scores):>10.4f}")
    print("\nAUROC ~0.5 = detector blind to that channel; ~1.0 = fully detects.")
    print("Compare true_fwd (zero noise floor) vs pedm (learned, realistic floor).")


if __name__ == "__main__":
    main()
