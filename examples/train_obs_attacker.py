"""Train observation-space attackers on a frozen Ant victim (paper-faithful).

Two attackers, both with a per-step L2 observation-perturbation budget B
(applied in max-abs-normalized observation space, following Franzmeyer et al.,
ICLR 2024, App. A.7):

  * ``ppo``      - standard SA-RL / SA-MDP-style attacker: reward = -victim_reward.
                    Effective but statistically detectable.
  * ``illusory`` - epsilon-illusory attacker trained with the paper's dual-ascent
                    objective: reward = -victim_reward - lambda * ||o_t - p(o_{t-1},
                    a_{t-1})||^2, where p is the TRUE environment transition
                    function (reconstructed from MuJoCo state). lambda is updated
                    by dual ascent toward a detectability (KL) target eps_kl.

Both attackers are trained with PPO against the frozen HARL victim
``actor_agent0.pt``. Perturbations are bounded by ``--budget`` (L2, normalized).

Example:
    python examples/train_obs_attacker.py --mode both \
        --victim_run results/robust_victim/Ant-v4/mappo/victim6m/seed-00001-* \
        --budget 0.2 --num_env_steps 500000
"""
import argparse
import glob
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.attacks.adversary_ac import AdversaryActorCritic
from examples.eval_robust_detector import Victim


# --------------------------------------------------------------------------- #
#  True Ant dynamics p(o, a) via MuJoCo state reconstruction                    #
# --------------------------------------------------------------------------- #
class AntTrueDynamics:
    """One-step true transition p(o, a) via obs<->state reconstruction.

    Works for MuJoCo locomotion envs whose obs = [qpos[k:], qvel], i.e. the first
    ``k`` position coordinates are excluded from the observation (Ant: x,y -> k=2;
    HalfCheetah/Hopper/Walker: x -> k=1). ``k`` and the split are inferred from the
    env's nq/nv and obs dim, so no per-env hardcoding is needed.
    """

    def __init__(self, scenario="Ant-v4"):
        self.sim = gym.make(scenario).unwrapped
        self.sim.reset(seed=0)
        self.nq = int(self.sim.model.nq)
        self.nv = int(self.sim.model.nv)
        obs_dim = int(self.sim.observation_space.shape[0])
        self.pos_len = obs_dim - self.nv     # qpos entries kept in the obs
        self.k = self.nq - self.pos_len      # leading position coords excluded
        assert self.pos_len > 0 and 0 <= self.k <= self.nq, (
            f"cannot reconstruct state for {scenario}: nq={self.nq} nv={self.nv} obs={obs_dim}")

    def next_obs(self, obs, action):
        qpos = np.concatenate([np.zeros(self.k, np.float64), obs[:self.pos_len]])
        qvel = obs[self.pos_len:self.pos_len + self.nv]
        self.sim.set_state(qpos, qvel)
        self.sim.do_simulation(action, self.sim.frame_skip)
        return self.sim._get_obs().astype(np.float32)


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #
def victim_act_batch(victim, obs_batch):
    """Deterministic victim actions for a batch of observations."""
    n = obs_batch.shape[0]
    rnn = np.zeros((n, victim.recurrent_n, victim.hidden), np.float32)
    masks = np.ones((n, 1), np.float32)
    with torch.no_grad():
        actions, _, _ = victim.actor(
            obs_batch.astype(np.float32), rnn, masks, deterministic=True
        )
    return actions.detach().cpu().numpy().astype(np.float32)


def l2_project(delta, budget):
    """Project rows of ``delta`` onto the L2 ball of radius ``budget``."""
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    factor = np.minimum(1.0, budget / (norm + 1e-8))
    return delta * factor


def linf_project(delta, budget):
    """Per-dimension tanh-squash onto the L-infinity ball of radius ``budget`` (2026-09-20,
    added for a direct ablation -- "illusory方法也得改成L无限的budget跑一次" -- against the
    role-split MARL line's own L-infinity/tanh-squash geometry, see
    `harl/envs/mujoco_marl/mujoco_marl_env.py`. Unlike `l2_project`, has no flat/zero-gradient
    saturation region (ATLA-inspired, same mechanism used throughout this project's later
    HARL-native environments)."""
    return budget * np.tanh(delta)


def project_budget(delta, budget, scale, norm="l2"):
    """Returns the FINAL physical-units perturbation (already de-normalized by `scale`) --
    callers should NOT multiply the result by `scale` again.

    2026-09-21 bug fix (user-caught, "应该是有一套方法使用多个地图才对啊，一个地图一个限制方式
    也太奇怪了" -- budget must have a consistent, environment-agnostic physical meaning, not one
    that silently depends on each environment's own `scale` distribution): the L2 path used to
    project `delta` onto a `budget`-radius ball FIRST (in raw/normalized units) and only
    de-normalize by `scale` AFTERWARD (`l2_project(delta, budget) * scale`) -- since the final
    physical L2 norm is then `||l2_project(delta,budget) * scale||`, which depends entirely on
    `scale`'s own magnitude/spread, "budget=0.2" ended up meaning wildly different ACTUAL
    physical perturbation sizes across environments (confirmed empirically: HalfCheetah's
    concealer-only forgery stayed near budget, but Hopper's -- whose obs_scale ranges 0.22 to
    10.0, a 45x spread dominated by a few large dims -- measured L2 norm 1.11-1.31 in physical
    units, 5-6x over the nominal 0.2). Fixed by reordering: scale into physical units FIRST
    (`delta * scale`, still preserving each dim's relative natural-variability weighting), THEN
    cap the TOTAL L2 norm of that already-physical vector to `budget` -- so `budget` now bounds
    the real physical perturbation size EXACTLY and IDENTICALLY regardless of environment.
    The L-infinity path is unaffected (each dimension already gets its own independent
    `budget*scale[dim]` cap by design -- that per-dimension proportionality was always the
    intended behavior, not a portability bug, since L-infinity never lets budget "leak" across
    dimensions the way the L2 ball's cross-dimension coupling did)."""
    if norm == "linf":
        return budget * np.tanh(delta) * scale
    return l2_project(delta * scale, budget)


def resolve_victim_run(pattern):
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"no victim run matched {pattern}")
    return matches[-1]


def estimate_obs_scale(scenario, victim, act_low, act_high, n_steps=6000, seed=7):
    """Max-abs observation scale from clean victim rollouts (paper normalization)."""
    env = gym.make(scenario)
    obs, _ = env.reset(seed=seed)
    victim.reset()
    acc = np.abs(np.asarray(obs, np.float32))
    for t in range(n_steps):
        a = np.clip(victim.act(obs), act_low, act_high)
        obs, _, term, trunc, _ = env.step(a)
        obs = np.asarray(obs, np.float32)
        acc = np.maximum(acc, np.abs(obs))
        if term or trunc:
            obs, _ = env.reset()
            victim.reset()
    env.close()
    return np.maximum(acc, 1e-3).astype(np.float32)


def estimate_scales(scenario, victim, act_low, act_high, n_steps=6000, seed=7):
    """Max-abs observation AND victim-action scales from clean rollouts."""
    env = gym.make(scenario)
    obs, _ = env.reset(seed=seed)
    victim.reset()
    o_acc = np.abs(np.asarray(obs, np.float32))
    a_acc = np.zeros_like(act_low)
    for t in range(n_steps):
        a = np.clip(victim.act(obs), act_low, act_high)
        a_acc = np.maximum(a_acc, np.abs(a))
        obs, _, term, trunc, _ = env.step(a)
        obs = np.asarray(obs, np.float32)
        o_acc = np.maximum(o_acc, np.abs(obs))
        if term or trunc:
            obs, _ = env.reset()
            victim.reset()
    env.close()
    return (np.maximum(o_acc, 1e-3).astype(np.float32),
            np.maximum(a_acc, 1e-3).astype(np.float32))


# --------------------------------------------------------------------------- #
#  Training                                                                      #
# --------------------------------------------------------------------------- #
def train(mode, args, victim, obs_scale, act_scale, device):
    scenario = args.scenario
    n_envs = args.n_envs
    T = args.rollout_len
    obs_dim = obs_scale.shape[0]
    channel = args.attack  # "obs" or "act"

    envs = [gym.make(scenario) for _ in range(n_envs)]
    dyn = AntTrueDynamics(scenario)
    act_dim = envs[0].action_space.shape[0]
    act_low = envs[0].action_space.low.astype(np.float32)
    act_high = envs[0].action_space.high.astype(np.float32)
    ascale = act_scale[None, :]

    illu_in = args.illusory_input  # 'privileged' (feeds true-dyn expected) or 'history' (paper-faithful)
    if channel == "obs":
        if mode != "illusory":
            in_dim = obs_dim
        elif illu_in == "history":
            in_dim = 2 * obs_dim + act_dim   # [s, o_rep_prev, a_prev] (no privileged expected)
        else:
            in_dim = 3 * obs_dim             # [s, o_rep_prev, expected]
        out_dim = obs_dim
    else:  # act channel: input [s, a_v] (+ expected if illusory privileged), output delta_a
        if mode != "illusory":
            in_dim = obs_dim + act_dim
        elif illu_in == "history":
            in_dim = obs_dim + act_dim       # [s, a_v] (penalty-only, no expected)
        else:
            in_dim = obs_dim + act_dim + obs_dim
        out_dim = act_dim
    net = AdversaryActorCritic(in_dim, out_dim, hidden_size=256, activation="tanh").to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    lam = args.lambda_init if mode == "illusory" else 0.0
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        if mode == "illusory" and "lambda" in ck:
            lam = float(ck["lambda"])  # resume dual-ascent multiplier
        print(f"[{channel}_{mode}] resumed weights from {args.init_from} (lambda={lam:.3f})")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            entity=args.wandb_entity or None,
            name=f"{channel}_{mode}_b{args.budget}_seed{args.seed}",
            job_type=f"{channel}_{mode}",
            config={**vars(args), "attacker_mode": mode, "channel": channel},
            reinit=True,
        )

    # per-env state
    obs = np.zeros((n_envs, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=args.seed + i)
        obs[i] = o
    victim.reset()  # victim is MLP: rnn unused, batched calls are stateless
    last_pert = obs.copy()
    last_vact = np.zeros((n_envs, envs[0].action_space.shape[0]), np.float32)
    last_done = np.zeros(n_envs, np.float32)

    scale = obs_scale[None, :]
    num_updates = int(args.num_env_steps // (T * n_envs))
    recent_penalty = []
    ret_hist = deque(maxlen=100)
    start = time.time()

    for update in range(1, num_updates + 1):
        buf_obs = np.zeros((T, n_envs, in_dim), np.float32)
        buf_act = np.zeros((T, n_envs, out_dim), np.float32)
        buf_logp = np.zeros((T, n_envs), np.float32)
        buf_val = np.zeros((T, n_envs), np.float32)
        buf_rew = np.zeros((T, n_envs), np.float32)
        buf_done = np.zeros((T, n_envs), np.float32)
        ep_ret = np.zeros(n_envs, np.float32)
        returns_log, pen_log = [], []

        for t in range(T):
            # expected next obs from TRUE dynamics p(last_pert, last_vact)
            expected = np.zeros((n_envs, obs_dim), np.float32)
            for i in range(n_envs):
                expected[i] = dyn.next_obs(last_pert[i], last_vact[i])
            mask = last_done[:, None]
            expected = expected * (1 - mask) + obs * mask
            pert_prev = last_pert * (1 - mask) + obs * mask
            a_prev = last_vact * (1 - mask)   # victim's previous action (0 on reset)

            obs_n = obs / scale
            if channel == "obs":
                if mode == "illusory":
                    if illu_in == "history":
                        adv_in = np.concatenate(
                            [obs_n, pert_prev / scale, a_prev], axis=1
                        ).astype(np.float32)
                    else:
                        adv_in = np.concatenate(
                            [obs_n, pert_prev / scale, expected / scale], axis=1
                        ).astype(np.float32)
                else:
                    adv_in = obs_n.astype(np.float32)
                with torch.no_grad():
                    a_t, logp_t, val_t = net.act(torch.as_tensor(adv_in, device=device))
                delta = a_t.cpu().numpy()  # normalized perturbation
                phi_n = project_budget(delta, args.budget, scale, args.budget_norm)
                obs_vic = obs + phi_n  # already in physical units
                a_vic = np.clip(victim_act_batch(victim, obs_vic), act_low, act_high)
                env_action = a_vic
                new_last_pert, new_last_vact = obs_vic, a_vic
            else:  # act channel: victim sees the TRUE obs; attacker perturbs the action
                a_vic = np.clip(victim_act_batch(victim, obs), act_low, act_high)
                if mode == "illusory" and illu_in != "history":
                    adv_in = np.concatenate([obs_n, a_vic, expected / scale], axis=1).astype(np.float32)
                else:
                    adv_in = np.concatenate([obs_n, a_vic], axis=1).astype(np.float32)
                with torch.no_grad():
                    a_t, logp_t, val_t = net.act(torch.as_tensor(adv_in, device=device))
                delta = a_t.cpu().numpy()  # normalized action perturbation
                phi_a = project_budget(delta, args.budget, ascale, args.budget_norm)
                env_action = np.clip(a_vic + phi_a, act_low, act_high)
                obs_vic = obs  # detector/dynamics observe the true (unperturbed) obs
                new_last_pert, new_last_vact = obs, a_vic

            # detectability penalty: ||o_vic - expected||^2 in normalized space
            penalty = np.sum(((obs_vic - expected) / scale) ** 2, axis=1)
            penalty = penalty * (1 - last_done)

            next_obs = np.zeros_like(obs)
            rew = np.zeros(n_envs, np.float32)
            done = np.zeros(n_envs, np.float32)
            for i in range(n_envs):
                no, r, term, trunc, _ = envs[i].step(env_action[i])
                d = term or trunc
                ep_ret[i] += r
                if d:
                    returns_log.append(ep_ret[i]); ep_ret[i] = 0.0
                    no, _ = envs[i].reset()
                next_obs[i] = no
                rew[i] = r
                done[i] = float(d)

            adv_reward = -rew - lam * penalty

            buf_obs[t] = adv_in
            buf_act[t] = delta
            buf_logp[t] = logp_t.cpu().numpy()
            buf_val[t] = val_t.cpu().numpy()
            buf_rew[t] = adv_reward
            buf_done[t] = done
            pen_log.append(penalty[last_done == 0].mean() if np.any(last_done == 0) else 0.0)

            last_pert = new_last_pert
            last_vact = new_last_vact
            last_done = done
            obs = next_obs

        # bootstrap value
        obs_n = obs / scale
        expected = np.zeros((n_envs, obs_dim), np.float32)
        for i in range(n_envs):
            expected[i] = dyn.next_obs(last_pert[i], last_vact[i])
        if channel == "obs":
            if mode == "illusory":
                if illu_in == "history":
                    adv_in = np.concatenate([obs_n, last_pert / scale, last_vact], axis=1)
                else:
                    adv_in = np.concatenate([obs_n, last_pert / scale, expected / scale], axis=1)
            else:
                adv_in = obs_n
        else:  # act
            a_v_boot = np.clip(victim_act_batch(victim, obs), act_low, act_high)
            if mode == "illusory" and illu_in != "history":
                adv_in = np.concatenate([obs_n, a_v_boot, expected / scale], axis=1)
            else:
                adv_in = np.concatenate([obs_n, a_v_boot], axis=1)
        with torch.no_grad():
            last_val = net.get_value(
                torch.as_tensor(adv_in.astype(np.float32), device=device)
            ).cpu().numpy()

        # GAE
        adv = np.zeros((T, n_envs), np.float32)
        gae = np.zeros(n_envs, np.float32)
        nv = last_val
        for t in reversed(range(T)):
            nonterm = 1.0 - buf_done[t]
            d = buf_rew[t] + args.gamma * nv * nonterm - buf_val[t]
            gae = d + args.gamma * args.gae_lambda * nonterm * gae
            adv[t] = gae
            nv = buf_val[t]
        ret = adv + buf_val

        # PPO update
        b_obs = torch.as_tensor(buf_obs.reshape(T * n_envs, in_dim), device=device)
        b_act = torch.as_tensor(buf_act.reshape(T * n_envs, out_dim), device=device)
        b_logp = torch.as_tensor(buf_logp.reshape(-1), device=device)
        b_adv = torch.as_tensor(adv.reshape(-1), device=device)
        b_ret = torch.as_tensor(ret.reshape(-1), device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        bs = T * n_envs
        mb = bs // args.num_mini_batch
        ploss_acc, vloss_acc, ent_acc = [], [], []
        for _ in range(args.ppo_epoch):
            perm = torch.randperm(bs, device=device)
            for s in range(0, bs, mb):
                idx = perm[s:s + mb]
                nlp, ent, val = net.evaluate_actions(b_obs[idx], b_act[idx])
                ratio = torch.exp(nlp - b_logp[idx])
                s1 = ratio * b_adv[idx]
                s2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * b_adv[idx]
                ploss = -torch.min(s1, s2).mean()
                vloss = 0.5 * (val - b_ret[idx]).pow(2).mean()
                loss = ploss + args.vf_coef * vloss - args.ent_coef * ent.mean()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                optim.step()
                ploss_acc.append(ploss.item())
                vloss_acc.append(vloss.item())
                ent_acc.append(ent.mean().item())

        # dual ascent on lambda (illusory only)
        mean_pen = float(np.mean(pen_log)) if pen_log else 0.0
        if mode == "illusory":
            recent_penalty.append(mean_pen)
            d_kl = float(np.mean(recent_penalty[-10:]))
            lam = max(lam + args.alpha_lambda * (d_kl - args.eps_kl), 0.0)

        ret_hist.extend(returns_log)
        steps = update * T * n_envs
        ret_mean = float(np.mean(ret_hist)) if ret_hist else float("nan")
        if run is not None:
            run.log({
                "victim_return": ret_mean,
                "detectability_penalty": mean_pen,
                "lambda": lam,
                "policy_loss": float(np.mean(ploss_acc)),
                "value_loss": float(np.mean(vloss_acc)),
                "entropy": float(np.mean(ent_acc)),
                "update": update,
            }, step=steps)

        if update % args.log_interval == 0 or update == 1:
            sps = int(steps / (time.time() - start))
            print(f"[{channel}_{mode}] upd {update}/{num_updates} step {steps} "
                  f"victim_ret {ret_mean:.1f} penalty {mean_pen:.4f} "
                  f"lambda {lam:.3f} ({sps} sps)")

    for e in envs:
        e.close()
    if run is not None:
        run.finish()

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"attacker_{channel}_{mode}.pt")
    torch.save({"state_dict": net.state_dict(), "mode": mode, "channel": channel,
                "in_dim": in_dim, "out_dim": out_dim, "obs_dim": obs_dim,
                "budget": args.budget, "budget_norm": args.budget_norm, "lambda": lam,
                "illusory_input": args.illusory_input,
                "obs_scale": obs_scale, "act_scale": act_scale}, path)
    print(f"[{channel}_{mode}] saved -> {path}")
    return path


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["ppo", "illusory", "both"], default="both")
    p.add_argument("--attack", choices=["obs", "act"], default="obs",
                   help="Attack channel: perturb the observation or the action.")
    p.add_argument("--scenario", default="Ant-v4")
    p.add_argument("--victim_run", default="")
    p.add_argument("--budget", type=float, default=0.2,
                   help="Per-step perturbation budget (normalized obs space; L2-ball radius or "
                        "L-infinity per-dim cap, depending on --budget_norm).")
    p.add_argument("--budget_norm", choices=["l2", "linf"], default="l2",
                   help="l2 (default, original): project onto an L2 ball via l2_project. "
                        "linf (2026-09-20 ablation, ATLA-style tanh-squash): per-dim cap, no "
                        "flat/zero-gradient saturation region -- matches the role-split MARL "
                        "line's own geometry (harl/envs/mujoco_marl/mujoco_marl_env.py), for a "
                        "direct comparison isolating norm choice from architecture.")
    p.add_argument("--num_env_steps", type=int, default=4000000)
    p.add_argument("--init_from", default="", help="resume attacker weights (and illusory lambda) from this checkpoint")
    p.add_argument("--illusory_input", choices=["privileged", "history"], default="privileged",
                   help="privileged: feed true-dyn expected to actor (released-code style); "
                        "history: [s, o_rep_prev, a_prev], p used only as training penalty (paper-faithful)")
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--rollout_len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--ppo_epoch", type=int, default=4)
    p.add_argument("--num_mini_batch", type=int, default=4)
    # illusory dual-ascent params (paper App. A.7)
    p.add_argument("--lambda_init", type=float, default=10.0)
    p.add_argument("--alpha_lambda", type=float, default=0.1)
    p.add_argument("--eps_kl", type=float, default=0.1,
                   help="Detectability (KL) target for dual ascent.")
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_dir", default="results/obs_attackers/Ant-v4")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    # wandb
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    p.add_argument("--wandb_project", default="illusory_obs_attackers")
    p.add_argument("--wandb_group", default="Ant-v4")
    p.add_argument("--wandb_entity", default="")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    victim_pat = args.victim_run or (
        "results/robust_victim/Ant-v4/mappo/*/seed-00001-*"
    )
    victim_run = resolve_victim_run(victim_pat)
    print(f"victim_run: {victim_run}")

    from gymnasium.spaces import Box
    probe = gym.make(args.scenario)
    obs_dim = probe.observation_space.shape[0]
    act_space = Box(probe.action_space.low, probe.action_space.high,
                    probe.action_space.shape, np.float32)
    act_low, act_high = act_space.low.astype(np.float32), act_space.high.astype(np.float32)
    probe.close()

    victim = Victim(victim_run, obs_dim, act_space, device)
    print("estimating observation + action scales from clean rollouts...")
    obs_scale, act_scale = estimate_scales(args.scenario, victim, act_low, act_high)

    modes = ["ppo", "illusory"] if args.mode == "both" else [args.mode]
    for m in modes:
        train(m, args, victim, obs_scale, act_scale, device)


if __name__ == "__main__":
    main()
