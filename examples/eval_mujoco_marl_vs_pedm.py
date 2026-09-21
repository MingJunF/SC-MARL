"""Independent-detector sanity check for `mujoco_marl` HARL-native checkpoints (2026-09-20).

Training itself only ever scores Concealment's own forged (o_rep, a_rep) record against
`AntTrueDynamics`'s narrative-SELF-CONSISTENCY check (see `mujoco_marl_env.py`'s module
docstring) -- a real but "local" stealth surrogate that never compares against real training
data, only against the deceiver's own prior self-report. This script instead scores that same
(o_rep, a_rep) record with an INDEPENDENT, already-trained PEDM detector
(`results/obs_attackers/HalfCheetah-v4/pedm_detector.pt`, trained on real clean HalfCheetah
transitions, never on this attacker's own forged data) -- the actual "would a real distribution-
based detector flag this" answer, mirroring `eval_act_forgery_vs_pedm.py`'s role for the
safety_marl line.

No saved threshold/.meta exists for this checkpoint, so one is calibrated here from held-out
CLEAN (zero-perturbation) episodes at a given quantile (default 0.97, matching this project's
own established "3%% FPR" convention for HalfCheetah, e.g. `eval_or_cusum_table.py`).

Usage:
    python -m examples.eval_mujoco_marl_vs_pedm --run_dir results/mujoco_marl/HalfCheetah-v4/mappo_alt/harl_native_hc_v7_epscost_fixed --episodes 15
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import gymnasium as gym

from examples.eval_robust_detector import Victim
from harl.algorithms.actors import ALGO_REGISTRY
from harl.detectors.pedm_detector import PEDMDetector
from harl.detectors.cotd_detector import COTDDetector


def build_detector(name, obs_dim, act_dim, ckpt_path):
    """`--detector pedm` (default, deterministic forward-prediction dynamics model) or
    `--detector cotd` (2026-09-22, CVAE-ensemble reconstruction detector with split-conformal
    calibration, reproducing arXiv:2503.05238 -- see harl/detectors/cotd_detector.py's module
    docstring). Both expose the same fit/predict_scores/calibrate_threshold interface, so
    everything downstream (OR/CUSUM calibration and scoring) is unchanged either way."""
    if name == "pedm":
        d = PEDMDetector(obs_dim=obs_dim, action_dim=act_dim, n_part=100, device="cpu")
    elif name == "cotd":
        d = COTDDetector(obs_dim=obs_dim, action_dim=act_dim, device="cpu")
    else:
        raise ValueError(f"unknown --detector: {name}")
    d.load(ckpt_path)
    return d


def cusum_stat(scores, mu, sigma, k=0.5):
    """One-sided CUSUM on standardized per-step scores: S_t = max(0, S_{t-1} + z_t - k),
    z_t = (score_t - mu) / sigma. `k` (in standardized units, default 0.5 = "half-sigma
    allowance") is the classic Page (1954) CUSUM slack constant -- large enough that clean noise
    doesn't accumulate indefinitely, small enough to stay sensitive to a persistent small shift
    an OR/single-threshold test would miss (2026-09-20, per explicit user follow-up "cusum也是
    吗" -- an OR/episode-max verdict is a WEAKER test than CUSUM: this project's own earlier
    HalfCheetah illusory-attacker finding was OR-det=0.000 but CUSUM-det=1.00, i.e. it evaded
    the single-threshold check while still accumulating a detectable persistent drift)."""
    z = (np.asarray(scores, np.float64) - mu) / sigma
    s = 0.0
    out = np.empty_like(z)
    for i, zi in enumerate(z):
        s = max(0.0, s + zi - k)
        out[i] = s
    return out


def latest_seed_dir(run_dir):
    if os.path.isdir(os.path.join(run_dir, "models")):
        return run_dir
    candidates = sorted(glob.glob(os.path.join(run_dir, "seed-*")))
    if not candidates:
        raise FileNotFoundError(f"no seed-* subdirectory under {run_dir}")
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--quantile", type=float, default=0.97)
    ap.add_argument("--pedm_ckpt", type=str, default=None,
                     help="Override the detector checkpoint path; defaults to "
                          "results/obs_attackers/<scenario>/<pedm|cotd>_detector.pt")
    ap.add_argument("--detector", type=str, default="pedm", choices=["pedm", "cotd"],
                     help="pedm (default, deterministic forward-prediction) or cotd "
                          "(CVAE-ensemble reconstruction detector, arXiv:2503.05238)")
    args = ap.parse_args()

    seed_dir = latest_seed_dir(args.run_dir)
    with open(os.path.join(seed_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    main_args, algo_args, env_args = config["main_args"], config["algo_args"], config["env_args"]

    device = torch.device("cpu")
    scenario = env_args.get("scenario", "HalfCheetah-v4")
    detector_ckpt = args.pedm_ckpt or f"results/obs_attackers/{scenario}/{args.detector}_detector.pt"
    probe = gym.make(scenario)
    obs_dim = int(probe.observation_space.shape[0])
    act_dim = int(probe.action_space.shape[0])
    lo, hi = probe.action_space.low.astype(np.float32), probe.action_space.high.astype(np.float32)
    probe.close()

    from gymnasium.spaces import Box
    act_space = Box(lo, hi, (act_dim,), np.float32)
    victim = Victim(env_args["victim_run"], obs_dim, act_space, device)

    disruptor_eps = float(env_args.get("disruptor_eps", 0.4))
    hidden_eps = float(env_args.get("hidden_eps", 0.2))
    hidden_act_eps = float(env_args.get("hidden_act_eps", hidden_eps))
    budget_norm = env_args.get("budget_norm", "linf")
    from examples.train_obs_attacker import estimate_scales, project_budget
    obs_scale_path = env_args.get("obs_scale_path")
    if obs_scale_path:
        obs_scale = np.load(obs_scale_path).astype(np.float32)
        _, act_scale = estimate_scales(scenario, victim, lo, hi)
    else:
        obs_scale, act_scale = estimate_scales(scenario, victim, lo, hi)

    detector = build_detector(args.detector, obs_dim, act_dim, detector_ckpt)

    obs_pad = act_pad = obs_dim + act_dim
    obs_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_pad,), dtype=np.float32)
    act_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(act_pad,), dtype=np.float32)

    actors = []
    for agent_id in range(2):
        agent = ALGO_REGISTRY[main_args["algo"]](
            {**algo_args["model"], **algo_args["algo"]}, obs_box, act_box, device=device,
        )
        ckpt_path = os.path.join(seed_dir, "models", f"actor_agent{agent_id}.pt")
        agent.actor.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        agent.prep_rollout()
        actors.append(agent)

    recurrent_n = algo_args["model"]["recurrent_n"]
    rnn_hidden_size = algo_args["model"]["hidden_sizes"][-1]

    @torch.no_grad()
    def act(agent_id, obs, rnn_state, mask):
        a, rnn_state = actors[agent_id].act(obs[None], rnn_state, mask, None, deterministic=True)
        return a.cpu().numpy()[0], rnn_state

    def run(seed, attack, max_steps=1000):
        env = gym.make(scenario)
        o, _ = env.reset(seed=seed)
        o = o.astype(np.float32)
        victim.reset()
        rnn = [np.zeros((1, recurrent_n, rnn_hidden_size), np.float32) for _ in range(2)]
        mask = [np.ones((1, 1), np.float32) for _ in range(2)]
        a_v = np.clip(victim.act(o), lo, hi)
        obs_list, act_list = [o.copy()], []
        ret = 0.0
        for t in range(max_steps):
            local_obs = np.concatenate([o, a_v]).astype(np.float32)
            if attack:
                raw_hid, rnn[1] = act(1, local_obs, rnn[1], mask[1])
                raw_dis, rnn[0] = act(0, local_obs, rnn[0], mask[0])
                phi_hid_obs = project_budget(raw_hid[:obs_dim], hidden_eps, obs_scale, budget_norm)
                phi_hid_act = project_budget(raw_hid[obs_dim:obs_dim + act_dim], hidden_act_eps, act_scale, budget_norm)
                phi_dis = project_budget(raw_dis[:act_dim], disruptor_eps, act_scale, budget_norm)
            else:
                phi_hid_obs = np.zeros(obs_dim, np.float32)
                phi_hid_act = np.zeros(act_dim, np.float32)
                phi_dis = np.zeros(act_dim, np.float32)

            o_rep = o + phi_hid_obs
            a_rep = np.clip(a_v + phi_hid_act, lo, hi)
            a_exec = np.clip(a_v + phi_dis, lo, hi)

            act_list.append(a_rep.astype(np.float32))
            obs_list.append(o_rep.astype(np.float32))  # this is the NEXT record's "prev obs"... (see below)

            no, r, term, trunc, info = env.step(a_exec)
            ret += r
            no = no.astype(np.float32)
            a_v = np.clip(victim.act(no), lo, hi)
            o = no
            if term or trunc:
                break
        env.close()
        # obs_list currently holds [o_0 (real), o_rep_0, o_rep_1, ..., o_rep_{T-1}] -- the deceiver's
        # own forged record is o_rep_0..o_rep_{T-1} paired with a_rep_0..a_rep_{T-1} (act_list[i]
        # causes the transition obs_list[i] -> obs_list[i+1] in this indexing once the leading
        # real o_0 is dropped).
        obs_seq = np.array(obs_list[1:], np.float32)
        act_seq = np.array(act_list[:-1], np.float32)
        return obs_seq, act_seq, ret

    print(f"[1/2] scoring {args.episodes} held-out CLEAN episodes to calibrate threshold...")
    clean_scores, clean_rets = [], []
    for i in range(args.episodes):
        o_seq, a_seq, ret = run(4000 + i, attack=False)
        clean_scores.append(detector.predict_scores(o_seq, a_seq))
        clean_rets.append(ret)
    threshold = detector.calibrate_threshold(clean_scores, quantile=args.quantile)
    clean_or = np.array([float(np.max(s)) for s in clean_scores])
    or_tau = float(np.quantile(clean_or, args.quantile))

    # CUSUM calibration (2026-09-20, per user follow-up "cusum也是吗"): standardize against the
    # CLEAN per-step score distribution, run CUSUM on each clean episode, calibrate h at the
    # same quantile/FPR convention as the OR threshold above (episode-level "did it ever cross").
    clean_flat = np.concatenate([np.ravel(s) for s in clean_scores])
    mu_clean, sigma_clean = float(clean_flat.mean()), float(clean_flat.std() + 1e-8)
    clean_cusum_peaks = np.array([cusum_stat(s, mu_clean, sigma_clean).max() for s in clean_scores])
    cusum_h = float(np.quantile(clean_cusum_peaks, args.quantile))

    print(f"[2/2] scoring {args.episodes} attacked episodes (Disruptor+Concealment, deterministic)...")
    rets, per_step, or_scores, cusum_peaks, cusum_delays = [], [], [], [], []
    for i in range(args.episodes):
        o_seq, a_seq, ret = run(9000 + i, attack=True)
        scores = detector.predict_scores(o_seq, a_seq)
        per_step.append(float(np.mean(scores)))
        or_scores.append(float(np.max(scores)))
        rets.append(ret)
        s_t = cusum_stat(scores, mu_clean, sigma_clean)
        cusum_peaks.append(float(s_t.max()))
        crossed = np.nonzero(s_t > cusum_h)[0]
        cusum_delays.append(int(crossed[0]) if len(crossed) else None)
        print(f"ep{i}: ret={ret:.2f} {args.detector}_per_step={per_step[-1]:.4f} {args.detector}_OR={or_scores[-1]:.4f} "
              f"cusum_peak={cusum_peaks[-1]:.2f}"
              + (f" DETECTED@t={cusum_delays[-1]}" if cusum_delays[-1] is not None else ""))

    n_flag_or = sum(1 for s in or_scores if s > or_tau)
    n_flag_cusum = sum(1 for d in cusum_delays if d is not None)
    delays = [d for d in cusum_delays if d is not None]
    print(f"\n=== {seed_dir} ===")
    print(f"clean:    mean_ret={np.mean(clean_rets):.2f}  per-step_thr(q{args.quantile})={threshold:.4f}  "
          f"OR_tau={or_tau:.4f}  cusum_h(q{args.quantile})={cusum_h:.2f} (mu={mu_clean:.4f}, sigma={sigma_clean:.4f})")
    print(f"attacked: mean_ret={np.mean(rets):.2f}  per-step={np.mean(per_step):.4f}  "
          f"OR={np.mean(or_scores):.4f}  OR_flagged={n_flag_or}/{args.episodes}  "
          f"CUSUM_flagged={n_flag_cusum}/{args.episodes}"
          + (f"  mean_delay={np.mean(delays):.1f} steps" if delays else ""))
    print(f"return drop (attack effect): {np.mean(clean_rets) - np.mean(rets):.2f}")


if __name__ == "__main__":
    main()
