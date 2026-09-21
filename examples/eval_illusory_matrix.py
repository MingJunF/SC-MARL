"""Generalized (parameterized) PEDM/CUSUM eval for single-agent illusory obs-attacker
checkpoints saved by `train_obs_attacker.py` (2026-09-21) -- used for the multi-seed,
multi-env (HalfCheetah/Hopper) experiment matrix. Same methodology as
`eval_mujoco_marl_vs_pedm.py` (same victim, same PEDM detector, same CUSUM helper) so results
are directly comparable across the role-split and single-agent baselines.

Usage:
    python -m examples.eval_illusory_matrix --ckpt <path to attacker_obs_illusory.pt> \
        --scenario HalfCheetah-v4 --victim_run <victim run dir> \
        --pedm_ckpt results/obs_attackers/HalfCheetah-v4/pedm_detector.pt --episodes 15
"""
import argparse

import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Box

from examples.eval_robust_detector import Victim
from examples.train_obs_attacker import AntTrueDynamics, project_budget
from harl.attacks.adversary_ac import AdversaryActorCritic
from harl.detectors.pedm_detector import PEDMDetector
from examples.eval_mujoco_marl_vs_pedm import cusum_stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument("--victim_run", type=str, required=True)
    ap.add_argument("--pedm_ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--quantile", type=float, default=0.97)
    args = ap.parse_args()

    device = "cpu"
    probe = gym.make(args.scenario)
    obs_dim = int(probe.observation_space.shape[0])
    act_dim = int(probe.action_space.shape[0])
    lo, hi = probe.action_space.low.astype(np.float32), probe.action_space.high.astype(np.float32)
    probe.close()

    act_space = Box(lo, hi, (act_dim,), np.float32)
    victim = Victim(args.victim_run, obs_dim, act_space, device)
    dyn = AntTrueDynamics(args.scenario)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    budget = float(ck["budget"])
    budget_norm = ck.get("budget_norm", "l2")
    in_dim, out_dim = int(ck["in_dim"]), int(ck["out_dim"])
    obs_scale = np.asarray(ck["obs_scale"], np.float32)
    print(f"loaded {args.ckpt}: budget({budget_norm})={budget} in_dim={in_dim} out_dim={out_dim}")
    net = AdversaryActorCritic(in_dim, out_dim, hidden_size=256, activation="tanh")
    net.load_state_dict(ck["state_dict"])
    net.eval()

    pedm = PEDMDetector(obs_dim=obs_dim, action_dim=act_dim, n_part=100, device="cpu")
    pedm.load(args.pedm_ckpt)

    scale = obs_scale

    def run(seed, attack, max_steps=1000):
        env = gym.make(args.scenario)
        o, _ = env.reset(seed=seed)
        o = np.asarray(o, np.float32)
        victim.reset()
        last_pert = o.copy()
        last_vact = np.zeros(act_dim, np.float32)
        obs_list, act_list = [o.copy()], []
        ret = 0.0
        for t in range(max_steps):
            expected = dyn.next_obs(last_pert, last_vact)
            if attack:
                adv_in = np.concatenate([o / scale, last_pert / scale, expected / scale]).astype(np.float32)
                with torch.no_grad():
                    delta, _, _ = net.act(torch.as_tensor(adv_in[None]), deterministic=True)
                phi_n = project_budget(delta.numpy(), budget, scale, budget_norm)[0]
                obs_vic = o + phi_n
            else:
                obs_vic = o.copy()
            a_vic = np.clip(victim.act(obs_vic), lo, hi)
            no, r, term, trunc, info = env.step(a_vic)
            ret += r
            no = np.asarray(no, np.float32)

            obs_list.append(obs_vic.astype(np.float32))
            act_list.append(a_vic.astype(np.float32))
            last_pert, last_vact = obs_vic, a_vic
            o = no
            if term or trunc:
                break
        env.close()
        obs_seq = np.array(obs_list[1:], np.float32)
        act_seq = np.array(act_list[:-1], np.float32)
        return obs_seq, act_seq, ret

    clean_scores, clean_rets = [], []
    for i in range(args.episodes):
        o_seq, a_seq, ret = run(4000 + i, attack=False)
        clean_scores.append(pedm.predict_scores(o_seq, a_seq))
        clean_rets.append(ret)
    threshold = pedm.calibrate_threshold(clean_scores, quantile=args.quantile)
    clean_or = np.array([float(np.max(s)) for s in clean_scores])
    or_tau = float(np.quantile(clean_or, args.quantile))
    clean_flat = np.concatenate([np.ravel(s) for s in clean_scores])
    mu_clean, sigma_clean = float(clean_flat.mean()), float(clean_flat.std() + 1e-8)
    clean_cusum_peaks = np.array([cusum_stat(s, mu_clean, sigma_clean).max() for s in clean_scores])
    cusum_h = float(np.quantile(clean_cusum_peaks, args.quantile))

    rets, per_step, or_scores, cusum_delays = [], [], [], []
    for i in range(args.episodes):
        o_seq, a_seq, ret = run(9000 + i, attack=True)
        scores = pedm.predict_scores(o_seq, a_seq)
        per_step.append(float(np.mean(scores)))
        or_scores.append(float(np.max(scores)))
        rets.append(ret)
        s_t = cusum_stat(scores, mu_clean, sigma_clean)
        crossed = np.nonzero(s_t > cusum_h)[0]
        cusum_delays.append(int(crossed[0]) if len(crossed) else None)

    n_flag_or = sum(1 for s in or_scores if s > or_tau)
    n_flag_cusum = sum(1 for d in cusum_delays if d is not None)
    print(f"clean: mean_ret={np.mean(clean_rets):.2f} OR_tau={or_tau:.4f} cusum_h={cusum_h:.2f}")
    print(f"attacked: mean_ret={np.mean(rets):.2f} per-step={np.mean(per_step):.4f} "
          f"OR={np.mean(or_scores):.4f} OR_flagged={n_flag_or}/{args.episodes} "
          f"CUSUM_flagged={n_flag_cusum}/{args.episodes}")
    print(f"return drop: {np.mean(clean_rets) - np.mean(rets):.2f}")


if __name__ == "__main__":
    main()
