"""Fit a fresh PEDM detector on clean victim rollouts and save it (2026-09-21) -- needed for
envs (e.g. Hopper-v4) that don't already have a saved `pedm_detector.pt` the way
`results/obs_attackers/HalfCheetah-v4/pedm_detector.pt` does. Mirrors the fit flow already
used inline in `examples/eval_robust_detector.py`, factored out as a standalone, reusable
training script that just saves the fitted detector to disk.

Usage:
    python -m examples.train_pedm_detector --scenario Hopper-v4 \
        --victim_run results/robust_victim/Hopper-v4/mappo/victim6m/seed-00001-2026-06-29-12-56-28 \
        --out results/obs_attackers/Hopper-v4/pedm_detector.pt
"""
import argparse

import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Box

from examples.eval_robust_detector import Victim
from harl.detectors.pedm_detector import PEDMDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument("--victim_run", type=str, required=True)
    ap.add_argument("--n_clean_fit", type=int, default=30)
    ap.add_argument("--dyn_epochs", type=int, default=100)
    ap.add_argument("--n_part", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cpu"

    probe = gym.make(args.scenario)
    obs_dim = int(probe.observation_space.shape[0])
    act_dim = int(probe.action_space.shape[0])
    lo, hi = probe.action_space.low.astype(np.float32), probe.action_space.high.astype(np.float32)
    probe.close()

    act_space = Box(lo, hi, (act_dim,), np.float32)
    victim = Victim(args.victim_run, obs_dim, act_space, device)

    def rollout(seed, max_steps=1000):
        env = gym.make(args.scenario)
        o, _ = env.reset(seed=seed)
        o = np.asarray(o, np.float32)
        victim.reset()
        obs_seq = [o.copy()]
        act_seq = []
        for t in range(max_steps):
            a = np.clip(victim.act(o), lo, hi)
            no, r, term, trunc, info = env.step(a)
            no = np.asarray(no, np.float32)
            act_seq.append(a.astype(np.float32))
            obs_seq.append(no.copy())
            o = no
            if term or trunc:
                break
        env.close()
        return np.array(obs_seq, np.float32), np.array(act_seq, np.float32)

    print(f"[1/2] collecting {args.n_clean_fit} clean episodes on {args.scenario}...")
    fit_obs, fit_act, fit_nobs = [], [], []
    for ep in range(args.n_clean_fit):
        o, a = rollout(1000 + ep)
        fit_obs.append(o[:-1])
        fit_act.append(a)
        fit_nobs.append(o[1:])
    fit_obs = np.concatenate(fit_obs)
    fit_act = np.concatenate(fit_act)
    fit_nobs = np.concatenate(fit_nobs)
    print(f"    collected {len(fit_obs)} transitions")

    detector = PEDMDetector(obs_dim=obs_dim, action_dim=act_dim, n_part=args.n_part, device=device)
    print(f"[2/2] fitting PEDM ({args.dyn_epochs} epochs)...")
    detector.fit(fit_obs, fit_act, fit_nobs, n_train_epochs=args.dyn_epochs, verbose=True)

    detector.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
