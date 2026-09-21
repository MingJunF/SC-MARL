"""Compute a robust, per-dimension "clean natural variability" scale for the victim's augmented
observation (2026-09-20, per user's distribution-aware concealment parameterization design).

Root cause this addresses: Concealment's obs-forgery perturbation was scaled by `scale_full`
(the VICTIM's own observation normalizer std, from training the victim policy) -- a reasonable-
looking but WRONG reference. Direct inspection showed PEDM's own training-data variance for
goal_lidar bins 6/7 is tiny (~0.002-0.004) compared to neighboring bins (~0.02-0.25) -- a natural
data artifact (the goal rarely fell in those two angular bins during clean data collection), NOT
something `scale_full` has any way to know about, since it reflects a completely different
statistic (the victim's own observation normalization, unrelated to which dimensions are "boring"
across an actual clean SafetyPointGoal1 trajectory). A uniform-by-scale_full perturbation treats
a 0.01 move on a near-constant-zero dimension the same as a 0.01 move on a highly-variable one --
but only the former looks bizarre relative to what a REAL clean trajectory ever does.

This script estimates that "how much does this dimension naturally move across clean episodes"
scale directly from clean reference rollouts (detector-agnostic by design -- it never reads
PEDM's own internal stats, only the environment's own clean behavior, so the same scale would
apply unchanged if PEDM were swapped for COTD, a trajectory classifier, or anything else).

Uses a robust (MAD-based) scale estimator, not raw std, since exactly the dimensions this is
meant to protect (near-constant, low-coverage ones) are also the ones where a plain std estimate
is least reliable/most prone to fluke variance from a handful of samples.

Usage:
    python -m examples.compute_clean_obs_scale --n_episodes 64 --out results/safety_gym/SafetyPointGoal1/clean_obs_robust_scale.npy
"""
import argparse

import numpy as np
import gymnasium as gym

import safety_gymnasium  # noqa: F401
import examples.safety_hazard_patch  # noqa: F401
import examples.safety_layout_patch  # noqa: F401
from examples.train_safety_attacker import SafetyVictim

VICTIM_CKPT = "results/safety_gym/SafetyPointGoal1/victim/victim_v23_backup_41pct.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, default="SafetyPointGoal1Gymnasium-v0")
    ap.add_argument("--n_episodes", type=int, default=64)
    ap.add_argument("--ep_len", type=int, default=1000)
    ap.add_argument("--sigma_floor", type=float, default=0.01,
                     help="Absolute floor for the robust scale (in raw observation units). "
                          "2026-09-20: switched from a median-relative floor after discovering "
                          "49/64 dims have EXACTLY-zero MAD (>50%% of samples equal the exact "
                          "median, typically 0.0, for many lidar bins) -- a median-relative floor "
                          "degenerates to 0 when the median itself is 0. A small fixed absolute "
                          "floor avoids this regardless of how many dims collapse.")
    ap.add_argument("--out", type=str,
                     default="results/safety_gym/SafetyPointGoal1/clean_obs_robust_scale.npy")
    args = ap.parse_args()

    device = "cpu"
    victim = SafetyVictim(VICTIM_CKPT, device)
    full_dim = victim.full_obs_dim

    all_obs = []
    for i in range(args.n_episodes):
        env = gym.make(args.scenario)
        o, _ = env.reset(seed=20000 + i)
        steps_since_goal = 0
        for t in range(args.ep_len):
            o_aug = victim._augment(o.astype(np.float32), steps_since_goal, env)  # noqa: SLF001
            all_obs.append(o_aug.copy())
            a_v = victim.act_from_augmented(o_aug)
            env_action = np.clip(a_v, -1.0, 1.0)
            no, r, term, trunc, info = env.step(env_action)
            steps_since_goal = 0 if info.get("goal_met", False) else steps_since_goal + 1
            o = no
            if term or trunc:
                break
        env.close()
        if (i + 1) % 16 == 0:
            print(f"  collected {i + 1}/{args.n_episodes} clean episodes...")

    X = np.array(all_obs, dtype=np.float64)  # (n_steps, full_dim)
    print(f"total clean steps collected: {X.shape[0]}")

    # 2026-09-20: percentile-range estimator (q95-q05)/3.29, not MAD -- MAD collapses to EXACTLY
    # 0 whenever >50% of samples equal the exact median (true for 49/64 dims here, mostly lidar
    # bins that read exactly 0.0 most of the time since a single hazard/goal rarely lands in most
    # of the 16 angular bins). The 90th-percentile range still captures "how far the occasional
    # nonzero reading goes" for a mostly-zero dim, as long as >5% of samples are nonzero.
    q05 = np.percentile(X, 5, axis=0)
    q95 = np.percentile(X, 95, axis=0)
    robust_std = (q95 - q05) / 3.29  # matches a Normal's own q95-q05 spread, for comparability

    scale = np.maximum(robust_std, args.sigma_floor)

    print("robust scale (min/median/max):", scale.min(), np.median(scale), scale.max())
    print("floor applied (absolute):", args.sigma_floor)
    n_floored = int((robust_std < args.sigma_floor).sum())
    print(f"{n_floored}/{full_dim} dims were floored (had near-zero natural variability)")

    np.save(args.out, scale.astype(np.float32))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
