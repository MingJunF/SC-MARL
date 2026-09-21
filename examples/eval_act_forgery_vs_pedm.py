"""Independent-detector sanity check for HARL-native checkpoints trained WITH the action-forgery
channel (2026-09-19, `hidden_act_eps` in `safety_marl_env.py`) -- e.g. `harl_native_v21_act_forgery`,
`harl_native_v22_act_forgery_normfix`. Supersedes `eval_harl_native_vs_pedm.py` for these
checkpoints ONLY (that script hardcodes act_pad=full_dim=64 and scores PEDM with the honest a_v --
both wrong for a checkpoint trained with the widened act_pad=full_dim+act_dim=66 action space and
the a_rep audit target).

PEDM here audits the DECEIVER'S OWN DOUBLY-FORGED RECORD -- `(o_rep, a_rep)`, where BOTH halves
are fabricated by Concealment (o_rep = obs_aug + phi_hid_obs*scale, a_rep = a_v + phi_hid_act,
clipped by hidden_eps/hidden_act_eps respectively) -- NOT `(o_rep, a_v)` (the pre-2026-09-19
threat model, which paired a fake observation with the HONEST action and left a structural hole:
a_v is fixed by ground truth and Concealment couldn't influence it at all). Real physical
actuation (`a_exec = a_v + phi_dis`, Disruptor's channel) never appears in the audited record,
by design, in either version -- deceiving PEDM is entirely Concealment's job.

Usage:
    python -m examples.eval_act_forgery_vs_pedm --run_dir results/safety_marl/SafetyPointGoal1Gymnasium-v0/mappo_lagr/harl_native_v22_act_forgery_normfix --episodes 15
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import gymnasium as gym

import safety_gymnasium  # noqa: F401
import examples.safety_hazard_patch  # noqa: F401
import examples.safety_layout_patch  # noqa: F401
from examples.train_safety_attacker import SafetyVictim
from harl.algorithms.actors import ALGO_REGISTRY
from harl.detectors.pedm_detector import PEDMDetector

VICTIM_CKPT = "results/safety_gym/SafetyPointGoal1/victim/victim_v23_backup_41pct.pt"
PEDM_CKPT = "results/safety_gym/SafetyPointGoal1/pedm_detector_v23.pt"


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
    args = ap.parse_args()

    seed_dir = latest_seed_dir(args.run_dir)
    with open(os.path.join(seed_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    main_args, algo_args, env_args = config["main_args"], config["algo_args"], config["env_args"]

    device = torch.device("cpu")
    victim = SafetyVictim(env_args["victim_ckpt"], device)
    full_dim = victim.full_obs_dim
    act_dim = 2
    disruptor_eps = float(env_args.get("disruptor_eps", 1.5))
    hidden_eps = float(env_args.get("hidden_eps", 0.2))
    hidden_act_eps = float(env_args.get("hidden_act_eps", 0.2))
    # 2026-09-20: updated to match safety_marl_env.py's current mechanics exactly -- this eval
    # script REIMPLEMENTS the forgery computation standalone (doesn't call the actual env), so it
    # silently goes stale whenever that mechanism changes. Two changes since this script was
    # written: (1) tanh-squash instead of hard clip for all three channels (obs-forgery, action-
    # forgery, disruptor) -- see the ATLA-inspired fix; (2) obs-forgery scaled by
    # `clean_obs_scale` (per-dimension natural clean variability) instead of `scale_full` (the
    # victim's own, unrelated, observation normalizer) -- see the distribution-aware
    # parameterization fix. Scoring a checkpoint trained under the NEW mechanics with the OLD
    # formula here would silently reconstruct the WRONG forged trajectory.
    alpha_obs = float(env_args.get("alpha_obs", 1.0))
    clean_obs_scale_path = env_args.get(
        "clean_obs_scale_path",
        "results/safety_gym/SafetyPointGoal1/clean_obs_robust_scale.npy",
    )
    clean_obs_scale = np.load(clean_obs_scale_path).astype(np.float32)

    pedm = PEDMDetector(obs_dim=60, action_dim=act_dim, n_part=100, device="cpu")
    pedm.load(PEDM_CKPT)
    meta = torch.load(PEDM_CKPT + ".meta", map_location="cpu", weights_only=False)
    THR, OR_THR = float(meta["threshold"]), float(meta["or_tau"])
    print(f"PEDM thresholds: per-step={THR:.4f}  OR={OR_THR:.4f}")

    # 2026-09-19: both obs_pad and act_pad are full_dim+act_dim now -- BOTH agents observe
    # [obs_aug, a_v], and Concealment's action packs [obs-forgery(full_dim), action-forgery(act_dim)].
    obs_pad = act_pad = full_dim + act_dim
    obs_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_pad,), dtype=np.float32)
    act_box = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_pad,), dtype=np.float32)

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

    def run(seed, max_steps=1000):
        env = gym.make(env_args["scenario"])
        lo, hi = env.action_space.low.astype(np.float32), env.action_space.high.astype(np.float32)
        o, _ = env.reset(seed=seed)
        steps_since_goal = 0
        obs_list, act_list = [], []  # (o_rep, a_rep) -- the deceiver's OWN doubly-forged record
        ret, true_cost = 0.0, 0.0
        rnn = [np.zeros((1, recurrent_n, rnn_hidden_size), np.float32) for _ in range(2)]
        mask = [np.ones((1, 1), np.float32) for _ in range(2)]

        obs_aug = victim._augment(o.astype(np.float32), steps_since_goal, env)
        a_v = victim.act_from_augmented(obs_aug)
        for t in range(max_steps):
            local_obs = np.concatenate([obs_aug, a_v]).astype(np.float32)  # both agents see this

            raw_hid, rnn[1] = act(1, local_obs, rnn[1], mask[1])
            u_obs = np.tanh(raw_hid[:full_dim])
            u_act = np.tanh(raw_hid[full_dim:full_dim + act_dim])
            phi_hid_obs = np.clip(alpha_obs * clean_obs_scale * u_obs, -hidden_eps, hidden_eps)
            phi_hid_act = hidden_act_eps * u_act
            o_rep = obs_aug + phi_hid_obs  # forged obs -- never fed to the victim
            a_rep = np.clip(a_v + phi_hid_act, lo, hi)  # forged action -- never fed to a_exec

            obs_list.append(o_rep[:60].astype(np.float32))
            act_list.append(a_rep.astype(np.float32))

            raw_dis, rnn[0] = act(0, local_obs, rnn[0], mask[0])
            phi_dis = disruptor_eps * np.tanh(raw_dis[:act_dim])
            a_exec = np.clip(a_v + phi_dis, lo, hi)

            no, r, term, trunc, info = env.step(a_exec)
            ret += r
            true_cost += float(info.get("cost", 0.0))
            steps_since_goal = 0 if info.get("goal_met", False) else steps_since_goal + 1
            obs_aug = victim._augment(no.astype(np.float32), steps_since_goal, env)
            a_v = victim.act_from_augmented(obs_aug)
            if term or trunc:
                break
        env.close()
        # act_list[i] causes the transition obs_list[i] -> obs_list[i+1]; PEDM requires
        # act_seq exactly one shorter than obs_seq, so the last (dangling) action is dropped.
        return np.array(obs_list, np.float32), np.array(act_list[:-1], np.float32), ret, true_cost

    rets, true_costs, per_step, or_scores = [], [], [], []
    for i in range(args.episodes):
        obs_seq, act_seq, ret, tc = run(9000 + i)
        scores = pedm.predict_scores(obs_seq, act_seq)
        per_step.append(float(np.mean(scores)))
        or_scores.append(float(np.max(scores)))
        rets.append(ret)
        true_costs.append(tc)
        print(f"ep{i}: ret={ret:.2f} true_cost={tc:.1f} pedm_per_step={per_step[-1]:.4f} pedm_OR={or_scores[-1]:.4f}")

    n_flag = sum(1 for s in or_scores if s > OR_THR)
    print(f"\n=== {seed_dir} ===")
    print(f"mean_ret={np.mean(rets):.2f} mean_true_cost={np.mean(true_costs):.2f} "
          f"per-step={np.mean(per_step):.4f} OR={np.mean(or_scores):.4f} flagged={n_flag}/{args.episodes}")


if __name__ == "__main__":
    main()
