"""Offline deterministic evaluation of a HARL-native mappo_lagr/mappo_hard checkpoint (2026-09-17).

Per explicit request: no eval-thread pool runs DURING training (that memory went to more
train workers instead); only the final saved checkpoint gets evaluated, and only as a
separate, single-process, deterministic pass after training. This script loads a run
directory's config.json + models/actor_agent*.pt, replays N deterministic episodes through
the *same* SafetyMARLEnv the run trained on, and reports:
  - true_cost: the real safety_gymnasium hazard cost (physical attack effectiveness)
  - stealth_cost: the illusory-consistency cost against the true dynamics model (what the
    Lagrangian/hard constraint was trained against)
  - reward: haz_shaping team reward

Usage:
    python -m examples.eval_harl_native_final --run_dir results/safety_marl/SafetyPointGoal1Gymnasium-v0/mappo_lagr/harl_native_v2_noeval/seed-00001-<timestamp> --episodes 15
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from harl.algorithms.actors import ALGO_REGISTRY
from harl.envs.safety_marl.safety_marl_env import SafetyMARLEnv


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
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    seed_dir = latest_seed_dir(args.run_dir)
    with open(os.path.join(seed_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    main_args = config["main_args"]
    algo_args = config["algo_args"]
    env_args = config["env_args"]

    device = torch.device(args.device)
    env = SafetyMARLEnv(env_args)
    num_agents = env.n_agents

    actors = []
    for agent_id in range(num_agents):
        agent = ALGO_REGISTRY[main_args["algo"]](
            {**algo_args["model"], **algo_args["algo"]},
            env.observation_space[agent_id],
            env.action_space[agent_id],
            device=device,
        )
        ckpt_path = os.path.join(seed_dir, "models", f"actor_agent{agent_id}.pt")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.actor.load_state_dict(state_dict)
        agent.prep_rollout()
        actors.append(agent)

    recurrent_n = algo_args["model"]["recurrent_n"]
    rnn_hidden_size = algo_args["model"]["hidden_sizes"][-1]

    true_costs, stealth_costs, rewards, ep_lens = [], [], [], []
    for ep in range(args.episodes):
        obs, share_obs, avail = env.reset()
        rnn_states = [
            np.zeros((1, recurrent_n, rnn_hidden_size), dtype=np.float32)
            for _ in range(num_agents)
        ]
        masks = [np.ones((1, 1), dtype=np.float32) for _ in range(num_agents)]

        ep_true_cost = 0.0
        ep_stealth_cost = 0.0
        ep_reward = 0.0
        t = 0
        done = False
        while not done:
            actions = []
            for agent_id in range(num_agents):
                a, rnn_states[agent_id] = actors[agent_id].act(
                    obs[agent_id][None], rnn_states[agent_id], masks[agent_id],
                    None, deterministic=True,
                )
                actions.append(a.detach().cpu().numpy()[0])
            obs, share_obs, rew, dones, infos, avail = env.step(actions)
            ep_true_cost += infos[0]["true_cost"]
            ep_stealth_cost += infos[0]["cost"]
            ep_reward += rew[0][0]
            done = dones[0]
            t += 1
            if t >= env.max_cycles:
                break
        true_costs.append(ep_true_cost)
        stealth_costs.append(ep_stealth_cost / max(t, 1))
        rewards.append(ep_reward)
        ep_lens.append(t)
        print(f"ep {ep}: len={t} true_cost_sum={ep_true_cost:.1f} "
              f"stealth_cost_mean={ep_stealth_cost / max(t, 1):.4f} reward={ep_reward:.2f}")

    env.close()
    print("\n=== summary ({} episodes, {}) ===".format(args.episodes, seed_dir))
    print(f"true_cost_sum:    mean={np.mean(true_costs):.2f} median={np.median(true_costs):.2f} "
          f"max={np.max(true_costs):.2f}")
    print(f"stealth_cost_avg: mean={np.mean(stealth_costs):.4f} median={np.median(stealth_costs):.4f} "
          f"max={np.max(stealth_costs):.4f}")
    print(f"reward:           mean={np.mean(rewards):.2f}")
    print(f"ep_len:           mean={np.mean(ep_lens):.1f}")


if __name__ == "__main__":
    main()
