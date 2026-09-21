"""Evaluate the PEDM anomaly detector against a trained Robust-Gymnasium attack.

This is a *self-contained* replay: it loads a single-agent HARL victim policy
and its corresponding two trained robust attackers (observation attacker +
action attacker), reproduces the robust-attack perturbation pipeline on the
plain gymnasium MuJoCo environment, and measures how well the PEDM detector
separates clean from attacked observation transitions.

The victim is a HARL ``StochasticPolicy`` (env ``robust_victim``, algo
``mappo``). The attackers are HARL ``DeterministicPolicy`` networks (env
``robust_attack``). See ``harl/detectors`` for the detector.

Example:
    python examples/eval_robust_detector.py \
        --attack_run results/robust_attack/Ant-v4/iddpg/attack_iddpg_ant_eps010_seed1/seed-00001-2026-07-02-03-24-35

If ``--victim_run`` is omitted it is auto-resolved from the attack's scenario
and seed under ``results/robust_victim``.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from gymnasium.spaces import Box
from scipy.stats import rankdata

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.models.policy_models.deterministic_policy import DeterministicPolicy
from harl.detectors.pedm_detector import PEDMDetector


# --------------------------------------------------------------------------- #
#  Victim / attacker wrappers                                                  #
# --------------------------------------------------------------------------- #
class Victim:
    """Frozen HARL StochasticPolicy victim (deterministic actions)."""

    def __init__(self, victim_run, obs_dim, act_space, device):
        cfg = json.load(open(os.path.join(victim_run, "config.json")))
        args = {**cfg["algo_args"]["model"], **cfg["algo_args"]["algo"]}
        self.device = device
        self.recurrent_n = args["recurrent_n"]
        self.hidden = args["hidden_sizes"][-1]
        obs_space = Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.actor = StochasticPolicy(args, obs_space, act_space, device)
        state_dict = torch.load(
            os.path.join(victim_run, "models", "actor_agent0.pt"),
            map_location=device,
        )
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        self.reset()

    def reset(self):
        self._rnn = np.zeros((1, self.recurrent_n, self.hidden), dtype=np.float32)

    @torch.no_grad()
    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        masks = np.ones((1, 1), dtype=np.float32)
        action, _, rnn = self.actor(obs, self._rnn, masks, deterministic=True)
        self._rnn = rnn.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0].astype(np.float32)


class Attacker:
    """Trained HARL DeterministicPolicy attacker (obs or action perturbation)."""

    def __init__(self, model_path, aug_obs_dim, pad_dim, budget_max, device):
        args = {
            "hidden_sizes": [128, 128],
            "activation_func": "relu",
            "final_activation_func": "tanh",
        }
        obs_space = Box(-np.inf, np.inf, (aug_obs_dim,), np.float32)
        act_space = Box(-budget_max, budget_max, (pad_dim,), np.float32)
        self.net = DeterministicPolicy(args, obs_space, act_space, device)
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, aug_obs):
        x = torch.as_tensor(aug_obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        return self.net(x).cpu().numpy()[0].astype(np.float32)


# --------------------------------------------------------------------------- #
#  Replay environment                                                          #
# --------------------------------------------------------------------------- #
class RobustAttackReplay:
    """Reproduces the robust-attack pipeline on a plain gymnasium env."""

    def __init__(self, attack_run, victim_run, device, episode_length=None):
        acfg = json.load(open(os.path.join(attack_run, "config.json")))
        env_args = acfg["env_args"]
        self.scenario = env_args["scenario"]
        self.attack_observation = bool(env_args.get("attack_observation", True))
        self.attack_action = bool(env_args.get("attack_action", True))
        self.eps_obs = float(env_args.get("epsilon_observation", 0.1))
        self.eps_act = float(env_args.get("epsilon_action", 0.1))
        self.clip_observation = bool(env_args.get("clip_observation", False))
        self.device = device

        self.env = gym.make(self.scenario)
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))
        self.act_low = self.env.action_space.low.astype(np.float32)
        self.act_high = self.env.action_space.high.astype(np.float32)
        self._obs_low = self.env.observation_space.low.astype(np.float32)
        self._obs_high = self.env.observation_space.high.astype(np.float32)
        self.episode_length = episode_length or int(
            json.load(open(os.path.join(victim_run, "config.json")))["algo_args"][
                "train"
            ]["episode_length"]
        )

        act_space = Box(self.act_low, self.act_high, (self.act_dim,), np.float32)
        self.victim = Victim(victim_run, self.obs_dim, act_space, device)

        # attacker augmented obs = [state, victim_action, budget_frac]
        self.aug_obs_dim = self.obs_dim + self.act_dim + 1
        self.pad_dim = max(self.obs_dim, self.act_dim)
        budget_max = max(self.eps_obs, self.eps_act)
        models_dir = os.path.join(attack_run, "models")
        self.obs_attacker = (
            Attacker(
                os.path.join(models_dir, "actor_agent0.pt"),
                self.aug_obs_dim, self.pad_dim, budget_max, device,
            )
            if self.attack_observation
            else None
        )
        # agent index of the action attacker depends on whether obs attack exists
        act_idx = 1 if self.attack_observation else 0
        self.act_attacker = (
            Attacker(
                os.path.join(models_dir, f"actor_agent{act_idx}.pt"),
                self.aug_obs_dim, self.pad_dim, budget_max, device,
            )
            if self.attack_action
            else None
        )

    def _build_aug(self, state_slot, action_slot):
        return np.concatenate(
            [state_slot, action_slot, np.array([1.0], np.float32)]
        ).astype(np.float32)

    def rollout(self, attack, seed=None):
        """Run one episode; return (obs_seq [T+1,obs], act_seq [T,act], ret)."""
        state, _ = self.env.reset(seed=seed)
        state = np.asarray(state, dtype=np.float32)
        self.victim.reset()
        last_delta_o = np.zeros(self.obs_dim, dtype=np.float32)
        obs_seq, act_seq = [], []
        ret = 0.0

        for _t in range(self.episode_length):
            if attack and self.obs_attacker is not None:
                a0_in = self._build_aug(state, np.zeros(self.act_dim, np.float32))
                delta_o = np.clip(
                    self.obs_attacker(a0_in)[: self.obs_dim],
                    -self.eps_obs, self.eps_obs,
                )
                victim_obs = state + delta_o
                if self.clip_observation:
                    victim_obs = np.clip(victim_obs, self._obs_low, self._obs_high)
            else:
                delta_o = np.zeros(self.obs_dim, dtype=np.float32)
                victim_obs = state

            victim_action = self.victim.act(victim_obs)

            if attack and self.act_attacker is not None:
                # action attacker sees victim action on the (stale) corrupted obs
                stale = state + last_delta_o
                if self.clip_observation:
                    stale = np.clip(stale, self._obs_low, self._obs_high)
                stale_action = self.victim.act(stale)
                self.victim._rnn = np.zeros_like(self.victim._rnn)  # peek: no advance
                a1_in = self._build_aug(np.zeros(self.obs_dim, np.float32), stale_action)
                delta_a = np.clip(
                    self.act_attacker(a1_in)[: self.act_dim],
                    -self.eps_act, self.eps_act,
                )
            else:
                delta_a = np.zeros(self.act_dim, dtype=np.float32)

            applied = np.clip(victim_action + delta_a, self.act_low, self.act_high)
            last_delta_o = delta_o

            obs_seq.append(victim_obs.astype(np.float32))
            act_seq.append(victim_action.astype(np.float32))

            next_state, reward, term, trunc, _ = self.env.step(applied)
            ret += float(reward)
            state = np.asarray(next_state, dtype=np.float32)
            if term or trunc:
                break

        # final observation the victim would see (for the last transition target)
        if attack and self.obs_attacker is not None:
            a0_in = self._build_aug(state, np.zeros(self.act_dim, np.float32))
            delta_o = np.clip(
                self.obs_attacker(a0_in)[: self.obs_dim], -self.eps_obs, self.eps_obs
            )
            final_obs = state + delta_o
            if self.clip_observation:
                final_obs = np.clip(final_obs, self._obs_low, self._obs_high)
        else:
            final_obs = state
        obs_seq.append(final_obs.astype(np.float32))

        return np.array(obs_seq), np.array(act_seq), ret


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #
def auroc(neg_scores, pos_scores):
    """Area under ROC: P(score(attacked) > score(clean))."""
    neg = np.asarray(neg_scores)
    pos = np.asarray(pos_scores)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    r = rankdata(scores)
    n_pos, n_neg = len(pos), len(neg)
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(name, scores):
    s = np.concatenate([np.ravel(x) for x in scores])
    qs = np.quantile(s, [0.5, 0.9, 0.95, 0.99])
    print(
        f"  {name:<10} n={s.size:<6} mean={s.mean():.4f} "
        f"p50={qs[0]:.4f} p90={qs[1]:.4f} p95={qs[2]:.4f} p99={qs[3]:.4f}"
    )
    return s


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def resolve_victim_run(attack_run):
    acfg = json.load(open(os.path.join(attack_run, "config.json")))
    scenario = acfg["env_args"]["scenario"]
    seed = acfg["algo_args"]["seed"]["seed"]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(
        repo, "results", "robust_victim", scenario, "mappo", "*",
        f"seed-{seed:05d}-*",
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(
            f"could not auto-resolve victim run for {scenario} seed {seed}; "
            f"pass --victim_run explicitly (searched {pattern})"
        )
    return matches[-1]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--attack_run", type=str, required=True,
                        help="Attack seed dir containing config.json and models/.")
    parser.add_argument("--victim_run", type=str, default="",
                        help="Victim seed dir; auto-resolved if omitted.")
    parser.add_argument("--n_clean_fit", type=int, default=30,
                        help="Clean episodes used to fit the dynamics model.")
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Episodes per setting (clean / attacked) for scoring.")
    parser.add_argument("--episode_length", type=int, default=0,
                        help="Override episode length (0 = victim default).")
    parser.add_argument("--dyn_epochs", type=int, default=100)
    parser.add_argument("--criterion", type=str, default="pred_error_samples")
    parser.add_argument("--aggregation", type=str, default="min_mean")
    parser.add_argument("--n_part", type=int, default=100)
    parser.add_argument("--quantile", type=float, default=0.95,
                        help="Clean-score quantile used as the detection threshold.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    victim_run = args.victim_run or resolve_victim_run(args.attack_run)
    print(f"attack_run : {args.attack_run}")
    print(f"victim_run : {victim_run}")

    replay = RobustAttackReplay(
        args.attack_run, victim_run, device,
        episode_length=args.episode_length or None,
    )
    print(
        f"scenario={replay.scenario} obs_dim={replay.obs_dim} act_dim={replay.act_dim} "
        f"attack_obs={replay.attack_observation}(eps={replay.eps_obs}) "
        f"attack_act={replay.attack_action}(eps={replay.eps_act}) "
        f"ep_len={replay.episode_length}"
    )

    # 1) collect clean data and fit the dynamics model
    print(f"\n[1/4] collecting {args.n_clean_fit} clean episodes to fit PEDM...")
    fit_obs, fit_act, fit_nobs = [], [], []
    clean_returns = []
    for ep in range(args.n_clean_fit):
        o, a, r = replay.rollout(attack=False, seed=1000 + ep)
        fit_obs.append(o[:-1]); fit_act.append(a); fit_nobs.append(o[1:])
        clean_returns.append(r)
    fit_obs = np.concatenate(fit_obs); fit_act = np.concatenate(fit_act)
    fit_nobs = np.concatenate(fit_nobs)

    detector = PEDMDetector(
        obs_dim=replay.obs_dim, action_dim=replay.act_dim,
        criterion=args.criterion, aggregation_function=args.aggregation,
        n_part=args.n_part, device=device,
    )
    print(f"      fitting on {len(fit_obs)} transitions ({args.dyn_epochs} epochs)...")
    detector.fit(fit_obs, fit_act, fit_nobs, n_train_epochs=args.dyn_epochs, verbose=True)

    # 2) score held-out clean episodes and calibrate the threshold
    print(f"\n[2/4] scoring {args.n_eval} held-out clean episodes...")
    clean_scores, eval_clean_returns = [], []
    for ep in range(args.n_eval):
        o, a, r = replay.rollout(attack=False, seed=5000 + ep)
        clean_scores.append(detector.predict_scores(o, a))
        eval_clean_returns.append(r)
    threshold = detector.calibrate_threshold(clean_scores, quantile=args.quantile)

    # 3) score attacked episodes
    print(f"[3/4] scoring {args.n_eval} attacked episodes...")
    attacked_scores, attacked_returns = [], []
    for ep in range(args.n_eval):
        o, a, r = replay.rollout(attack=True, seed=9000 + ep)
        attacked_scores.append(detector.predict_scores(o, a))
        attacked_returns.append(r)

    # 4) report
    clean_flat = np.concatenate([np.ravel(s) for s in clean_scores])
    attacked_flat = np.concatenate([np.ravel(s) for s in attacked_scores])
    fp_rate = float(np.mean(clean_flat > threshold))
    det_rate = float(np.mean(attacked_flat > threshold))
    auc = auroc(clean_flat, attacked_flat)

    # episode-level scores (max / mean anomaly score per episode)
    clean_max = np.array([np.max(s) for s in clean_scores])
    attacked_max = np.array([np.max(s) for s in attacked_scores])
    clean_mean = np.array([np.mean(s) for s in clean_scores])
    attacked_mean = np.array([np.mean(s) for s in attacked_scores])
    ep_thr_max = float(np.quantile(clean_max, args.quantile))
    ep_thr_mean = float(np.quantile(clean_mean, args.quantile))

    print("\n[4/4] ===================== RESULTS =====================")
    print(f"criterion={args.criterion} aggregation={args.aggregation} "
          f"quantile={args.quantile}")
    print(f"detection threshold (clean q{int(args.quantile*100)}) = {threshold:.5f}")
    print("per-transition anomaly-score stats:")
    summarize("clean", clean_scores)
    summarize("attacked", attacked_scores)
    print("----------------------- per-transition --------------------")
    print(f"  detection rate (attacked flagged) : {det_rate:.3f}")
    print(f"  false-positive rate (clean flagged): {fp_rate:.3f}")
    print(f"  ROC-AUC (clean vs attacked)        : {auc:.3f}")
    print("----------------------- per-episode -----------------------")
    print(f"  max-score  threshold={ep_thr_max:.4f}  "
          f"detect={np.mean(attacked_max > ep_thr_max):.3f}  "
          f"fp={np.mean(clean_max > ep_thr_max):.3f}  "
          f"AUROC={auroc(clean_max, attacked_max):.3f}")
    print(f"  mean-score threshold={ep_thr_mean:.4f}  "
          f"detect={np.mean(attacked_mean > ep_thr_mean):.3f}  "
          f"fp={np.mean(clean_mean > ep_thr_mean):.3f}  "
          f"AUROC={auroc(clean_mean, attacked_mean):.3f}")
    print("----------------------- attack effect ---------------------")
    print(f"  victim return  clean   : {np.mean(eval_clean_returns):.1f} "
          f"+/- {np.std(eval_clean_returns):.1f}")
    print(f"  victim return  attacked: {np.mean(attacked_returns):.1f} "
          f"+/- {np.std(attacked_returns):.1f}")
    print(f"  return drop            : "
          f"{np.mean(eval_clean_returns) - np.mean(attacked_returns):.1f}")
    print("=====================================================")


if __name__ == "__main__":
    main()
