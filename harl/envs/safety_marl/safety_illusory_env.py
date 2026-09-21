"""Single-agent, direct-obs observation-perturbation attacker on SafetyPointGoal1 (2026-09-18),
built to be as FAITHFUL as possible to the ORIGINAL Illu-Attacks-Jax reference implementation's
own mechanism (`Illu-Attacks-Jax/train_adversary.py`), per explicit user instruction ("跑跑
illusory的工作，他们的攻击是攻击victim注意，尽可能用他们代码库的逻辑来") -- this is deliberately
NOT the Disruptor+Concealment two-role split used elsewhere in this project's `SafetyMARLEnv`.

Matches their `_env_step` exactly:
  obs_victim = last_obsv + action          # DIRECT: reaches the victim's own real decision,
                                            # no role-split. Their OWN code has NO epsilon/budget
                                            # clip at all (confirmed by reading their source: the
                                            # only clip anywhere is an episode-boundary sanity
                                            # clamp, not a perturbation-magnitude bound) -- but
                                            # 2026-09-19, per explicit user follow-up ("v16也加上
                                            # budget限制"), this env now ALSO supports an OPTIONAL
                                            # L-infinity `budget` (env_args["budget"], None/<=0 =
                                            # unclipped/reference-faithful, matching this
                                            # project's own `hidden_eps` convention for the same
                                            # 64-dim obs-perturbation channel in `SafetyMARLEnv`)
                                            # for a more controlled/comparable variant.
  action_victim = pi_vic(obs_victim)       # victim acts on the FORGED observation directly
  l2_norm = ||obs_victim - obs_victim_expected||     # RAW L2, no per-dim scale normalization
  illu_reward = -l2_norm
  reward = -reward_victim + lambda_illu * illu_reward

Two substitutions for this project's own env (both already established/validated elsewhere in
this codebase, not new departures):
  - `-reward_victim` -> `haz_shaping` (dense hazard-approach potential + one-off entry bonus,
    from `examples/train_safety_attacker.py`/`SafetyMARLEnv`) -- Safety-Gym's own raw task
    reward is blind to hazard contact, so `-victim_reward` alone teaches nothing on this env
    (established earlier this project); `haz_shaping` is the "damage" slot's already-working
    substitute, not a deviation from the reference's actual intent.
  - `obs_victim_expected` (their brax `sim_env.simulate_step`, a TRUE dynamics model) ->
    this project's own already-fit `attacker_wm_v23_fulldim.pt` (`MLPDyn`), architecturally
    unrelated to any detector, exactly the same role.

`lambda_illu` is a FIXED constant (`env_args["lambda_illu"]`), never adapted during training --
matching the reference exactly (their own `lambda_illu` is a hyperparameter swept OFFLINE via
grid search, not dual-ascended). Because reward is a SINGLE combined channel (no separate cost
critic/GAE stream), this env needs NO custom runner at all -- plain, unmodified HARL `mappo`
(`OnPolicyMARunner`) already does everything required.
"""
import copy
import numpy as np
import torch
import gymnasium as gym

import safety_gymnasium  # noqa: F401 -- registers Safety* env ids
import examples.safety_hazard_patch  # noqa: F401
import examples.safety_layout_patch  # noqa: F401
from examples.train_safety_attacker import SafetyVictim
from examples.fit_attacker_mlp import MLPDyn


class SafetyIllusoryEnv:
    """Single-agent direct-obs illusory attacker, faithfully reproducing Illu-Attacks-Jax."""

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args.get("scenario", "SafetyPointGoal1Gymnasium-v0")
        self.lambda_illu = float(args.get("lambda_illu", 1.0))
        self.max_cycles = int(args.get("max_cycles", 1000))
        # L-infinity budget (2026-09-19, per explicit user follow-up "v16也加上budget限制"):
        # None/<=0 means unclipped (the original reference's own behavior, kept as an option);
        # a positive value clips per-dimension to [-budget,budget], matching this project's own
        # `hidden_eps` convention for the same 64-dim obs-perturbation channel in `SafetyMARLEnv`.
        budget = args.get("budget", None)
        self.budget = float(budget) if budget is not None and float(budget) > 0 else None

        device = "cpu"
        self.victim = SafetyVictim(args["victim_ckpt"], device)
        self.full_dim = self.victim.full_obs_dim  # 64: raw+time+hazard
        self.act_dim = 2

        self.wm = MLPDyn.load(args["wm_ckpt"], device)
        assert self.wm.obs_dim == self.full_dim, (
            f"world model obs_dim={self.wm.obs_dim} != victim full_dim={self.full_dim}")

        self.env = gym.make(self.scenario)
        self.lo = self.env.action_space.low.astype(np.float32)
        self.hi = self.env.action_space.high.astype(np.float32)

        self.n_agents = 1
        obs_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.full_dim,), dtype=np.float32)
        # Wide but finite Box for HARL's interface only -- NOT enforced via clipping in step()
        # (see module docstring: the reference never clips the adversary's own action either).
        act_box = gym.spaces.Box(low=-10.0, high=10.0, shape=(self.full_dim,), dtype=np.float32)
        self.observation_space = [obs_box]
        self.action_space = [act_box]
        self.share_observation_space = [obs_box]

        self._seed = 0
        self.cur_step = 0

    def _dist_to_hazard(self):
        task = self.env.unwrapped.task
        return float(np.linalg.norm(task.agent.pos[:2] - task.hazards.pos[0][:2]))

    def reset(self):
        self._seed += 1
        self.cur_step = 0
        raw_obs, _ = self.env.reset(seed=self._seed)
        self.steps_since_goal = 0
        self.last_dist_hazard = self._dist_to_hazard()
        self.was_inside_hazard = False
        obs_aug = self.victim._augment(raw_obs.astype(np.float32), self.steps_since_goal, self.env)  # noqa: SLF001
        self.obs_aug = obs_aug
        self.last_reported_obs = obs_aug.copy()
        self.last_env_action = np.zeros(self.act_dim, np.float32)
        return [obs_aug], [obs_aug.copy()], self.get_avail_actions()

    def step(self, actions):
        action = np.asarray(actions[0], np.float32)[:self.full_dim]
        if self.budget is not None:
            action = np.clip(action, -self.budget, self.budget)

        o_rep = self.obs_aug + action  # direct additive perturbation, matches obs_victim = last_obsv + action
        a_v = self.victim.act_from_augmented(o_rep)  # victim decides FROM the forged observation
        a_exec = np.clip(a_v, self.lo, self.hi)

        no, r, term, trunc, info = self.env.step(a_exec)
        self.cur_step += 1
        if self.cur_step >= self.max_cycles:
            trunc = True
        c = float(info.get("cost", 0.0))

        dh_new = self._dist_to_hazard()
        haz_shaping = (self.last_dist_hazard - dh_new) * 1.0
        if c > 0 and not self.was_inside_hazard:
            haz_shaping += 10.0
        self.was_inside_hazard = c > 0
        self.last_dist_hazard = dh_new

        with torch.no_grad():
            expected = self.wm.next_obs(self.last_reported_obs, self.last_env_action)
        l2_norm = float(np.linalg.norm(o_rep - expected))  # RAW L2, no per-dim scale (matches reference)
        illu_reward = -l2_norm
        reward = float(haz_shaping) + self.lambda_illu * illu_reward

        self.last_reported_obs = o_rep.copy()
        self.last_env_action = a_exec.copy()

        self.steps_since_goal = 0 if info.get("goal_met", False) else self.steps_since_goal + 1
        obs_aug_next = self.victim._augment(no.astype(np.float32), self.steps_since_goal, self.env)  # noqa: SLF001
        self.obs_aug = obs_aug_next

        done = bool(term or trunc)
        rewards = [[reward]]
        infos = [{"true_cost": c, "illu_reward": illu_reward, "haz_shaping": haz_shaping,
                  "bad_transition": bool(trunc and not term)}]
        dones = [done]
        return [obs_aug_next], [obs_aug_next.copy()], rewards, dones, infos, self.get_avail_actions()

    def get_avail_actions(self):
        return None

    def seed(self, seed):
        self._seed = seed

    def render(self):
        pass

    def close(self):
        self.env.close()
