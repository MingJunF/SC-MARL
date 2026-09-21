"""HARL-native multi-agent env porting the SafetyMARLEnv Disruptor/Concealment role-split
threat model (see `harl/envs/safety_marl/safety_marl_env.py`'s module docstring for the full
history) onto a plain MuJoCo locomotion victim (HalfCheetah-v4 by default), per explicit user
request (2026-09-20) to re-run the same architecture on a much simpler benchmark after
SafetyPointGoal1's own scenario complexity became a suspect for why every checkpoint with a
real, sustained attack has also failed to stay stealthy.

Same role split, same two-agent structure, same Lagrangian runner (`OnPolicyLagrRunner` /
`mappo_lagr`) as safety_marl -- ONLY the env layer differs:
  agent 0 = Disruptor: perturbs the victim's action. `a_exec = a_v + phi_dis` (per-dim
            tanh-squashed to budget `disruptor_eps`, L-infinity) is the ONLY channel that
            reaches the real simulator.
  agent 1 = Concealment: fabricates BOTH halves of the record the stealth cost audits -- a
            forged observation `o_rep` (L-infinity budget `hidden_eps`) and a forged action
            `a_rep` (L-infinity budget `hidden_act_eps`, added to the TRUE `a_v`) -- NEITHER
            ever reaches the victim's own decision or the simulator. Zero physical effect,
            exactly as Concealment is defined in safety_marl_env.py.

Budgets (2026-09-20, explicit user values "act0.4，obs0.2，concealment也是0.2", superseding this
project's earlier L2-ball "core budget" numbers which were calibrated for a different norm):
`disruptor_eps` (B_act) = 0.4, `hidden_eps` (B_obs) = 0.2, `hidden_act_eps` = 0.2 -- symmetric
with `hidden_eps`, mirroring safety_marl's own "give both Concealment channels the same budget"
convention.

2026-09-20, per explicit user correction ("l无线不要l2" -- L-infinity, not L2): perturbations
use the SAME per-dimension tanh-squash mechanism as safety_marl_env.py (`phi_n = eps *
tanh(raw)`), NOT an L2-ball projection -- an earlier version of this port used
`examples/train_obs_attacker.py`'s `l2_project`, but the user wants this HARL-native port to
match safety_marl's own L-infinity/tanh geometry instead. tanh has no flat/zero-gradient region
(unlike a hard per-dim clip), so this keeps the ATLA-inspired saturation fix's benefit while
switching norms.

2026-09-20, second correction, per explicit user framing ("illusory是怎么限制的？现在就是把
illusory的限制拓展到marl，只是这里有个专门的concelaer罢了" -- this IS illusory's own
constraint, just extended to MARL with a dedicated Concealment agent, not a new invention):
`train_obs_attacker.py`'s reference illusory implementation applies its budget in NORMALIZED
space and only de-normalizes AFTER projecting -- `phi_n = l2_project(raw, budget); obs_vic =
obs + phi_n * scale` (`scale` = per-dim max-abs from clean rollouts, from this same
`estimate_scales` helper) -- an earlier version of this env's obs-forgery skipped that
de-normalization step entirely (`phi_hid_obs = hidden_eps * tanh(raw)`, applied directly in RAW
physical units). That gap is exactly what caused the very high (~300-400) truedyn stealth cost
observed even from an untrained policy: HalfCheetah's 17 obs dims have wildly different natural
scales, so a RAW 0.2-unit perturbation on a dimension whose own clean-rollout range is much
smaller than 0.2 is already a huge, out-of-distribution jump for that dimension --
`AntTrueDynamics`'s exact physics simulator (unlike PEDM's smoothed learned approximation) is
chaotically sensitive to exactly this kind of jump, so the one-step prediction error explodes.
Fixed: `phi_n = eps * tanh(raw)` (still per-dim tanh-squashed, still L-infinity -- the norm
switch from the FIRST correction above stays), THEN de-normalized by `obs_scale`/`act_scale`
(this env's own equivalent of the reference's `scale`/`ascale`) before being added to the real
quantity, for ALL THREE channels (obs-forgery, action-forgery, AND Disruptor's physical
attack -- HalfCheetah's action space is already uniformly [-1,1] so `act_scale` is close to 1
and barely changes Disruptor's own behavior, but is included for exact fidelity to "just
illusory, extended"). The stealth-cost penalty itself also now matches the reference's own
SUM-OF-SQUARES convention (`sum(((o_rep-expected)/obs_scale)**2)`, no final sqrt) instead of an
L2 norm, so `eps_cost` can be calibrated against the same order of magnitude as the reference's
own `eps_kl` (paper default 0.1), not an arbitrarily-chosen number.

Reward: BOTH agents get `-r_env` (the victim's own clean task reward, negated) -- mirrors
`train_marl_attacker.py`'s `r_att = -r_victim` convention exactly. There is no separate
"physical safety cost" channel here the way SafetyPointGoal1 has hazard contact; degrading the
victim's own task performance IS the whole point of the attack, so it doubles as both the
reward AND (negated) the `true_cost` diagnostic already wired into `OnPolicyLagrRunner`'s
`true_cost_per_step`/`true_cost_per_episode` logging.

Stealth cost: `stealth_source` is fixed to "truedyn" ONLY (2026-09-20, per explicit user
instruction "这里限制注意不要用pedm而是truedyn") -- no PEDM branch exists in this env at all, by
design, unlike safety_marl_env's multi-mode setup. Uses `AntTrueDynamics`
(`examples/train_obs_attacker.py`, generic across Ant/HalfCheetah/Hopper/Walker via qpos/qvel
reconstruction from the observation) as the "detector": the EXACT ground-truth one-step MuJoCo
dynamics, zero approximation error -- the strongest possible defender a real detector could
only approximate. `cost_t = ||o_rep_t - true_dyn(last_reported_obs, last_a_rep)|| / obs_scale`,
checking whether `o_rep_t` is consistent with what SHOULD have followed from Concealment's own
previously-reported `(last_reported_obs, last_a_rep)` -- exactly the same "is the deceiver's own
narrative self-consistent" check as safety_marl_env's "wm" mode, just against the true simulator
instead of a learned MLPDyn. If Concealment cannot evade even this, no amount of PEDM
mis-calibration could have been the bottleneck.
"""
import copy

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from examples.train_obs_attacker import AntTrueDynamics, estimate_scales, project_budget
from examples.eval_robust_detector import Victim


class MujocoMARLEnv:
    """2-agent (Disruptor, Concealment) env wrapping a frozen HARL MuJoCo victim."""

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args.get("scenario", "HalfCheetah-v4")
        self.disruptor_eps = float(args.get("disruptor_eps", 0.4))  # B_act
        self.hidden_eps = float(args.get("hidden_eps", 0.2))  # B_obs
        # Symmetric with hidden_eps by default (2026-09-20 explicit user value) -- mirrors
        # safety_marl's own "give both Concealment channels the same budget" convention.
        self.hidden_act_eps = float(args.get("hidden_act_eps", self.hidden_eps))
        self.max_cycles = int(args.get("max_cycles", 1000))
        # 2026-09-20, per explicit user ablation request ("v7改成L2约束跑一版本看看"): default
        # stays "linf" (v7's own geometry, UNCHANGED for any existing/future linf launch) --
        # "l2" reproduces train_obs_attacker.py's own L2-ball-then-denormalize convention
        # (`phi = l2_project(raw, eps) * scale`) instead of the per-dim tanh-squash, so the
        # role-split architecture itself can be tested under the SAME norm choice already
        # ablated for the single-agent illusory baseline.
        self.budget_norm = args.get("budget_norm", "linf")
        assert self.budget_norm in ("linf", "l2")

        device = "cpu"
        probe = gym.make(self.scenario)
        self.obs_dim = int(probe.observation_space.shape[0])
        self.act_dim = int(probe.action_space.shape[0])
        self.lo = probe.action_space.low.astype(np.float32)
        self.hi = probe.action_space.high.astype(np.float32)
        probe.close()

        act_space = Box(self.lo, self.hi, (self.act_dim,), np.float32)
        self.victim = Victim(args["victim_run"], self.obs_dim, act_space, device)

        # The ground-truth one-step dynamics model doubling as the "detector" -- see module
        # docstring. Uses its OWN internal mujoco sim instance, entirely separate from
        # `self.env` below, so probing it never perturbs the real trajectory.
        self.dyn = AntTrueDynamics(self.scenario)

        # Per-dim max-abs scale from clean rollouts (`train_obs_attacker.py`'s own
        # `estimate_scales` helper) -- used to DE-NORMALIZE every tanh-squashed perturbation
        # below (`phi = eps*tanh(raw)*scale`), exactly like the reference illusory
        # implementation's `obs_vic = obs + phi_n*scale` / `env_action = a_vic + phi_a*ascale`.
        # Without this, a fixed RAW budget (e.g. hidden_eps=0.2) is wildly out-of-scale for
        # whichever obs dims happen to have a naturally smaller range than 0.2.
        scale_path = args.get("obs_scale_path")
        if scale_path:
            self.obs_scale = np.load(scale_path).astype(np.float32)
            self.act_scale = np.ones(self.act_dim, np.float32)
        else:
            self.obs_scale, self.act_scale = estimate_scales(
                self.scenario, self.victim, self.lo, self.hi
            )
        assert self.obs_scale.shape == (self.obs_dim,), (
            f"obs_scale shape {self.obs_scale.shape} != obs_dim ({self.obs_dim},)"
        )

        self.env = gym.make(self.scenario)

        self.n_agents = 2
        # Padded, UNIFORM per-agent spaces (HARL's vec-env stacking requires this -- see
        # safety_marl_env.py's module docstring for the same pattern). Both agents observe
        # [obs, a_v]; Concealment's action packs obs-forgery (first obs_dim dims) + action-
        # forgery (next act_dim dims); Disruptor reads only its own first act_dim dims of the
        # same padded width, the rest is unused padding for it.
        self.obs_pad = self.obs_dim + self.act_dim
        self.act_pad = self.obs_dim + self.act_dim
        obs_box = Box(low=-np.inf, high=np.inf, shape=(self.obs_pad,), dtype=np.float32)
        act_box = Box(low=-np.inf, high=np.inf, shape=(self.act_pad,), dtype=np.float32)
        self.observation_space = [obs_box, obs_box]
        self.action_space = [act_box, act_box]
        self.share_observation_space = [obs_box, obs_box]

        self._seed = 0
        self.cur_step = 0

    def _local_obs(self, obs, a_v):
        local_obs = np.concatenate([obs, a_v]).astype(np.float32)
        return [local_obs.copy(), local_obs.copy()]

    def reset(self):
        self._seed += 1
        self.cur_step = 0
        obs, _ = self.env.reset(seed=self._seed)
        obs = obs.astype(np.float32)
        self.victim.reset()
        # ONE victim.act() call per real timestep, in sequential order -- the victim's actor is
        # genuinely recurrent (recurrent_n=1), so calling it more than once per env step (or out
        # of order) would silently corrupt its hidden state relative to how it's actually
        # deployed. This is why a_v for step t is always computed and CACHED at the end of step
        # t-1 (or here, at reset), never recomputed mid-step from scratch.
        a_v0 = np.clip(self.victim.act(obs), self.lo, self.hi)
        self.last_obs = obs
        self.last_a_v = a_v0
        # clean init: no forgery has happened yet, so the reported obs/action start truthful
        self.last_reported_obs = obs.copy()
        self.last_a_rep = a_v0.copy()
        out_obs = self._local_obs(obs, a_v0)
        share_obs = [np.concatenate([obs, a_v0]).astype(np.float32)] * self.n_agents
        return out_obs, share_obs, self.get_avail_actions()

    def step(self, actions):
        # actions: (n_agents, act_pad) -- agent 0 = Disruptor (first act_dim dims), agent 1 =
        # Concealment (first obs_dim dims = obs-forgery, next act_dim dims = action-forgery).
        raw_dis = np.asarray(actions[0], np.float32)[:self.act_dim]
        raw_hid = np.asarray(actions[1], np.float32)
        raw_hid_obs = raw_hid[:self.obs_dim]
        raw_hid_act = raw_hid[self.obs_dim:self.obs_dim + self.act_dim]

        # Budget applied so it bounds the FINAL PHYSICAL perturbation consistently across any
        # environment (2026-09-21 fix, see `project_budget`'s own docstring in
        # train_obs_attacker.py for the full before/after -- the L2 path used to project THEN
        # scale, letting the physical norm balloon far past `budget` whenever `obs_scale`/
        # `act_scale` had a wide spread; now scales first and caps the physical norm directly).
        # `self.budget_norm` selects the projection geometry ("linf": per-dim tanh-squash, each
        # dim independently capped at `budget*scale[dim]`; "l2": total physical L2 norm capped
        # at exactly `budget`, regardless of environment).
        phi_hid_obs = project_budget(raw_hid_obs, self.hidden_eps, self.obs_scale, self.budget_norm)
        o_rep = self.last_obs + phi_hid_obs  # forged record ONLY -- never fed to the victim

        # ROLE-SPLIT (mirrors safety_marl_env.py's standing instruction): victim acts on the
        # TRUE observation, cached from the end of the previous step -- Concealment's forgery
        # has zero causal effect on the victim or the real trajectory.
        a_v = self.last_a_v

        phi_hid_act = project_budget(raw_hid_act, self.hidden_act_eps, self.act_scale, self.budget_norm)
        a_rep = np.clip(a_v + phi_hid_act, self.lo, self.hi)  # forged action, never executed

        phi_dis = project_budget(raw_dis, self.disruptor_eps, self.act_scale, self.budget_norm)
        a_exec = np.clip(a_v + phi_dis, self.lo, self.hi)  # the ONLY channel that acts for real

        no, r, term, trunc, info = self.env.step(a_exec)
        self.cur_step += 1
        if self.cur_step >= self.max_cycles:
            trunc = True
        no = no.astype(np.float32)

        # Stealth cost: is o_rep_t consistent with what the TRUE dynamics say should have
        # followed from Concealment's own previously-reported (last_reported_obs, last_a_rep)?
        # See module docstring -- this is safety_marl_env's "wm" mode with the learned MLPDyn
        # swapped for the exact ground-truth simulator. Sum-of-squares (no sqrt), matching the
        # reference illusory implementation's own `penalty = sum(((obs_vic-expected)/scale)**2)`
        # convention exactly, so `eps_cost` can be calibrated against the same order of
        # magnitude as the reference's own `eps_kl` (paper default 0.1).
        expected = self.dyn.next_obs(self.last_reported_obs, self.last_a_rep)
        stealth_cost = float(np.sum(((o_rep - expected) / self.obs_scale) ** 2))

        self.last_reported_obs = o_rep.copy()
        self.last_a_rep = a_rep.copy()
        self.last_obs = no
        # Next real timestep's a_v, computed now (see reset()'s comment on call ordering).
        a_v_next = np.clip(self.victim.act(no), self.lo, self.hi)
        self.last_a_v = a_v_next

        obs_out = self._local_obs(no, a_v_next)
        share_obs = [np.concatenate([no, a_v_next]).astype(np.float32)] * self.n_agents
        done = bool(term or trunc)
        # Both agents get the same team reward: degrading the victim's own task performance is
        # the entire point of the attack (mirrors train_marl_attacker.py's `r_att = -r_victim`).
        rewards = [[float(-r)], [float(-r)]]
        info_common = {
            "cost": stealth_cost,
            # No separate physical-safety channel exists for this env family (unlike
            # SafetyPointGoal1's hazard cost) -- reuse the negated task reward as the
            # `true_cost` diagnostic so OnPolicyLagrRunner's existing true_cost_per_step/
            # true_cost_per_episode logging shows "how much damage is actually happening" for
            # free, without env-specific changes to the runner.
            "true_cost": float(-r),
            "bad_transition": bool(trunc and not term),
            "a_v": a_v.astype(np.float32),
            "a_rep": a_rep.astype(np.float32),
            "o_rep_scaled": (o_rep / self.obs_scale).astype(np.float32),
        }
        infos = [dict(info_common), dict(info_common)]
        dones = [done, done]
        return obs_out, share_obs, rewards, dones, infos, self.get_avail_actions()

    def get_avail_actions(self):
        return None  # continuous action spaces only

    def seed(self, seed):
        self._seed = seed

    def render(self):
        pass

    def close(self):
        self.env.close()
