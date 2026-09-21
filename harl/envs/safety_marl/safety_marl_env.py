"""HARL-native multi-agent env for the SafetyPointGoal1 Disruptor+Concealment threat model
(2026-09-17), per explicit user request to properly integrate with HARL's own MAPPO/runner
pipeline instead of continuing hand-rolled standalone PPO scripts.

Two heterogeneous agents, ROLE-SPLIT (standing instruction, 2026-09-18 -- verbatim: "performance
attacker（act）attacker攻击victim。concealment只攻击"detector"。concealment对victim本身毫无影响"):
  agent 0 = Disruptor / performance attacker: perturbs the victim's action (budget
            `disruptor_eps`, L-infinity) -- the ONLY channel allowed to affect the victim's real
            behavior / the real trajectory.
  agent 1 = Concealment / deceiver: fabricates BOTH halves of the record PEDM audits -- a
            forged observation `o_rep` (budget `hidden_eps`, L-infinity, added to `obs_aug`)
            AND a forged action `a_rep` (budget `hidden_act_eps`, L-infinity, added to the
            TRUE `a_v`) -- but NEITHER ever reaches the victim's own decision or the real
            actuator: `a_v = victim.act_from_augmented(obs_aug)` always uses the TRUE
            observation, and `a_exec = a_v + phi_dis` (Disruptor's channel) is the only thing
            physically executed. Concealment's only effect on anything is through the
            stealth-cost computation below (scored on `(o_rep, a_rep)`, its own doubly-forged
            record).
            (2026-09-17/18: an earlier DIRECT-OBS version of this env fed `o_rep` into the
            victim's own decision instead -- found, via an ablation that gave Concealment a
            nonzero `true_cost` with Disruptor disabled, to let Concealment indirectly steer
            real victim behavior, which is exactly what this role-split forbids. Reverted.)
            (2026-09-19, per explicit user correction "让concealment的动作也可以conceal...
            可以篡改pedmdetector观测到的观测和动作": auditing `(o_rep, a_v)` -- fake obs paired
            with the HONEST action -- left a structural hole. `a_v` is fixed by ground truth and
            Concealment cannot influence it at all, so whenever the true trajectory is actually
            near a hazard (Disruptor's doing), the victim's real evasive reaction shows up in
            `a_v` regardless of how good `o_rep` is -- PEDM was catching an inherent
            (o_rep, a_v) INCONSISTENCY, not evidence Concealment could ever fix by improving the
            observation forgery alone. Giving Concealment its own action-forgery channel
            (`a_rep = a_v + phi_hid_act`, PEDM-only, zero physical effect) closes that hole: now
            both halves of the audited record are under Concealment's control, so it can make
            them mutually consistent instead of being stuck reconciling a fake observation with
            an honest, hazard-revealing action.)

Both agents' obs/action spaces are PADDED to a common width (HARL's own vec-env stacking
requires uniform per-agent dims -- see `pettingzoo_mpe_env.py`'s `ss.pad_action_space_v0`/
`pad_observations_v0` for the same pattern in this repo). Real content lives in the first K
dims; the rest are zero and simply unused by the corresponding agent's own logic.

Reward: BOTH agents get the same per-step physical-effectiveness reward `haz_shaping` (dense
hazard-approach potential + one-off entry bonus, the already-validated design from
`examples/train_safety_attacker.py`) -- Safety-Gym's raw task reward is blind to hazard contact.

Stealth cost, two selectable sources via `stealth_source` env arg:

- "wm" (original, 2026-09-17 per "不要用pedmscore。用类似illusory的truedym"): per-STEP
  illusory-consistency cost against the ALREADY-FIT clean-dynamics model
  (`attacker_wm_v23_fulldim.pt`, MLPDyn) -- `cost_t = ||o_rep_t - (last_reported_obs +
  wm(last_reported_obs, last_env_action))|| / scale`. DIAGNOSED (2026-09-17/18) to have a
  structural blind spot: this signal stayed flat (~2-4) across training even as the REAL
  physical attack (true_cost) grew 15x (19->288) -- a smooth/predictable sustained
  hazard-dwelling trajectory scores just as "consistent" as a brief one, so constraining it
  provides no resistance to escalating attack duration/severity.

- "pedm_or" (2026-09-18, per explicit user redirect "尝试用pedm的orscore先验证prototype",
  CORRECTED 2026-09-18 per explicit user correction of the threat model): per-step cost is
  the INCREMENT to the running max of PEDM's (`pedm_detector_v23.pt`, independent,
  never-trained-against-here) per-transition anomaly score over the DECEIVER'S OWN FORGED
  RECORD -- `(o_rep_t[:60], a_v_t)` where `a_v_t` is the action the victim itself decided
  FROM `o_rep_t` -- NOT ground truth (raw env obs / the actually-executed `a_exec`).
  Concealment's whole job is to deceive whatever audits the victim's own reported
  experience; Disruptor's downstream action tampering (`a_exec = a_v + phi_dis`) never
  appears in that record at all, by design. `cost_t = max(0, pedm_score_t -
  running_max_before_this_step)`, checking whether `o_rep_t` is consistent with
  `(o_rep_{t-1}, a_v_{t-1})` -- this directly targets the OR (episode-max) criterion
  actually used for detection verdicts (see `_tmp_eval_scmarl_variants_vs_pedm.py`/
  `eval_harl_native_vs_pedm.py`), unlike a per-step MEAN-style signal which (as found
  repeatedly in this project) can look low while the OR/max stays over threshold. Cost is 0
  except at moments that would set a NEW episode peak, so GAE naturally learns to avoid
  ever spiking above the trajectory's running worst point instead of just averaging it down.
  DIAGNOSED (2026-09-18, after role-split let Disruptor's actual constraint pressure be
  isolated cleanly): this increment is nonzero on only ~1.2% of steps (measured on the clean
  baseline), so `lam*adv_c` multiplies against ~0 on ~99% of steps NO MATTER how large lambda
  gets -- Disruptor's real attack (`true_cost`) kept growing with Disruptor fully active even
  at lambda~6.5/20, because the cost-advantage signal it's actually judged against is almost
  always zero regardless of what it does.

- "pedm_dense" (2026-09-18, per explicit user follow-up after the pedm_or sparsity diagnosis
  above): same PEDM scoring as "pedm_or" but WITHOUT the running-max transform -- `cost_t =
  pedm_score_t` directly, every step, dense and nonzero whenever the (o_rep,a_v) transition
  looks anomalous at all. Trades away the "exact decomposition of OR" property (this compresses
  the per-step MEAN, not the peak -- the same "mean vs OR" gap found repeatedly elsewhere in
  this project) in exchange for a signal dense enough for lambda/GAE to have a REAL, every-step
  effect on both agents, not just on the rare peak-setting moment.

Delivered through `info[i]["cost"]` (both agents), NOT through the reward channel, so a
Lagrangian/hard-constraint runner can inject it into a SEPARATE cost-critic/GAE stream
without touching MAPPO's own reward-maximizing update at all.
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
from harl.detectors.pedm_detector import PEDMDetector


def _dist_to_hazard(env):
    task = env.unwrapped.task
    return float(np.linalg.norm(task.agent.pos[:2] - task.hazards.pos[0][:2]))


class SafetyMARLEnv:
    """2-agent (Disruptor, Concealment) env wrapping a frozen SafetyVictim + SafetyPointGoal1."""

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args.get("scenario", "SafetyPointGoal1Gymnasium-v0")
        self.disruptor_eps = float(args.get("disruptor_eps", 1.5))
        self.hidden_eps = float(args.get("hidden_eps", 0.2))
        # 2026-09-19: budget for Concealment's NEW action-forgery channel (a_rep = a_v +
        # phi_hid_act, PEDM-only, zero physical effect) -- see module docstring. Default 0.2,
        # symmetric with hidden_eps per explicit user instruction ("给都限定budget 0.2") -- both
        # forged fields of the (o_rep,a_rep) record get the same budget.
        self.hidden_act_eps = float(args.get("hidden_act_eps", 0.2))
        self.max_cycles = int(args.get("max_cycles", 1000))

        device = "cpu"
        self.victim = SafetyVictim(args["victim_ckpt"], device)
        self.full_dim = self.victim.full_obs_dim  # 64: raw+time+hazard
        self.act_dim = 2
        self.scale_full = np.maximum(np.sqrt(self.victim.obs_norm.var[:self.full_dim]).astype(np.float32), 0.01)
        # 2026-09-20, user-designed distribution-aware concealment parameterization: the obs-
        # forgery perturbation used to be scaled by `scale_full` (the VICTIM's own observation
        # normalizer) -- a plausible-looking but wrong reference, confirmed via direct
        # inspection: PEDM's own training-data variance for two specific goal_lidar bins was
        # ~70x smaller than neighboring bins (0.002-0.004 vs 0.02-0.25), a natural artifact of
        # how rarely the goal fell in those two angular bins during clean data collection --
        # something `scale_full` has no way to know about, since it's a completely different
        # statistic. A uniform-by-scale_full perturbation treats a small absolute move on a
        # near-constant dimension the same as an equal move on a highly variable one, but only
        # the former looks bizarre relative to what a real clean trajectory ever does. Fixed:
        # `clean_obs_robust_scale` (precomputed by `examples/compute_clean_obs_scale.py` from
        # clean reference rollouts, a percentile-range estimator with an absolute floor -- see
        # that script's docstring) replaces `scale_full` specifically for the PERTURBATION
        # magnitude, so the same actor output represents roughly the same "how many natural
        # clean-units did this move" on every dimension, not the same absolute delta. Detector-
        # agnostic by design: this never reads PEDM's own internal stats, only the environment's
        # own clean behavior (so unchanged if PEDM were swapped for a different detector).
        clean_scale_path = args.get(
            "clean_obs_scale_path",
            "results/safety_gym/SafetyPointGoal1/clean_obs_robust_scale.npy",
        )
        self.clean_obs_scale = np.load(clean_scale_path).astype(np.float32)
        assert self.clean_obs_scale.shape == (self.full_dim,), (
            f"clean_obs_scale shape {self.clean_obs_scale.shape} != full_dim ({self.full_dim},) "
            f"-- re-run examples.compute_clean_obs_scale if the victim/obs layout changed"
        )
        self.alpha_obs = float(args.get("alpha_obs", 1.0))

        self.stealth_source = args.get("stealth_source", "wm")
        if self.stealth_source == "wm":
            self.wm = MLPDyn.load(args["wm_ckpt"], device)
            assert self.wm.obs_dim == self.full_dim, (
                f"world model obs_dim={self.wm.obs_dim} != victim full_dim={self.full_dim}")
        elif self.stealth_source in ("pedm_or", "pedm_dense"):
            self.pedm = PEDMDetector(obs_dim=60, action_dim=self.act_dim, n_part=100, device=device)
            self.pedm.load(args["pedm_ckpt"])
        elif self.stealth_source == "traj_critic":
            pass  # no per-step cost computed here at all -- see module docstring section below
        else:
            raise ValueError(f"unknown stealth_source {self.stealth_source!r}")

        self.env = gym.make(self.scenario)
        self.lo = self.env.action_space.low.astype(np.float32)
        self.hi = self.env.action_space.high.astype(np.float32)

        self.n_agents = 2
        # Padded, UNIFORM per-agent spaces (HARL's vec-env stacking requires this -- see module
        # docstring). obs width = full_dim + act_dim: BOTH agents now observe [obs_aug, a_v]
        # (Concealment needs to see the true a_v to forge a plausible a_rep near it, exactly as
        # Disruptor already needed a_v to forge phi_dis near it). Action width = full_dim +
        # act_dim too (2026-09-19, widened from full_dim-only): Concealment's action now packs
        # BOTH the obs-perturbation (first full_dim dims, unchanged) AND the NEW action-forgery
        # perturbation (next act_dim dims). Disruptor still only reads its own first act_dim
        # dims of this same padded width; the rest is unused padding for it, as before.
        self.obs_pad = self.full_dim + self.act_dim
        self.act_pad = self.full_dim + self.act_dim
        obs_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_pad,), dtype=np.float32)
        act_box = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.act_pad,), dtype=np.float32)
        self.observation_space = [obs_box, obs_box]
        self.action_space = [act_box, act_box]
        self.share_observation_space = [obs_box, obs_box]

        self._seed = 0
        self.cur_step = 0
        self._ready = False

    def _local_obs(self, obs_aug, a_v):
        # Both agents observe [obs_aug, a_v] (2026-09-19): Concealment needs the true a_v to
        # forge a plausible a_rep near it, just as Disruptor needs it to forge phi_dis near it.
        local_obs = np.concatenate([obs_aug, a_v]).astype(np.float32)  # already obs_pad-wide
        return [local_obs.copy(), local_obs.copy()]

    def reset(self):
        self._seed += 1
        self.cur_step = 0
        raw_obs, _ = self.env.reset(seed=self._seed)
        self.steps_since_goal = 0
        self.last_dist_hazard = _dist_to_hazard(self.env)
        self.was_inside_hazard = False
        obs_aug = self.victim._augment(raw_obs.astype(np.float32), self.steps_since_goal, self.env)  # noqa: SLF001
        self.obs_aug = obs_aug
        # clean init: no deception has happened yet, so the "reported" state starts truthful
        self.last_reported_obs = obs_aug.copy()
        self.last_env_action = np.zeros(self.act_dim, np.float32)
        self.last_raw_obs = raw_obs.astype(np.float32)
        self.running_max_pedm = 0.0
        a_v0 = self.victim.act_from_augmented(obs_aug)
        self.last_a_v = a_v0.copy()
        # clean init: no forgery has happened yet, so the reported action starts truthful too
        self.last_a_rep = a_v0.copy()
        obs = self._local_obs(obs_aug, a_v0)
        share_obs = [obs_aug.copy() if False else np.concatenate([obs_aug, a_v0]).astype(np.float32)] * self.n_agents
        self._ready = True
        return obs, share_obs, self.get_avail_actions()

    def step(self, actions):
        # actions: (n_agents, act_pad) -- agent 0 = Disruptor (uses first act_dim dims),
        # agent 1 = Concealment (uses first full_dim dims for obs-forgery, next act_dim dims
        # for action-forgery).
        raw_dis = np.asarray(actions[0], np.float32)[:self.act_dim]
        raw_hid = np.asarray(actions[1], np.float32)
        raw_hid_obs = raw_hid[:self.full_dim]
        raw_hid_act = raw_hid[self.full_dim:self.full_dim + self.act_dim]

        # 2026-09-19, ATLA-inspired fix: tanh-SQUASH the raw actor output into the budget instead
        # of hard-clipping it. Hard clip has a flat (zero-gradient) region outside the budget --
        # once an actor's raw mean drifts out there (confirmed empirically for BOTH Disruptor and
        # Concealment), every sample lands on the exact same boundary corner regardless of
        # context, and nothing in the training signal can tell it to come back. tanh has no flat
        # region (its gradient only asymptotically -> 0, never exactly 0), so the action stays
        # continuously sensitive to the raw output's own magnitude even deep into the "saturated"
        # range. This does NOT include the SAC-style log-prob Jacobian correction for the
        # squashing transform -- log_prob/entropy/ratio are still computed on the raw
        # (pre-squash) Gaussian sample throughout, exactly as they were under hard-clip (which
        # made the identical simplification: consistent between collection and update, just not
        # a literal "log-prob of the executed action"). This matches the rigor level PPO
        # implementations commonly use when clipping/squashing actions externally, and avoids a
        # much larger, riskier change to HARL's shared distribution/update code.
        # 2026-09-20, distribution-aware parameterization (see __init__'s `clean_obs_scale`
        # comment): the actor output is squashed to u in [-1,1] first (same tanh mechanic as
        # before, still no flat/zero-gradient region), THEN scaled by `alpha_obs *
        # clean_obs_scale` -- so the SAME actor output represents roughly the same number of
        # "natural clean-units" of movement on every dimension, instead of the same absolute
        # delta. `hidden_eps` is kept as an absolute L∞ safety cap (rarely binds in practice once
        # clean_obs_scale itself is small almost everywhere, but stays as a hard backstop).
        u_obs = np.tanh(raw_hid_obs)
        phi_hid = np.clip(self.alpha_obs * self.clean_obs_scale * u_obs, -self.hidden_eps, self.hidden_eps)
        o_rep = self.obs_aug + phi_hid  # forged record ONLY -- never fed to the victim

        # ROLE-SPLIT (2026-09-18, standing instruction): victim acts on the TRUE observation --
        # Concealment's forgery has ZERO causal effect on the victim or the real trajectory, by
        # design. Disruptor's phi_dis is the ONLY thing that ever reaches the real actuator.
        a_v = self.victim.act_from_augmented(self.obs_aug)

        # Action-forgery channel (2026-09-19): a_rep is a SECOND fabrication, paired with o_rep
        # as "the deceiver's own forged record" that PEDM audits -- never used for a_exec, never
        # seen by the victim, exists purely to let Concealment make (o_rep, a_rep) mutually
        # consistent instead of being stuck pairing a fake observation with the honest a_v.
        phi_hid_act = self.hidden_act_eps * np.tanh(raw_hid_act)
        a_rep = np.clip(a_v + phi_hid_act, self.lo, self.hi)

        phi_dis = self.disruptor_eps * np.tanh(raw_dis)
        a_exec = np.clip(a_v + phi_dis, self.lo, self.hi)

        no, r, term, trunc, info = self.env.step(a_exec)
        self.cur_step += 1
        if self.cur_step >= self.max_cycles:
            trunc = True
        c = float(info.get("cost", 0.0))

        dh_new = _dist_to_hazard(self.env)
        haz_shaping = (self.last_dist_hazard - dh_new) * 1.0
        if c > 0 and not self.was_inside_hazard:
            haz_shaping += 10.0
        self.was_inside_hazard = c > 0
        self.last_dist_hazard = dh_new

        # Per-step stealth cost (module docstring: "wm" = illusory-consistency of the reported
        # o_rep trajectory; "pedm_or" = running-max-increment of an independent detector's score
        # on the DECEIVER'S OWN FORGED RECORD -- what the victim itself perceived (o_rep) and
        # believed it was doing (a_v), NOT ground truth. PEDM is auditing the victim's reported
        # experience, which is exactly what Concealment forges; Disruptor's downstream action
        # tampering (a_exec = a_v + phi_dis) never appears in that record at all, by design --
        # deceiving PEDM is entirely Concealment's job, independent of what Disruptor does.
        if self.stealth_source == "wm":
            # 2026-09-20 bug fix (user-caught, via "那用truedyn/worldmodel而不是pedm"): this
            # branch was NEVER updated when pedm_or/pedm_dense were fixed to audit the
            # DECEIVER'S OWN FORGED RECORD (o_rep, a_rep) instead of ground truth -- it was still
            # feeding `self.last_env_action` (= a_exec, the REAL executed action, INCLUDING
            # Disruptor's actual tampering) into the world model, checking "is o_rep consistent
            # with what REALLY happened" instead of "is o_rep consistent with Concealment's own
            # self-reported (o_rep, a_rep) narrative". That's an even more direct threat-model
            # violation than the old a_v bug: comparing against the true executed action makes
            # concealment nearly impossible by construction, independent of how good the forgery
            # actually is. Fixed to use `self.last_a_rep`, matching pedm_or/pedm_dense's own fix.
            with torch.no_grad():
                x = torch.as_tensor(np.concatenate([self.last_reported_obs, self.last_a_rep])[None], dtype=torch.float32)
                expected = self.last_reported_obs + self.wm(x)[0].numpy()
            stealth_cost = float(np.linalg.norm((o_rep - expected) / self.scale_full))
        elif self.stealth_source in ("pedm_or", "pedm_dense"):
            # Checks whether o_rep_t follows from (o_rep_{t-1}, a_rep_{t-1}) -- a_rep_{t-1} is
            # Concealment's OWN forged action reported as having "caused" this transition (NOT
            # a_v_{t-1}, the honest action -- 2026-09-19 fix, see module docstring); a_rep (just
            # computed above, from this step) hasn't caused anything yet and belongs to the NEXT
            # transition instead.
            score = float(self.pedm.predict_scores(
                np.stack([self.last_reported_obs[:60], o_rep[:60].astype(np.float32)]),
                self.last_a_rep[None].astype(np.float32),
            )[0])
            if self.stealth_source == "pedm_or":
                stealth_cost = max(0.0, score - self.running_max_pedm)
                self.running_max_pedm = max(self.running_max_pedm, score)
            else:  # pedm_dense (2026-09-18): raw per-step score, no running-max transform --
                # gives Disruptor a real, non-sparse gradient every single step (pedm_or's
                # running-max increment is nonzero on only ~1.2% of steps, so lambda*adv_c was
                # multiplying against ~0 almost always, no matter how large lambda got).
                # Accepts the "compresses the MEAN, not the OR/peak" tradeoff this project has
                # run into before, in exchange for a signal dense enough for GAE/lambda to
                # actually act on every step.
                stealth_cost = score
        else:  # traj_critic -- no per-step cost; the runner assembles/scores whole episodes
            stealth_cost = 0.0
        self.last_reported_obs = o_rep.copy()
        self.last_env_action = a_exec.copy()
        self.last_raw_obs = no.astype(np.float32)
        self.last_a_v = a_v.copy()
        self.last_a_rep = a_rep.copy()

        self.steps_since_goal = 0 if info.get("goal_met", False) else self.steps_since_goal + 1
        obs_aug_next = self.victim._augment(no.astype(np.float32), self.steps_since_goal, self.env)  # noqa: SLF001
        self.obs_aug = obs_aug_next
        a_v_next = self.victim.act_from_augmented(obs_aug_next)

        obs = self._local_obs(obs_aug_next, a_v_next)
        share_obs = [np.concatenate([obs_aug_next, a_v_next]).astype(np.float32)] * self.n_agents
        done = bool(term or trunc)
        rewards = [[float(haz_shaping)], [float(haz_shaping)]]
        # o_rep_scaled/a_rep: the deceiver's own DOUBLY-forged record for this step (both halves
        # under Concealment's control, 2026-09-19), always exposed (cheap) so a
        # TrajectoryCritic-style runner-side mechanism can assemble whole-episode sequences from
        # the info stream regardless of stealth_source. `a_v` (honest action) is exposed too,
        # for scripts that still need it (e.g. tracking real physical behavior), but PEDM/
        # traj_critic auditing must use `a_rep`, never `a_v`.
        info_common = {
            "cost": stealth_cost, "true_cost": c, "bad_transition": bool(trunc and not term),
            "o_rep_scaled": (o_rep / self.scale_full).astype(np.float32),
            "a_v": a_v.astype(np.float32),
            "a_rep": a_rep.astype(np.float32),
        }
        infos = [dict(info_common), dict(info_common)]
        dones = [done, done]
        return obs, share_obs, rewards, dones, infos, self.get_avail_actions()

    def get_avail_actions(self):
        return None  # continuous action spaces only

    def seed(self, seed):
        self._seed = seed

    def render(self):
        pass

    def close(self):
        self.env.close()
