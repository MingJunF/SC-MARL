"""`SafetyVictim`: loads a `train_safety_victim.py`-format checkpoint (GaussianActor state_dict +
RunningNorm obs stats) for the SafetyPointGoal1 victim -- a small wrapper analogous to
`eval_robust_detector.Victim` but for this project's own plain-PPO checkpoint format (not HARL's
StochasticPolicy/MAPPO directory format). This is the only thing the current HARL-native
pipeline (`harl/envs/safety_marl/`) actually needs from this file.

TRIMMED 2026-09-18 (per "把前面的所有乱七八杂的版本和代码全删了。只保留现在的版本"): this file
used to also contain a full standalone obs-only/act-only attacker PPO training loop
(`LinfBoxPolicy`, `train()`, `main()`) from 2026-09-11/17 -- fully superseded by the HARL-native
`SafetyMARLEnv`/`OnPolicyLagrRunner` pipeline, so it was removed along with its own now-unused
imports (`AdversaryActorCritic`, `RadiusDirPolicy`, `gae`, `MLPDyn`) to avoid a stale dependency
on the deleted `diag_zero_action_learning.py`/`vibe_train_act_b4.py` chain.
"""
import numpy as np
import torch

import safety_gymnasium  # noqa: F401 -- registers Safety* env ids
import examples.safety_hazard_patch  # noqa: F401 -- randomized hazard size + flat entry cost
import examples.safety_layout_patch  # noqa: F401 -- corner spawn/goal + guaranteed-blocking hazard
from examples.train_safety_victim import GaussianActor, RunningNorm, HAZARD_FEAT_DIM, hazard_feat

RAW_SAFETY_OBS_DIM = 60  # SafetyPointGoal1's native observation width (no time feature)


class SafetyVictim:
    """Loads a `train_safety_victim.py` checkpoint (GaussianActor + RunningNorm obs stats),
    deterministic actions only -- analogous to eval_robust_detector.Victim but for this
    project's own plain-PPO checkpoint format, not HARL's StochasticPolicy/MAPPO directory.

    Time-aware victims (2026-09-16, obs_dim==RAW_SAFETY_OBS_DIM+1): auto-detected from the
    checkpoint's obs_dim. For these, every `act`/`act_batch` call MUST also pass
    `steps_since_goal` (steps since this env/batch-slot last reached a goal, as of BEFORE this
    action -- the same per-env counter every rollout loop in this project's Safety-Gym scripts
    already maintains for the speed_bonus reward) so the raw env observation can be augmented
    with `min(steps_since_goal/expected_steps_per_leg, 2.0)` exactly as `train_safety_victim.py`
    does. Non-time-aware victims ignore the extra argument entirely -- fully backward compatible.

    Hazard-aware victims (2026-09-16, obs_dim==RAW_SAFETY_OBS_DIM+1+HAZARD_FEAT_DIM): ALSO
    require an `env` argument (a single env for `act()`, or a list of envs matching the batch
    for `act_batch()`) so the egocentric `[ex, ey, radius]` hazard-boundary feature
    (`train_safety_victim.hazard_feat`) can be computed fresh each call -- this can't be derived
    from `obs` alone since the raw lidar observation doesn't expose hazard radius (see
    `hazard_feat`'s docstring for why)."""

    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.net = GaussianActor(ck["obs_dim"], ck["act_dim"]).to(device)
        self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        self.obs_norm = RunningNorm(ck["obs_dim"])
        self.obs_norm.mean = ck["obs_norm_mean"]; self.obs_norm.var = ck["obs_norm_var"]
        self.lo, self.hi = ck["lo"], ck["hi"]
        self.device = device
        self.hazard_aware = (ck["obs_dim"] == RAW_SAFETY_OBS_DIM + 1 + HAZARD_FEAT_DIM)
        self.time_aware = self.hazard_aware or (ck["obs_dim"] == RAW_SAFETY_OBS_DIM + 1)
        self.full_obs_dim = ck["obs_dim"]  # what net.act actually expects (raw+time+hazard)
        self.expected_steps_per_leg = 100.0

    def _augment(self, obs, steps_since_goal, env):
        if not self.time_aware:
            return obs
        if steps_since_goal is None:
            raise ValueError("this victim is time-aware (obs_dim>=RAW+1) -- "
                              "act()/act_batch() require steps_since_goal")
        t = np.minimum(np.asarray(steps_since_goal, np.float32) / self.expected_steps_per_leg, 2.0)
        if not self.hazard_aware:
            if obs.ndim == 1:
                return np.concatenate([obs, [float(t)]]).astype(np.float32)
            return np.concatenate([obs, t.reshape(-1, 1)], axis=1).astype(np.float32)
        if env is None:
            raise ValueError("this victim is hazard-aware (obs_dim=RAW+1+HAZARD_FEAT_DIM) -- "
                              "act()/act_batch() require env (a single env, or a list matching the batch)")
        if obs.ndim == 1:
            hz = hazard_feat(env)
            return np.concatenate([obs, [float(t)], hz]).astype(np.float32)
        hzs = np.stack([hazard_feat(e) for e in env], axis=0)
        return np.concatenate([obs, t.reshape(-1, 1), hzs], axis=1).astype(np.float32)

    @torch.no_grad()
    def act(self, obs, steps_since_goal=None, env=None):
        obs = self._augment(obs, steps_since_goal, env)
        obs_n = self.obs_norm.normalize(obs[None] if obs.ndim == 1 else obs)
        a, _, _ = self.net.act(torch.as_tensor(obs_n, device=self.device), deterministic=True)
        out = np.clip(a.cpu().numpy(), self.lo, self.hi)
        return out[0] if obs.ndim == 1 else out

    def act_batch(self, obs, steps_since_goal=None, env=None):
        return self.act(obs, steps_since_goal, env)

    @torch.no_grad()
    def act_from_augmented(self, obs_aug):
        """Like act()/act_batch(), but obs_aug is ALREADY the full augmented vector (raw+time+
        hazard, `full_obs_dim` wide) the network expects -- used by an obs-channel attacker that
        perturbs this full vector directly (2026-09-17). This is the ONLY way an obs-attacker can
        reach the hazard_feat channel: `_augment()` always re-derives hazard_feat from ground
        truth given a raw obs, so forging the raw 60-dim obs and calling act()/act_batch() can
        never touch the radius/ex/ey dims -- see the dose-response finding in memory that a pure
        radius-channel deception (budget~1.4) is far more effective than any raw-obs attacker
        found."""
        obs_n = self.obs_norm.normalize(obs_aug[None] if obs_aug.ndim == 1 else obs_aug)
        a, _, _ = self.net.act(torch.as_tensor(obs_n, device=self.device), deterministic=True)
        out = np.clip(a.cpu().numpy(), self.lo, self.hi)
        return out[0] if obs_aug.ndim == 1 else out
