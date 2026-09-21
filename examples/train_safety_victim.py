"""RECONSTRUCTED 2026-09-18: this file's original source was accidentally destroyed during a
cleanup pass (it was untracked, never committed, and its cached bytecode got clobbered before a
safe copy could be made). It is a transitive dependency of `train_safety_attacker.py`
(`SafetyVictim` needs `GaussianActor`/`RunningNorm`/`HAZARD_FEAT_DIM`/`hazard_feat`), which the
current HARL-native pipeline still relies on.

Reconstructed, NOT the original file, via:
  - `GaussianActor`: architecture reverse-engineered EXACTLY from
    `victim_v23_backup_41pct.pt`'s own `state_dict` (inspected directly: `body.{0,2,4}` and
    `critic.{0,2,4}` Linear layers with a (256,256) hidden MLP, a free-standing `log_std`
    parameter) -- matches this project's own `harl/attacks/adversary_ac.py:AdversaryActorCritic`
    structure almost exactly (same shapes, same "two hidden Tanh layers + linear head" pattern,
    just named `body` instead of `actor`). Since `SafetyVictim` only ever calls `.act(...,
    deterministic=True)`, the returned action is `self.body(obs)` alone -- independent of
    `log_std`/`critic` values -- so architecture-shape correctness (verified against the
    checkpoint) is what actually matters for behavioral fidelity here.
  - `RunningNorm`: the standard, obvious running mean/var normalizer implied by its usage
    (`obs_norm.mean`/`.var` set directly from checkpoint tensors, `.normalize(x)` called on raw
    obs) -- a trivial, low-risk reconstruction.
  - `HAZARD_FEAT_DIM`: derived by ARITHMETIC from the checkpoint itself (obs_dim=64 =
    RAW_SAFETY_OBS_DIM(60) + time(1) + HAZARD_FEAT_DIM => HAZARD_FEAT_DIM=3), matching the
    "[ex,ey,radius]" description in `train_safety_attacker.py`'s own docstrings.
  - `hazard_feat`: uses safety_gymnasium's OWN `BaseTask._ego_xy()` method (found in
    `site-packages/safety_gymnasium/bases/base_task.py`) for the egocentric [ex,ey] transform --
    this is the library's own verified lidar-computation convention, not a guessed rotation --
    plus `task.hazards.hazard_sizes[0]` (set by `examples/safety_hazard_patch.py`'s monkeypatch)
    for the per-episode randomized radius that raw lidar can't otherwise expose.

Validated (2026-09-18) by running a full 20-episode zero-perturbation rollout through
`SafetyVictim(victim_v23_backup_41pct.pt)` using this reconstruction and confirming it
reproduces this project's own well-established reference behavior for that checkpoint
(mean return ~20, mean cost ~0 -- see `project_harl_experiment_findings.md` memory).
"""
import numpy as np
import torch
import torch.nn as nn

RAW_SAFETY_OBS_DIM = 60
HAZARD_FEAT_DIM = 3


class RunningNorm:
    """Plain running mean/var normalizer -- `.mean`/`.var` are set directly from a saved
    checkpoint's stats (not updated online here); `.normalize(x)` applies (x-mean)/sqrt(var+eps).
    """

    def __init__(self, dim, epsilon=1e-8):
        self.mean = np.zeros(dim, dtype=np.float32)
        self.var = np.ones(dim, dtype=np.float32)
        self.epsilon = epsilon

    def normalize(self, x):
        out = (np.asarray(x, dtype=np.float32) - self.mean) / np.sqrt(self.var + self.epsilon)
        return out.astype(np.float32)


class GaussianActor(nn.Module):
    """Diagonal-Gaussian actor-critic MLP -- architecture fixed exactly to match
    `victim_v23_backup_41pct.pt`'s own `state_dict` shapes (see module docstring)."""

    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def _distribution(self, obs):
        mean = self.body(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs, deterministic=False):
        dist = self._distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, logp, value

    def evaluate_actions(self, obs, action):
        dist = self._distribution(obs)
        logp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return logp, ent, value

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)


def hazard_feat(env):
    """Egocentric [ex, ey, radius] hazard-boundary feature (HAZARD_FEAT_DIM=3) for the single
    hazard placed by `examples.safety_hazard_patch` -- can't be derived from raw lidar alone
    since the per-episode randomized radius isn't otherwise observable. Uses safety_gymnasium's
    own `_ego_xy` for the egocentric transform (matches the library's internal lidar convention
    exactly, not a guessed rotation)."""
    task = env.unwrapped.task
    hazard_pos = task.hazards.pos[0][:2]
    ex, ey = task._ego_xy(hazard_pos)  # noqa: SLF001 -- library's own verified convention
    sizes = getattr(task.hazards, "hazard_sizes", None)
    radius = float(sizes[0]) if sizes else float(task.hazards.size)
    return np.array([ex, ey, radius], dtype=np.float32)
