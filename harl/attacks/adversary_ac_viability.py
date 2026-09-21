"""Dual-critic actor-critic for the viability-aware act attacker.

Adds a SECOND value head `critic_fake` to the standard illusory adversary net.
The inherited `actor` + `critic` (the DAMAGE critic, estimating the return of
-r_victim) are untouched; the new head estimates

    V_fake(s) = E[ sum_k gamma^k c_{t+k} ]   (cost-to-go of the per-step
                concealment cost c_t = PEDM epistemic residual),

i.e. "how much unavoidable trajectory-discrepancy will accrue in the future if
we keep attacking from here". The act attacker then optimizes damage MINUS a
dual-weighted future-concealment advantage (see train_marl_viability.py), which
is the concrete realization of "maximize damage subject to staying in the
deception-viability kernel". The obs/deceiver agent keeps the single-critic
AdversaryActorCritic unchanged.
"""
import torch
import torch.nn as nn

from harl.attacks.adversary_ac import AdversaryActorCritic, orthogonal_init


class AdversaryActorCriticViability(AdversaryActorCritic):
    """AdversaryActorCritic + an extra value head V_fake (future concealment cost)."""

    def __init__(self, obs_dim, action_dim, hidden_size=256, activation="tanh"):
        super().__init__(obs_dim, action_dim, hidden_size, activation)
        act = nn.Tanh if activation == "tanh" else nn.ReLU
        self.critic_fake = nn.Sequential(
            orthogonal_init(nn.Linear(obs_dim, hidden_size)),
            act(),
            orthogonal_init(nn.Linear(hidden_size, hidden_size)),
            act(),
            orthogonal_init(nn.Linear(hidden_size, 1), gain=1.0),
        )

    def get_fake_value(self, obs):
        return self.critic_fake(obs).squeeze(-1)

    @torch.no_grad()
    def act_v(self, obs, deterministic=False):
        """Rollout step: return action, log_prob, V_damage, V_fake."""
        dist = self._distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        v_damage = self.critic(obs).squeeze(-1)
        v_fake = self.critic_fake(obs).squeeze(-1)
        return action, log_prob, v_damage, v_fake

    def evaluate_actions_v(self, obs, action):
        """PPO update: recompute log_prob, entropy, V_damage, V_fake."""
        dist = self._distribution(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        v_damage = self.critic(obs).squeeze(-1)
        v_fake = self.critic_fake(obs).squeeze(-1)
        return log_prob, entropy, v_damage, v_fake
