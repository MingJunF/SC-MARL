"""Actor-critic network for the illusory-attack adversary.

PyTorch port of the Gaussian actor-critic used by the illusory-attack
adversary. The adversary observes a concatenation of the true observation, the
last perturbed observation and the dynamics-model-predicted expected
observation, and outputs a perturbation vector of the observation dimension.
"""

import numpy as np
import torch
import torch.nn as nn


def orthogonal_init(layer, gain=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class AdversaryActorCritic(nn.Module):
    """Gaussian policy with a separate value head.

    Args:
        obs_dim: dimension of the adversary observation (typically 3 * victim
            observation dim).
        action_dim: dimension of the perturbation output (victim observation
            dim).
        hidden_size: width of the hidden layers.
        activation: "tanh" or "relu".
    """

    def __init__(self, obs_dim, action_dim, hidden_size=256, activation="tanh"):
        super().__init__()
        act = nn.Tanh if activation == "tanh" else nn.ReLU

        self.actor = nn.Sequential(
            orthogonal_init(nn.Linear(obs_dim, hidden_size)),
            act(),
            orthogonal_init(nn.Linear(hidden_size, hidden_size)),
            act(),
            orthogonal_init(nn.Linear(hidden_size, action_dim), gain=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            orthogonal_init(nn.Linear(obs_dim, hidden_size)),
            act(),
            orthogonal_init(nn.Linear(hidden_size, hidden_size)),
            act(),
            orthogonal_init(nn.Linear(hidden_size, 1), gain=1.0),
        )

    def _distribution(self, obs):
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Sample an action and value for rollout collection.

        Returns:
            action, log_prob, value (all torch tensors)
        """
        dist = self._distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, action):
        """Recompute log-prob, entropy and value for a PPO update."""
        dist = self._distribution(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value
