"""Probabilistic Ensemble Dynamics Model (PEDM).

PyTorch port of the state-action dynamics model used by the PEDM-OOD detector,
adapted for HARL. The model predicts the distribution over the next observation
given the current observation and action. It is trained on clean (unattacked)
victim trajectories and reused both as a forward model for the illusory-attack
reward and as the backbone of the anomaly detector.
"""

from typing import Tuple

import numpy as np
import torch

from harl.detectors.prob_ensemble import ProbEnsemble


class PEDM(ProbEnsemble):
    """Probabilistic ensemble dynamics model with state-action inputs."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        ens_size=5,
        hidden_sizes=(200, 200, 200),
        decays=None,
        lr=1e-3,
        normalize_data=True,
        activation_fn="swish",
        device="cpu",
    ):
        layer_sizes = [obs_dim + action_dim, *hidden_sizes, obs_dim]
        super().__init__(
            ens_size=ens_size,
            layer_sizes=layer_sizes,
            decays=decays,
            normalize_data=normalize_data,
            activation_fn=activation_fn,
            lr=lr,
            device=device,
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # predict the residual (next_obs - obs) and add it back
        self.obs_preproc = lambda obs: obs
        self.obs_postproc = lambda obs, pred: obs + pred
        self.targ_proc = lambda obs, n_obs: n_obs - obs

    def fit_transitions(
        self,
        obs,
        actions,
        next_obs,
        val_fraction=0.1,
        n_train_epochs=200,
        batch_size=512,
        verbose=False,
    ):
        """Fit the model on flat transition arrays.

        Args:
            obs: (N, obs_dim)
            actions: (N, action_dim)
            next_obs: (N, obs_dim)
        """
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)

        X = np.concatenate([self.obs_preproc(obs), actions], axis=-1)
        y = self.targ_proc(obs, next_obs)

        n = len(X)
        perm = np.random.permutation(n)
        X, y = X[perm], y[perm]
        n_val = int(n * val_fraction)
        X_val, y_val = X[:n_val], y[:n_val]
        X_train, y_train = X[n_val:], y[n_val:]

        return self.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            n_train_epochs=n_train_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    @torch.no_grad()
    def predict_next_state(
        self, state: torch.Tensor, action: torch.Tensor, n_part: int
    ) -> torch.Tensor:
        """Sample next-state predictions distributing particles over members."""
        _state = self._expand(self.obs_preproc(state), n_part)
        _acs = self._expand(action, n_part)
        inputs = torch.cat((_state, _acs), dim=-1)

        mean, var = self.forward(inputs)
        predictions = mean + torch.randn_like(mean, device=self.device) * var.sqrt()
        predictions = self._flatten(predictions, n_part)
        return self.obs_postproc(state, predictions)

    @torch.no_grad()
    def predict_next_obs_mean(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Deterministic mean next-observation prediction (ensemble average).

        Used as the forward model for the illusory-attack reward.

        Args:
            obs: (batch, obs_dim)
            actions: (batch, action_dim)
        Returns:
            (batch, obs_dim) predicted next observation
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        # (batch, dim) -> (ens_size, batch, dim)
        state = self.obs_preproc(obs_t).unsqueeze(0).repeat(self.ens_size, 1, 1)
        acs = act_t.unsqueeze(0).repeat(self.ens_size, 1, 1)
        inputs = torch.cat((state, acs), dim=-1)
        mean, _ = self.forward(inputs)  # (ens_size, batch, obs_dim)
        pred = obs_t + mean.mean(dim=0)
        return pred.cpu().numpy()

    @torch.no_grad()
    def one_step_batch_preds(
        self, states: np.ndarray, actions: np.ndarray, n_part: int
    ) -> np.ndarray:
        """Sampled next-state predictions for a batch of transitions.

        Returns:
            (n, n_part, obs_dim)
        """
        st = torch.from_numpy(states).float().to(self.device).detach()
        st = st.repeat_interleave(repeats=n_part, dim=0)
        act = torch.from_numpy(actions).float().to(self.device).detach()
        act = act.repeat_interleave(repeats=n_part, dim=0)
        preds = self.predict_next_state(state=st, action=act, n_part=n_part)
        preds = self._unflatten(preds, n_part=n_part)
        preds = preds.reshape(states.shape[0], n_part, -1)
        return preds.cpu().numpy()

    @torch.no_grad()
    def predict_mean_var(
        self, states: np.ndarray, actions: np.ndarray, n_part=5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of the next state for each ensemble member.

        Returns:
            (mean, var) each shaped (n, n_part, obs_dim)
        """
        st = torch.from_numpy(states).float().to(self.device).detach()
        st = st.repeat_interleave(repeats=n_part, dim=0)
        act = torch.from_numpy(actions).float().to(self.device).detach()
        act = act.repeat_interleave(repeats=n_part, dim=0)

        _state = self._expand(self.obs_preproc(st), n_part)
        _acs = self._expand(act, n_part)
        inputs = torch.cat((_state, _acs), dim=-1)

        mean, var = self.forward(inputs)

        mean = self._flatten(mean, n_part)
        mean = self.obs_postproc(st, mean)
        mean = self._unflatten(mean, n_part=n_part)
        mean = mean.reshape(states.shape[0], n_part, -1)
        var = self._unflatten(var, n_part=n_part)
        var = var.reshape(states.shape[0], n_part, -1)
        return mean, var

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "ens_size": self.ens_size,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
