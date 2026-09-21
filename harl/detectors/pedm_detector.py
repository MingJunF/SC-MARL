"""PEDM anomaly detector.

Given sequential (observation, action) transitions, the detector uses the
probabilistic ensemble dynamics model to produce an anomaly score per
transition. High scores indicate that an observed transition is inconsistent
with the learned environment dynamics, which is the signal used to flag
adversarially perturbed observations (out-of-distribution transitions).

Reference: Haider et al., "Out-of-Distribution Detection for Reinforcement
Learning Agents with Probabilistic Dynamics Models" and Franzmeyer et al.,
"Illusory Attacks: Information-theoretic detectability matters in adversarial
attacks" (ICLR 2024).
"""

import numpy as np
import torch
from scipy.special import ndtr
from torch.distributions.multivariate_normal import MultivariateNormal

from harl.detectors.pedm import PEDM


def one_step_batch_stats(preds, targets):
    """Per-dimension prediction error and particle spread.

    The arrays are kept 3-D so the aggregation function can reduce over the
    observation dimension (mean) and then over particles (e.g. min).

    Args:
        preds: (seq_len, n_part, obs_dim) sampled predictions
        targets: (seq_len, obs_dim) observed next states
    Returns:
        pred_err: (seq_len, n_part, obs_dim) absolute error per dimension
        pred_std: (seq_len, n_part, obs_dim) per-particle deviation from mean
    """
    targets = targets[:, None, :]  # (seq_len, 1, obs_dim)
    pred_err = np.abs(preds - targets)  # (seq_len, n_part, obs_dim)
    pred_std = np.abs(preds - preds.mean(axis=1, keepdims=True))
    return pred_err, pred_std


class PEDMDetector:
    """Probabilistic ensemble dynamics model OOD detector."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        ens_size=5,
        hidden_sizes=(200, 200, 200),
        lr=1e-3,
        n_part=100,
        criterion="pred_error_samples",
        aggregation_function="min_mean",
        device="cpu",
    ):
        self.dyn_model = PEDM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ens_size=ens_size,
            hidden_sizes=hidden_sizes,
            lr=lr,
            device=device,
        )
        self.n_part = n_part
        self.criterion = criterion
        self.aggregation_function = aggregation_function
        self.threshold = None

    def set_dyn_model(self, dyn_model: PEDM):
        """Reuse an already-trained dynamics model (e.g. the attacker's)."""
        self.dyn_model = dyn_model

    def fit(
        self,
        obs,
        actions,
        next_obs,
        n_train_epochs=200,
        batch_size=512,
        verbose=False,
    ):
        """Train the dynamics model on clean transitions."""
        return self.dyn_model.fit_transitions(
            obs,
            actions,
            next_obs,
            n_train_epochs=n_train_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def predict_scores(self, obs, acts) -> np.ndarray:
        """Anomaly score for each transition in a sequence.

        Args:
            obs: (seq_len, obs_dim) sequential observations
            acts: (seq_len - 1, action_dim) sequential actions
        Returns:
            scores: (seq_len - 1,) anomaly score per transition
        """
        obs = np.asarray(obs, dtype=np.float32)
        acts = np.asarray(acts, dtype=np.float32)

        if self.criterion in {"pred_error_samples", "pred_std_samples"}:
            preds = self.dyn_model.one_step_batch_preds(
                states=obs[:-1], actions=acts, n_part=self.n_part
            )
            pred_err, pred_std = one_step_batch_stats(preds, obs[1:])
            metric = pred_err if self.criterion == "pred_error_samples" else pred_std
            return self._aggregate(metric)

        if self.criterion == "pred_error_pdf":
            mean, var = self.dyn_model.predict_mean_var(
                states=obs[:-1], actions=acts, n_part=self.dyn_model.ens_size
            )
            n_obs = torch.from_numpy(obs[1:]).to(mean.device).float()
            log_probs = []
            for i in range(self.dyn_model.ens_size):
                m, v = mean[:, i, :], var[:, i, :]
                lp = MultivariateNormal(m, torch.diag_embed(v)).log_prob(n_obs)
                log_probs.append(lp.cpu().numpy())
            log_probs = np.vstack(log_probs)
            return -self._aggregate(log_probs)

        if self.criterion == "p_value":
            mean, var = self.dyn_model.predict_mean_var(
                states=obs[:-1], actions=acts, n_part=self.dyn_model.ens_size
            )
            mean = mean.cpu().numpy()
            std = var.sqrt().cpu().numpy() * 10
            n_obs = obs[1:]
            p_values = []
            for i in range(self.dyn_model.ens_size):
                m, s = mean[:, i, :], std[:, i, :]
                z = (n_obs - m) / s
                p = np.prod(ndtr(-np.abs(z)), axis=1)
                p_values.append(p)
            p_values = np.array(p_values)
            return -self._aggregate(p_values)

        raise ValueError(f"unknown criterion: {self.criterion}")

    def _aggregate(self, X) -> np.ndarray:
        """Aggregate per-particle / per-member scores into a per-step score.

        For the sample-based criteria ``X`` is (seq_len, n_part, obs_dim): the
        observation dimension is reduced with a mean and the particle axis with
        the leading reduction (e.g. ``min`` for ``min_mean``). For the
        distribution-based criteria ``X`` is (ens_size, seq_len) and only the
        ensemble axis is reduced. Both return a (seq_len,) array.
        """
        reducers = {
            "min": lambda a, ax: np.min(a, axis=ax),
            "max": lambda a, ax: np.max(a, axis=ax),
            "mean": lambda a, ax: np.mean(a, axis=ax),
            "median": lambda a, ax: np.median(a, axis=ax),
        }
        leading = self.aggregation_function.split("_")[0]
        if leading not in reducers:
            raise NotImplementedError(
                f"unknown aggregation: {self.aggregation_function}"
            )

        if self.criterion in {"pred_error_samples", "pred_std_samples"}:
            # (seq_len, n_part, obs_dim) -> mean over dims -> (seq_len, n_part)
            per_particle = np.mean(X, axis=-1)
            # reduce particle axis with the leading reduction -> (seq_len,)
            return reducers[leading](per_particle, -1)
        # (ens_size, seq_len) -> reduce ensemble axis -> (seq_len,)
        return reducers[leading](X, 0)

    def calibrate_threshold(self, clean_scores, quantile=0.95):
        """Set the detection threshold from clean-trajectory scores."""
        clean_scores = np.concatenate([np.ravel(s) for s in clean_scores])
        self.threshold = float(np.quantile(clean_scores, quantile))
        return self.threshold

    def detect(self, obs, acts):
        """Return per-transition boolean anomaly flags."""
        scores = self.predict_scores(obs, acts)
        if self.threshold is None:
            raise RuntimeError("threshold not calibrated; call calibrate_threshold")
        return scores > self.threshold

    def save(self, path):
        self.dyn_model.save(path)

    def load(self, path):
        self.dyn_model.load(path)
