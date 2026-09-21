"""COTD-style CVAE-ensemble OOD detector (2026-09-22).

Reproduction of the detector from "Out-Of-Distribution Detection in Deep RL via Transition
Estimation" (arXiv:2503.05238, called "COTD" in that paper's own results tables), added as a
second, architecturally-different detector to test the role-split SC-MARL attack against --
everything else in this project so far only evaluates against PEDM (a deterministic
probabilistic-ensemble forward dynamics model, `harl/detectors/pedm_detector.py`).

What the paper actually specifies (confirmed by reading the paper directly, not guessed):
  - A CVAE ENSEMBLE is trained on NOMINAL (clean, on-policy) transitions only. The encoder is
    q(z | s2, s1) and the decoder is p(s2 | z, s1) -- i.e. it is a RECONSTRUCTION-based score:
    at test time you already have the REAL next state s2 in hand (this is passive post-hoc
    monitoring, not a forward prediction), pass it through the encoder to get z, decode back
    through the (s1-conditioned) decoder, and see how well s2 reconstructs. This is a
    fundamentally different anomaly-detection paradigm from PEDM, which never sees s2 --
    PEDM predicts s2 from (s1,a1) alone and scores the prediction error. COTD instead asks
    "is this observed s2 the kind of thing this (s1)-conditioned decoder can explain," which
    can catch anomalies a forward predictor's own noise floor would wash out.
  - Anomaly score = MSE reconstruction error, averaged over the ensemble.
  - Threshold is set via INDUCTIVE (split) CONFORMAL PREDICTION on a held-out calibration set
    of clean transitions: pool M calibration transitions x N ensemble members into M*N scores,
    sort them, and take the order statistic at the appropriate conformal quantile -- giving a
    finite-sample marginal coverage guarantee on held-out clean data, not just an empirical
    quantile.
  - Detection in the paper is PER-TRANSITION (episode ends the instant one transition is
    flagged) -- no windowing, no CUSUM.

What the paper does NOT specify (confirmed absent, not overlooked): exact layer sizes,
activation functions, the reconstruction/KL loss balance (beta), how many ensemble members N,
and the exact per-transition ensemble-aggregation rule (mean vs something else) before/after
conformal pooling. The choices below fill those gaps with this project's own established
defaults (matching `harl/detectors/pedm.py`'s own ens_size=5, hidden_sizes convention, swish
activation) so results are as directly comparable to PEDM as the architecture difference
allows -- NOT a byte-exact reproduction, a faithful-in-spirit one. Differences from the
paper's own protocol, made deliberately to plug into this project's existing OR/CUSUM
evaluation harness (`eval_mujoco_marl_vs_pedm.py`, `eval_illusory_matrix.py`) instead of the
paper's own "flag one transition -> end episode" protocol:
  - `predict_scores()` returns the ENSEMBLE-MEAN per-step score (not the raw M*N per-member
    grid) so it drops straight into the same OR (episode-max) / CUSUM (accumulated) evaluation
    already used for PEDM -- the paper's own "any single flagged transition" protocol is
    exactly what this project's OR statistic already computes (was the per-episode peak ever
    over threshold), so nothing about the paper's actual detection semantics is lost by this
    substitution, only the M*N-grid version of the calibration internals.
  - `calibrate_threshold()` still performs genuine split-conformal calibration (pools clean
    per-step scores across held-out episodes and takes the corrected order statistic, not a
    naive `np.quantile`), matching the paper's OWN calibration mechanism, just applied to the
    already ensemble-averaged score stream instead of the raw M*N grid.
  - The conditioning input is (obs, action) jointly (matching PEDM's own convention), not just
    the paper's own "s1" (which implicitly encodes the action only because the paper always
    rolls out a FIXED deterministic policy Pi(s1) -- our attacker perturbs both channels
    independently, so the action must be an explicit input, not implicit).
"""

import numpy as np
import torch
import torch.nn as nn


class CVAE(nn.Module):
    """A single conditional VAE: encoder q(z|s2,cond), decoder p(s2|z,cond)."""

    def __init__(self, obs_dim, cond_dim, latent_dim=16, hidden_sizes=(200, 200)):
        super().__init__()
        self.obs_dim = obs_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        enc_layers = []
        in_f = obs_dim + cond_dim
        for h in hidden_sizes:
            enc_layers += [nn.Linear(in_f, h), nn.SiLU()]
            in_f = h
        self.encoder = nn.Sequential(*enc_layers)
        self.enc_mu = nn.Linear(in_f, latent_dim)
        self.enc_logvar = nn.Linear(in_f, latent_dim)

        dec_layers = []
        in_f = latent_dim + cond_dim
        for h in hidden_sizes:
            dec_layers += [nn.Linear(in_f, h), nn.SiLU()]
            in_f = h
        dec_layers += [nn.Linear(in_f, obs_dim)]
        self.decoder = nn.Sequential(*dec_layers)

        self.cond_mu = nn.Parameter(torch.zeros(1, cond_dim), requires_grad=False)
        self.cond_sigma = nn.Parameter(torch.ones(1, cond_dim), requires_grad=False)
        self.obs_mu = nn.Parameter(torch.zeros(1, obs_dim), requires_grad=False)
        self.obs_sigma = nn.Parameter(torch.ones(1, obs_dim), requires_grad=False)

    def set_norm_stats(self, cond_mu, cond_sigma, obs_mu, obs_sigma):
        self.cond_mu.data = cond_mu
        self.cond_sigma.data = cond_sigma
        self.obs_mu.data = obs_mu
        self.obs_sigma.data = obs_sigma

    def encode(self, s2, cond):
        s2n = (s2 - self.obs_mu) / self.obs_sigma
        condn = (cond - self.cond_mu) / self.cond_sigma
        h = self.encoder(torch.cat([s2n, condn], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, cond):
        condn = (cond - self.cond_mu) / self.cond_sigma
        recon_n = self.decoder(torch.cat([z, condn], dim=-1))
        return recon_n * self.obs_sigma + self.obs_mu

    def forward(self, s2, cond):
        mu, logvar = self.encode(s2, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

    def loss(self, s2, cond, beta=1.0):
        recon, mu, logvar = self.forward(s2, cond)
        recon_loss = ((recon - s2) ** 2).mean(-1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return (recon_loss + beta * kl).mean(), recon_loss.mean().item()

    @torch.no_grad()
    def reconstruction_error(self, s2, cond):
        """MSE using the ENCODER's own posterior mean (deterministic, no sampling noise) --
        this is the score used at eval time, matching the paper's "reconstruction error"
        framing without adding sampling variance to the anomaly signal itself."""
        mu, _ = self.encode(s2, cond)
        recon = self.decode(mu, cond)
        return ((recon - s2) ** 2).mean(-1)


class COTDDetector:
    """Ensemble-of-CVAE OOD detector with split-conformal threshold calibration.

    Mirrors `harl.detectors.pedm_detector.PEDMDetector`'s public interface (`fit`,
    `predict_scores`, `calibrate_threshold`, `detect`, `save`, `load`) so it is a drop-in
    alternative detector in the existing eval scripts.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        ens_size=5,
        latent_dim=16,
        hidden_sizes=(200, 200),
        beta=1.0,
        lr=1e-3,
        device="cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ens_size = ens_size
        self.latent_dim = latent_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.beta = beta
        self.device = torch.device(device)
        cond_dim = obs_dim + action_dim
        self.members = nn.ModuleList(
            [CVAE(obs_dim, cond_dim, latent_dim, hidden_sizes) for _ in range(ens_size)]
        ).to(self.device)
        self.optims = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.members]
        self.threshold = None

    def fit(
        self,
        obs,
        actions,
        next_obs,
        n_train_epochs=200,
        batch_size=512,
        verbose=False,
    ):
        """Train the CVAE ensemble on clean transitions.

        Each member trains on an independent bootstrap resample of the transition set
        (matching PEDM's own ensemble-diversity convention, `ProbEnsemble.fit`'s `idxs`).
        """
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        cond = np.concatenate([obs, actions], axis=-1)

        cond_t = torch.from_numpy(cond).to(self.device)
        s2_t = torch.from_numpy(next_obs).to(self.device)
        cond_mu = cond_t.mean(0, keepdim=True)
        cond_sigma = cond_t.std(0, keepdim=True).clamp_min(1e-6)
        obs_mu = s2_t.mean(0, keepdim=True)
        obs_sigma = s2_t.std(0, keepdim=True).clamp_min(1e-6)
        for m in self.members:
            m.set_norm_stats(cond_mu, cond_sigma, obs_mu, obs_sigma)

        n = len(cond_t)
        last_losses = [np.inf] * self.ens_size
        for ep in range(n_train_epochs):
            for mi, (member, optim) in enumerate(zip(self.members, self.optims)):
                idx = np.random.randint(0, n, size=n)
                acc, n_batches = 0.0, 0
                for start in range(0, n, batch_size):
                    b = idx[start : start + batch_size]
                    loss, recon_mse = member.loss(s2_t[b], cond_t[b], beta=self.beta)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    acc += recon_mse
                    n_batches += 1
                last_losses[mi] = acc / max(1, n_batches)
            if verbose and (ep % max(1, n_train_epochs // 10) == 0):
                print(f"[COTD] epoch {ep}/{n_train_epochs} recon_mse(per member)={last_losses}")
        return float(np.mean(last_losses))

    @torch.no_grad()
    def predict_scores(self, obs, acts) -> np.ndarray:
        """Ensemble-mean reconstruction-error anomaly score for each transition.

        Args:
            obs: (seq_len, obs_dim) sequential observations
            acts: (seq_len - 1, action_dim) sequential actions
        Returns:
            scores: (seq_len - 1,) anomaly score per transition
        """
        obs = np.asarray(obs, dtype=np.float32)
        acts = np.asarray(acts, dtype=np.float32)
        s1 = obs[:-1]
        s2 = obs[1:]
        cond = np.concatenate([s1, acts], axis=-1)
        cond_t = torch.from_numpy(cond).to(self.device)
        s2_t = torch.from_numpy(s2).to(self.device)
        per_member = torch.stack(
            [m.reconstruction_error(s2_t, cond_t) for m in self.members], dim=0
        )  # (ens_size, seq_len-1)
        return per_member.mean(0).cpu().numpy()

    def calibrate_threshold(self, clean_scores, quantile=0.97):
        """Split-conformal threshold from pooled held-out clean per-step scores.

        Uses the standard finite-sample conformal quantile correction (the ceil((1-delta)*
        (n+1))-th order statistic, not a plain np.quantile) so the threshold carries the same
        marginal coverage guarantee the paper's own calibration relies on.
        """
        pooled = np.sort(np.concatenate([np.ravel(s) for s in clean_scores]))
        n = len(pooled)
        idx = int(np.ceil(quantile * (n + 1))) - 1
        idx = min(max(idx, 0), n - 1)
        self.threshold = float(pooled[idx])
        return self.threshold

    def detect(self, obs, acts):
        scores = self.predict_scores(obs, acts)
        if self.threshold is None:
            raise RuntimeError("threshold not calibrated; call calibrate_threshold")
        return scores > self.threshold

    def save(self, path):
        torch.save(
            {
                "state_dict": self.members.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "ens_size": self.ens_size,
                "latent_dim": self.latent_dim,
                "hidden_sizes": self.hidden_sizes,
                "beta": self.beta,
                "threshold": self.threshold,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if tuple(ckpt["hidden_sizes"]) != self.hidden_sizes or ckpt["latent_dim"] != self.latent_dim \
                or ckpt["ens_size"] != self.ens_size:
            cond_dim = self.obs_dim + self.action_dim
            self.members = nn.ModuleList(
                [
                    CVAE(self.obs_dim, cond_dim, ckpt["latent_dim"], tuple(ckpt["hidden_sizes"]))
                    for _ in range(ckpt["ens_size"])
                ]
            ).to(self.device)
            self.ens_size = ckpt["ens_size"]
            self.latent_dim = ckpt["latent_dim"]
            self.hidden_sizes = tuple(ckpt["hidden_sizes"])
        self.members.load_state_dict(ckpt["state_dict"])
        self.threshold = ckpt.get("threshold")
