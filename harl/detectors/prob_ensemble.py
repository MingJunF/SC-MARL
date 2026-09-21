"""Probabilistic ensemble network.

Self-contained PyTorch port of the probabilistic ensemble used by the
PEDM-OOD detector (Haider et al.), adapted for HARL. The ensemble outputs a
Gaussian (mean, variance) over the target for every member network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm


def truncated_normal(size, std, mean=0.0):
    """Sample from a truncated normal (values beyond 2 std are re-drawn)."""
    return truncnorm.rvs(-2, 2, loc=mean, scale=std, size=size)


def get_affine_params(ens_size, in_features, out_features):
    """Create per-ensemble weight and bias parameters."""
    w = truncated_normal(
        size=(ens_size, in_features, out_features),
        std=1.0 / (2.0 * np.sqrt(in_features)),
    )
    w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
    b = nn.Parameter(torch.zeros(ens_size, 1, out_features))
    return w, b


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class LinLayer(nn.Module):
    """Batched linear layer, one weight matrix per ensemble member."""

    def __init__(self, ens_size, in_f, out_f, activation="swish"):
        super().__init__()
        self.lin_w, self.lin_b = get_affine_params(ens_size, in_f, out_f)
        if activation:
            self.activation = nn.ModuleDict(
                [
                    ["lrelu", nn.LeakyReLU()],
                    ["relu", nn.ReLU()],
                    ["swish", nn.SiLU()],
                ]
            )[activation]
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = x.matmul(self.lin_w) + self.lin_b
        x = self.activation(x)
        return x


class ProbEnsemble(nn.Module):
    """Probabilistic ensemble of MLPs predicting mean and variance."""

    def __init__(
        self,
        ens_size,
        layer_sizes,
        decays=None,
        normalize_data=True,
        activation_fn="swish",
        lr=0.001,
        device="cpu",
    ):
        super().__init__()

        self.ens_size = ens_size
        self.in_features = layer_sizes[0]
        # doubled because we output both mean and variance
        self.out_features = layer_sizes[-1] * 2
        self.decays = decays
        self.activation_fn = activation_fn

        self.fc_layers = nn.Sequential(
            *[
                LinLayer(ens_size, in_f, out_f, activation=self.activation_fn)
                for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:-1])
            ]
        )
        self.fc_layers.add_module(
            "out_layer",
            LinLayer(ens_size, layer_sizes[-2], self.out_features, activation=None),
        )

        self.inputs_mu = nn.Parameter(
            torch.zeros(1, self.in_features), requires_grad=False
        )
        self.inputs_sigma = nn.Parameter(
            torch.ones(1, self.in_features), requires_grad=False
        )

        self.max_logvar = nn.Parameter(
            torch.ones(1, self.out_features // 2, dtype=torch.float32) / 2.0
        )
        self.min_logvar = nn.Parameter(
            -torch.ones(1, self.out_features // 2, dtype=torch.float32) * 10.0
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device(device)
        self.to(self.device)
        self.normalize_data = normalize_data

        self.val_se_loss = None

    def compute_decays(self):
        dec = []
        for lin_dec, layer in zip(self.decays, self.fc_layers):
            dec.append(lin_dec * (layer.lin_w**2).sum() / 2.0)
        return sum(dec)

    def norm_data(self, X_train):
        mu = torch.mean(X_train, dim=0, keepdims=True)
        sigma = torch.std(X_train, dim=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        self.inputs_mu.data = mu.to(self.device).float()
        self.inputs_sigma.data = sigma.to(self.device).float()

    def forward(self, inputs, ret_logvar=False):
        if self.normalize_data:
            inputs = (inputs - self.inputs_mu) / self.inputs_sigma
        inputs = self.fc_layers(inputs)
        mean = inputs[:, :, : (self.out_features // 2)]
        logvar = inputs[:, :, (self.out_features // 2) :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if ret_logvar:
            return mean, logvar
        return mean, torch.exp(logvar)

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_train_epochs,
        batch_size=512,
        verbose=False,
    ):
        X_train = torch.from_numpy(X_train).to(self.device).float()
        y_train = torch.from_numpy(y_train).to(self.device).float()
        X_val = torch.from_numpy(X_val).to(self.device).float()
        y_val = torch.from_numpy(y_val).to(self.device).float()

        if self.normalize_data:
            self.norm_data(X_train)

        idxs = np.random.randint(len(X_train), size=[self.ens_size, len(X_train)])
        ep_train_loss = np.inf
        for ep in range(n_train_epochs):
            ep_train_loss = self.train_epoch(
                X_train, y_train, idxs=idxs, batch_size=batch_size
            )
            idxs = shuffle_rows(idxs)
            if verbose and (ep % max(1, n_train_epochs // 10) == 0):
                ep_val_loss = (
                    self.val_epoch(X_val, y_val).mean() if len(X_val) > 0 else np.nan
                )
                print(
                    f"[PEDM] epoch {ep}/{n_train_epochs} "
                    f"train_loss={ep_train_loss:.5f} val_loss={ep_val_loss}"
                )
        if len(X_val) > 0:
            self._val_threshold(X_val=X_val, y_val=y_val)
        return ep_train_loss

    def train_epoch(self, X_train, y_train, idxs, batch_size):
        self.train()
        acc_loss = []
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        for batch_num in range(num_batch):
            batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

            loss = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())

            if self.decays is not None:
                loss += self.compute_decays()

            input_mb = X_train[batch_idxs, :]
            target_mb = y_train[batch_idxs, :]

            mean, logvar = self.forward(input_mb, ret_logvar=True)
            inv_var = torch.exp(-logvar)

            train_losses = ((mean - target_mb) ** 2) * inv_var + logvar
            train_losses = train_losses.mean(-1).mean(-1).sum()

            loss += train_losses

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            acc_loss.append(loss.item())
        return float(np.mean(acc_loss))

    @torch.no_grad()
    def val_epoch(self, X_val, y_val, return_se=False):
        self.eval()
        idxs = np.random.randint(len(X_val), size=[self.ens_size, len(X_val)])
        val_in = X_val[idxs, :]
        val_targ = y_val[idxs, :]
        mean, _ = self.forward(val_in)
        se_loss = (mean - val_targ) ** 2
        mse_loss = se_loss.mean(-1).mean(-1)
        if not return_se:
            return mse_loss.cpu().numpy()
        return mse_loss.cpu().numpy(), se_loss.cpu().numpy()

    def _val_threshold(self, X_val, y_val):
        _, se_loss = self.val_epoch(X_val=X_val, y_val=y_val, return_se=True)
        self.val_se_loss = se_loss.mean(-1).min(0)

    def _expand(self, mat, n_part):
        dim = mat.shape[-1]
        reshaped = mat.reshape(-1, self.ens_size, n_part // self.ens_size, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.reshape(self.ens_size, -1, dim)
        return reshaped

    def _flatten(self, arr, n_part):
        dim = arr.shape[-1]
        reshaped = arr.reshape(self.ens_size, -1, n_part // self.ens_size, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.reshape(-1, dim)
        return reshaped

    def _unflatten(self, arr, n_part):
        dim = arr.shape[-1]
        n_opt = arr.shape[0] // n_part
        reshaped = arr.reshape(n_opt, self.ens_size, -1, dim)
        return reshaped
