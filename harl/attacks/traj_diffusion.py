"""Conditional trajectory-diffusion model: learns the distribution of plausible N-step
CLEAN future (observation, victim-action) trajectories given an H-step history, trained
ONLY on clean (no-attacker) victim rollouts (see examples/collect_clean_trajectories.py +
examples/train_traj_diffusion.py). Frozen after training, it can score any candidate
future trajectory's long-horizon plausibility via denoise_cost -- intended as a future
MARL attacker's early-warning deception-cost signal (validated in isolation by
examples/eval_traj_diffusion.py and examples/eval_traj_diffusion_discrimination.py; NOT
wired into any MARL training loop by this module).

x0/cond are assumed already scale-normalized by the caller -- this module does no obs/act
scaling itself.
"""
import math

import torch
import torch.nn as nn

from harl.attacks.adversary_ac import orthogonal_init


def sinusoidal_embedding(k, dim, max_period=10000):
    """k: (B,) long tensor of diffusion timesteps. Returns (B, dim) embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=k.device).float() / half)
    args = k.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TrajDiffusionNet(nn.Module):
    """eps_theta(x_k, k, cond) -> predicted noise, same shape as x_k."""

    def __init__(self, x_dim, cond_dim, hidden_size=512, time_dim=128, n_hidden=4):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            orthogonal_init(nn.Linear(time_dim, time_dim)), nn.SiLU(),
            orthogonal_init(nn.Linear(time_dim, time_dim)),
        )
        in_dim = x_dim + cond_dim + time_dim
        layers = [orthogonal_init(nn.Linear(in_dim, hidden_size)), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [orthogonal_init(nn.Linear(hidden_size, hidden_size)), nn.SiLU()]
        layers += [orthogonal_init(nn.Linear(hidden_size, x_dim), gain=0.01)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_k, k, cond):
        t_emb = self.time_mlp(sinusoidal_embedding(k, self.time_dim))
        h = torch.cat([x_k, cond, t_emb], dim=-1)
        return self.net(h)


class CondTrajDiffusion:
    """Wraps TrajDiffusionNet + a linear beta schedule (K=100 steps by default -- far fewer
    than the textbook 1000 since this is an MLP over a modest ~(H+N)*(obs_dim+act_dim)-dim
    vector, not images; a deliberate speed/quality tradeoff)."""

    def __init__(self, x_dim, cond_dim, K=100, beta_start=1e-4, beta_end=0.02,
                 hidden_size=512, device="cpu"):
        self.x_dim, self.cond_dim, self.K = x_dim, cond_dim, K
        self.device = device
        self.net = TrajDiffusionNet(x_dim, cond_dim, hidden_size=hidden_size).to(device)
        betas = torch.linspace(beta_start, beta_end, K, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas, self.alphas, self.alpha_bars = betas, alphas, alpha_bars

    def parameters(self):
        return self.net.parameters()

    def loss(self, x0, cond):
        """x0: (B, x_dim), cond: (B, cond_dim). Standard DDPM noise-prediction loss."""
        B = x0.shape[0]
        k = torch.randint(0, self.K, (B,), device=self.device)
        noise = torch.randn_like(x0)
        ab = self.alpha_bars[k][:, None]
        x_k = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred = self.net(x_k, k, cond)
        return ((pred - noise) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond, n_samples=1):
        """cond: (cond_dim,) or (n_samples, cond_dim). Returns (n_samples, x_dim)."""
        if cond.dim() == 1:
            cond = cond[None].expand(n_samples, -1)
        x = torch.randn(n_samples, self.x_dim, device=self.device)
        for k in reversed(range(self.K)):
            kk = torch.full((n_samples,), k, device=self.device, dtype=torch.long)
            eps = self.net(x, kk, cond)
            alpha, ab, beta = self.alphas[k], self.alpha_bars[k], self.betas[k]
            mean = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
            x = mean + beta.sqrt() * torch.randn_like(x) if k > 0 else mean
        return x

    @torch.no_grad()
    def denoise_cost(self, x_candidate, cond, k_probe=(10, 30, 50, 70, 90), n_repeats=3):
        """Module-5 cost: average noise-prediction MSE at several fixed noise levels,
        averaged over n_repeats independent noise draws per level (reduces Monte-Carlo
        variance). x_candidate: (B, x_dim), cond: (B, cond_dim). Returns (B,) cost per row."""
        costs = []
        for k in k_probe:
            k = min(k, self.K - 1)
            kk = torch.full((x_candidate.shape[0],), k, device=self.device, dtype=torch.long)
            ab = self.alpha_bars[k]
            for _ in range(n_repeats):
                noise = torch.randn_like(x_candidate)
                x_k = ab.sqrt() * x_candidate + (1 - ab).sqrt() * noise
                pred = self.net(x_k, kk, cond)
                costs.append(((pred - noise) ** 2).mean(dim=-1))
        return torch.stack(costs, dim=0).mean(dim=0)

    def save(self, path):
        torch.save({"state_dict": self.net.state_dict(), "x_dim": self.x_dim,
                    "cond_dim": self.cond_dim, "K": self.K,
                    "beta_start": float(self.betas[0]), "beta_end": float(self.betas[-1])}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        model = cls(ck["x_dim"], ck["cond_dim"], K=ck["K"], beta_start=ck["beta_start"],
                     beta_end=ck["beta_end"], device=device)
        model.net.load_state_dict(ck["state_dict"])
        model.net.eval()
        return model
