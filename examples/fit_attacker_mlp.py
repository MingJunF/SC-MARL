"""Attacker world model that is ARCHITECTURALLY INDEPENDENT of the PEDM detector.

Point raised: using a PEDM probabilistic ensemble as q_clean shares the detector's
exact inductive bias, so "the forge free-roll lands in PEDM support" is partly
circular. This fits a PLAIN deterministic MLP instead (single net, ReLU, MSE, no
ensemble, no variance head) as the deceiver's continuation model replacing truedyn.
If the closed-form deceiver still reaches the B=40 basin with THIS unrelated network,
the basin is a property of on-clean-manifold free-roll, not of copying the detector.

Saves `attacker_mlp.pt`. The MLPDyn class is imported by eval_deceiver_learnedwm.py.

    python examples/fit_attacker_mlp.py --scenario HalfCheetah-v4
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import Box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.eval_robust_detector import Victim
from examples.diagnose_detectors import rollout
from examples.train_obs_attacker import AntTrueDynamics, estimate_scales, resolve_victim_run


class MLPDyn(nn.Module):
    """Plain deterministic residual MLP: predicts (o_{t+1}-o_t) from [o_t,a_t].

    Deliberately NOTHING like PEDM: single network, ReLU, no probabilistic head, no
    ensemble, MSE loss, input/output z-scored. `.next_obs(o,a)` matches AntTrueDynamics.
    """
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim + act_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, obs_dim)]
        self.net = nn.Sequential(*layers)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        # input/target normalization buffers (filled at fit time)
        self.register_buffer("x_mu", torch.zeros(obs_dim + act_dim))
        self.register_buffer("x_sd", torch.ones(obs_dim + act_dim))
        self.register_buffer("y_mu", torch.zeros(obs_dim))
        self.register_buffer("y_sd", torch.ones(obs_dim))

    def forward(self, x):  # x = [o,a] raw
        xn = (x - self.x_mu) / self.x_sd
        return self.net(xn) * self.y_sd + self.y_mu  # de-normalized residual

    @torch.no_grad()
    def next_obs(self, obs, action):
        o = torch.as_tensor(np.asarray(obs, np.float32))
        a = torch.as_tensor(np.asarray(action, np.float32))
        x = torch.cat([o, a])[None]
        d = self.forward(x)[0].cpu().numpy()
        return (np.asarray(obs, np.float32) + d).astype(np.float32)

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(ck["obs_dim"], ck["act_dim"]).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        return m


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", default="HalfCheetah-v4")
    p.add_argument("--victim_glob", default="")
    p.add_argument("--out", default="")
    p.add_argument("--n_fit", type=int, default=40)
    p.add_argument("--episode_length", type=int, default=300)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=321)
    args = p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    default_glob = {"Ant-v4": f"results/robust_victim/{args.scenario}/mappo/*/seed-00001-*",
                    "HalfCheetah-v4": f"results/robust_victim/{args.scenario}/mappo/victim6m/seed-00001-*"}
    victim_glob = args.victim_glob or default_glob.get(
        args.scenario, f"results/robust_victim/{args.scenario}/mappo/*/seed-00001-*")
    out = args.out or f"results/obs_attackers/{args.scenario}/attacker_mlp.pt"
    victim_run = resolve_victim_run(victim_glob)
    print(f"victim_run: {victim_run}")

    probe = gym.make(args.scenario)
    obs_dim, act_dim = probe.observation_space.shape[0], probe.action_space.shape[0]
    act_space = Box(probe.action_space.low, probe.action_space.high, probe.action_space.shape, np.float32)
    lo, hi = act_space.low.astype(np.float32), act_space.high.astype(np.float32)
    probe.close()

    victim = Victim(victim_run, obs_dim, act_space, args.device)
    dyn = AntTrueDynamics(args.scenario)
    obs_scale, act_scale = estimate_scales(args.scenario, victim, lo, hi)
    scale, ascale = obs_scale[None], act_scale[None]

    print(f"collecting {args.n_fit} clean episodes (len {args.episode_length})...")
    cf = rollout(None, "obs", "ppo", 0.0, victim, dyn, args.scenario, scale, ascale,
                 lo, hi, args.n_fit, args.episode_length, args.device, 7000)
    fo = np.concatenate([e["o"][:-1] for e in cf]).astype(np.float32)
    fa = np.concatenate([e["a"] for e in cf]).astype(np.float32)
    fn = np.concatenate([e["o"][1:] for e in cf]).astype(np.float32)
    X = np.concatenate([fo, fa], 1); Y = fn - fo

    model = MLPDyn(obs_dim, act_dim).to(args.device)
    # fill normalization from clean data
    model.x_mu.copy_(torch.as_tensor(X.mean(0))); model.x_sd.copy_(torch.as_tensor(X.std(0) + 1e-6))
    model.y_mu.copy_(torch.as_tensor(Y.mean(0))); model.y_sd.copy_(torch.as_tensor(Y.std(0) + 1e-6))
    Xt = torch.as_tensor(X, device=args.device); Yt = torch.as_tensor(Y, device=args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(Xt)
    print(f"fitting plain MLP dyn on {n} transitions ({args.epochs} epochs)...")
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=args.device)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred = model(Xt[idx])
            loss = ((pred - Yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep % 40 == 0 or ep == args.epochs - 1:
            print(f"[MLP] epoch {ep}/{args.epochs} mse={tot / n:.5f}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)

    with torch.no_grad():
        pred_res = model(Xt).cpu().numpy()
    pred_next = fo + pred_res
    per = np.linalg.norm((pred_next - fn) / scale, axis=1)
    print(f"saved plain-MLP WM -> {out}")
    print(f"  one-step mean forward error (normalized L2): mean {per.mean():.4f}  p90 {np.quantile(per, 0.9):.4f}")


if __name__ == "__main__":
    main()
