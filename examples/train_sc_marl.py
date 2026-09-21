"""SC-MARL: Stealth-Constrained joint MARL attacker with trajectory-level stealth estimation
and role-conditioned dual credit assignment (2026-09-16, per explicit user design).

Two learned attacker policies against a FROZEN victim (victim/environment/detector never
update):

  * Hidden Attacker   pi^H: produces an observation mask/perturbation BEFORE the victim acts.
  * Performance Attacker pi^P: produces an action perturbation AFTER seeing the reported
    observation and the victim's chosen action.

Per-step causal order (matches the threat model exactly):

    z^H_t (true state)  ->  pi^H  ->  o_rep_t = s_t + phi^H_t
    o_rep_t  ->  victim  ->  a_v_t
    z^P_t = [o_rep_t, a_v_t]  ->  pi^P  ->  a_exec_t = a_v_t + phi^P_t
    environment.step(a_exec_t)  ->  s_{t+1}, r_v_t

Two trajectories are recorded: the PHYSICAL trajectory tau = (s_t, a_exec_t, r_v_t) and the
REPORTED trajectory zeta = (o_rep_t, a_v_t) -- the latter is what a detector/observer would
see (the victim's COMMANDED action, not the possibly Performance-tampered executed one, per
the user's explicit spec: "victim 和 detector 实际接收到的 reported observations 与 victim actions").

Trajectory-level stealth (NOT a per-step detector penalty): a GRU-encoder discriminator
`TrajectoryCritic` T_psi is trained each update to separate CLEAN reference episodes (fixed
victim, no attack, collected once up front) from this update's ATTACKED reported episodes
(GAIL/f-GAN style binary classifier). Its mean logit on attacked episodes is the KL-divergence
estimate C_hat_KL; its per-episode NEGATIVE logit is the stealth utility w_psi(zeta) (higher =
more clean-like). w_psi is paid ONLY as a terminal reward on the step where that episode
completes -- intermediate steps get stealth reward 0 -- then propagated backward through the
whole episode via GAE, exactly like the user's spec ("中间时间步 stealth reward 为零，episode 结束
时给予 wψ(ζ)"). A SINGLE shared dual-ascent lambda (not two) targets a stealth budget on
C_hat_KL, since both attackers jointly induce the same reported-trajectory distribution.

Role-conditioned dual credit: BOTH attacker networks are `AdversaryActorCriticViability`
(actor + TWO critic heads) reused unchanged from the existing viability/Stackelberg lines --
`critic` = effectiveness value V_eff (bootstraps -r_v_t), `critic_fake` = stealth value V_st
(bootstraps the terminal w_psi reward), each conditioned on that role's OWN decision-stage
input (z^H_t or z^P_t). Role-specific joint advantage: A_lambda = A_eff + lambda*A_st for BOTH
roles (same lambda, different A_eff/A_st streams since each role's critics see different
inputs and get gradient from its own actions).

Ordered actor update: Hidden Attacker is updated FIRST; the (old vs new) Hidden policy's
importance ratio on the SAME rollout observations is then folded into the Performance
Attacker's PPO surrogate as an extra multiplicative upstream-correction factor, so Performance's
update accounts for the just-shifted Hidden policy rather than the stale rollout-time one. This
is a practical PPO-level approximation (not a global-optimality guarantee), per the user's own
explicit caveat.

Known simplification vs the user's full spec (documented, not hidden): episode boundaries
within a fixed T-step rollout window are tracked per-env, and each COMPLETED episode's
(o_rep, a_v) sequence is padded/truncated to `--traj_max_len` for GRU batching; an episode
still in progress at the window boundary is DISCARDED for trajectory-critic purposes (not
carried across updates) to avoid cross-window state-carrying complexity -- its physical
(effectiveness) reward is unaffected, only its stealth-credit contribution is lost for that
partial episode.

Usage:
  python examples/train_sc_marl.py --scenario Ant-v4 --budget_obs 4 --budget_act 4 \
      --num_env_steps 2000000
"""
import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.spaces import Box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.attacks.adversary_ac_viability import AdversaryActorCriticViability
from examples.train_obs_attacker import l2_project, victim_act_batch, estimate_scales, resolve_victim_run
from examples.train_marl_attacker import gae
from examples.eval_robust_detector import Victim


class TrajectoryCritic(nn.Module):
    """GRU encoder over a reported-trajectory (o_rep/scale, a_v) sequence -> discriminator
    logit. Trained so CLEAN episodes get a LOW logit and ATTACKED episodes get a HIGH logit
    (GAIL/f-GAN style binary discriminator) -- the attacked-distribution mean logit is then a
    variational estimate of KL(attacked || clean), and the per-episode negative logit is the
    "how clean-like is this specific episode" stealth utility."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.gru = nn.GRU(obs_dim + act_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, seq, lengths):
        # seq: (B, L, obs_dim+act_dim) zero-padded; lengths: (B,) true episode lengths
        packed = nn.utils.rnn.pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1]).squeeze(-1)  # (B,) logit


def update_trajectory_critic(traj_critic, opt, clean_seq, clean_len, atk_seq, atk_len, epochs=4):
    """One outer-iteration update of T_psi; returns (C_hat_KL, w_psi per attacked episode)."""
    y = torch.cat([
        torch.zeros(clean_seq.shape[0], device=clean_seq.device),
        torch.ones(atk_seq.shape[0], device=atk_seq.device),
    ])
    seq = torch.cat([clean_seq, atk_seq], 0)
    length = torch.cat([clean_len, atk_len], 0)
    for _ in range(epochs):
        logits = traj_critic(seq, length)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        atk_logits = traj_critic(atk_seq, atk_len)
        ckl = float(atk_logits.mean().item())
        w_psi = (-atk_logits).cpu().numpy()
    return ckl, w_psi


def ppo_update_dual(net, optim, buf, args, in_dim, out_dim, device, upstream_ratio=None):
    """PPO update for one role's dual-critic net. `upstream_ratio` (optional, per-sample
    tensor) is the ordered-update cross-role correction: Performance Attacker's surrogate is
    additionally weighted by the (new/old) importance ratio of the JUST-UPDATED Hidden policy
    on the same rollout, per the ordered causal-chain update rule."""
    b_obs = torch.as_tensor(buf["obs"].reshape(-1, in_dim), device=device)
    b_act = torch.as_tensor(buf["act"].reshape(-1, out_dim), device=device)
    b_logp = torch.as_tensor(buf["logp"].reshape(-1), device=device)
    b_adv = torch.as_tensor(buf["adv"].reshape(-1), device=device)
    b_ret_eff = torch.as_tensor(buf["ret_eff"].reshape(-1), device=device)
    b_ret_st = torch.as_tensor(buf["ret_st"].reshape(-1), device=device)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
    up = upstream_ratio.detach() if upstream_ratio is not None else torch.ones_like(b_logp)
    bs = b_obs.shape[0]
    mb = max(1, bs // args.num_mini_batch)
    pl, vle, vls, en = [], [], [], []
    for _ in range(args.ppo_epoch):
        perm = torch.randperm(bs, device=device)
        for s in range(0, bs, mb):
            idx = perm[s:s + mb]
            nlp, ent, v_eff, v_st = net.evaluate_actions_v(b_obs[idx], b_act[idx])
            ratio = torch.exp(nlp - b_logp[idx]) * up[idx]
            s1 = ratio * b_adv[idx]
            s2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * b_adv[idx]
            ploss = -torch.min(s1, s2).mean()
            vloss_eff = 0.5 * (v_eff - b_ret_eff[idx]).pow(2).mean()
            vloss_st = 0.5 * (v_st - b_ret_st[idx]).pow(2).mean()
            loss = ploss + args.vf_coef * (vloss_eff + vloss_st) - args.ent_coef * ent.mean()
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optim.step()
            pl.append(ploss.item()); vle.append(vloss_eff.item()); vls.append(vloss_st.item()); en.append(ent.mean().item())
    return np.mean(pl), np.mean(vle), np.mean(vls), np.mean(en)


@torch.no_grad()
def collect_clean_reference(victim, scenario, obs_dim, act_dim, scale, lo, hi, n_episodes, ep_len, device, seed=9000):
    """One-off pool of clean (no-attack) reported episodes for the trajectory critic -- 'reported'
    here just equals the true (obs, victim_action) since there's no attack."""
    seqs = np.zeros((n_episodes, ep_len, obs_dim + act_dim), np.float32)
    lens = np.zeros(n_episodes, np.int64)
    for i in range(n_episodes):
        env = gym.make(scenario)
        o, _ = env.reset(seed=seed + i)
        for t in range(ep_len):
            a_v = np.clip(victim_act_batch(victim, o[None])[0], lo, hi)
            seqs[i, t] = np.concatenate([o / scale, a_v])
            no, r, term, trunc, _ = env.step(a_v)
            o = no
            if term or trunc:
                lens[i] = t + 1
                break
        else:
            lens[i] = ep_len
        env.close()
    return torch.as_tensor(seqs, device=device), torch.as_tensor(lens, device=device)


def train(args, victim, obs_scale, act_scale, device):
    scenario = args.scenario
    n_envs, T = args.num_envs, args.num_steps
    obs_dim = obs_scale.shape[0]
    scale = obs_scale[None, :]
    envs = [gym.make(scenario) for _ in range(n_envs)]
    act_dim = envs[0].action_space.shape[0]
    lo = envs[0].action_space.low.astype(np.float32)
    hi = envs[0].action_space.high.astype(np.float32)
    ascale = act_scale[None, :]

    h_in = obs_dim                    # Hidden sees the true state z^H_t
    p_in = obs_dim + act_dim          # Performance sees z^P_t = [o_rep, a_v]
    h_net = AdversaryActorCriticViability(h_in, obs_dim, hidden_size=256, activation="tanh").to(device)
    p_net = AdversaryActorCriticViability(p_in, act_dim, hidden_size=256, activation="tanh").to(device)
    h_opt = torch.optim.Adam(h_net.parameters(), lr=args.lr, eps=1e-5)
    p_opt = torch.optim.Adam(p_net.parameters(), lr=args.lr, eps=1e-5)

    traj_critic = TrajectoryCritic(obs_dim, act_dim, hidden=args.traj_hidden).to(device)
    traj_opt = torch.optim.Adam(traj_critic.parameters(), lr=args.traj_lr)

    print(f"[sc-marl] collecting {args.n_clean_ref} clean reference episodes...")
    clean_seq, clean_len = collect_clean_reference(
        victim, scenario, obs_dim, act_dim, obs_scale, lo, hi,
        args.n_clean_ref, args.traj_max_len, device)

    lam = args.lambda_init

    obs = np.zeros((n_envs, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=args.seed + i); obs[i] = o
    last_done = np.zeros(n_envs, np.float32)
    # Per-env in-progress reported-episode sequence (for the trajectory critic); reset whenever
    # that env's episode ends. Discarded (not carried) if still in-progress at window boundary.
    ep_seq = [[] for _ in range(n_envs)]

    num_updates = int(args.num_env_steps // (T * n_envs))
    ret_hist, ckl_hist = deque(maxlen=100), deque(maxlen=100)
    start = time.time()
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)

    def save_ckpt():
        torch.save({
            "h_state_dict": h_net.state_dict(), "p_state_dict": p_net.state_dict(),
            "traj_critic_state_dict": traj_critic.state_dict(),
            "h_in": h_in, "p_in": p_in, "obs_dim": obs_dim, "act_dim": act_dim,
            "budget_obs": args.budget_obs, "budget_act": args.budget_act,
            "obs_scale": obs_scale, "act_scale": act_scale, "lambda": float(lam),
            "scenario": scenario, "algo": "sc_marl",
        }, ckpt_path)

    for update in range(1, num_updates + 1):
        bh = {"obs": np.zeros((T, n_envs, h_in), np.float32), "act": np.zeros((T, n_envs, obs_dim), np.float32),
              "logp": np.zeros((T, n_envs), np.float32),
              "val_eff": np.zeros((T, n_envs), np.float32), "val_st": np.zeros((T, n_envs), np.float32)}
        bp = {"obs": np.zeros((T, n_envs, p_in), np.float32), "act": np.zeros((T, n_envs, act_dim), np.float32),
              "logp": np.zeros((T, n_envs), np.float32),
              "val_eff": np.zeros((T, n_envs), np.float32), "val_st": np.zeros((T, n_envs), np.float32)}
        rew_eff = np.zeros((T, n_envs), np.float32)   # shared physical-effectiveness reward
        rew_st = np.zeros((T, n_envs), np.float32)    # terminal-only stealth reward
        done_buf = np.zeros((T, n_envs), np.float32)
        ep_ret = np.zeros(n_envs, np.float32)
        returns_log = []
        atk_seqs_this_update, atk_lens_this_update = [], []

        for t in range(T):
            # ---- Hidden Attacker: z^H_t = true state ----
            h_adv_in = (obs / scale).astype(np.float32)
            with torch.no_grad():
                dh, h_logp, h_veff, h_vst = h_net.act_v(torch.as_tensor(h_adv_in, device=device))
            phi_h = l2_project(dh.cpu().numpy(), args.budget_obs)
            o_rep = obs + phi_h * scale

            a_v = np.clip(victim_act_batch(victim, o_rep), lo, hi)

            # ---- Performance Attacker: z^P_t = [o_rep, a_v] ----
            p_adv_in = np.concatenate([o_rep / scale, a_v], axis=1).astype(np.float32)
            with torch.no_grad():
                dp, p_logp, p_veff, p_vst = p_net.act_v(torch.as_tensor(p_adv_in, device=device))
            phi_p = l2_project(dp.cpu().numpy(), args.budget_act)
            a_exec = np.clip(a_v + phi_p * ascale, lo, hi)

            for i in range(n_envs):
                ep_seq[i].append(np.concatenate([o_rep[i] / obs_scale, a_v[i]]).astype(np.float32))

            next_obs = np.zeros_like(obs); rew = np.zeros(n_envs, np.float32); done = np.zeros(n_envs, np.float32)
            for i in range(n_envs):
                no, r, term, trunc, _ = envs[i].step(a_exec[i])
                d = term or trunc
                ep_ret[i] += r
                if d:
                    returns_log.append(ep_ret[i]); ep_ret[i] = 0.0
                    seq = np.array(ep_seq[i], np.float32)
                    L = min(len(seq), args.traj_max_len)
                    padded = np.zeros((args.traj_max_len, obs_dim + act_dim), np.float32)
                    padded[:L] = seq[:L]
                    atk_seqs_this_update.append(padded); atk_lens_this_update.append(L)
                    rew_st[t, i] = 0.0  # filled in after this update's trajectory-critic pass
                    ep_seq[i] = []
                    no, _ = envs[i].reset()
                next_obs[i] = no; rew[i] = r; done[i] = float(d)

            rew_eff[t] = -rew
            bh["obs"][t] = h_adv_in; bh["act"][t] = dh.cpu().numpy(); bh["logp"][t] = h_logp.cpu().numpy()
            bh["val_eff"][t] = h_veff.cpu().numpy(); bh["val_st"][t] = h_vst.cpu().numpy()
            bp["obs"][t] = p_adv_in; bp["act"][t] = dp.cpu().numpy(); bp["logp"][t] = p_logp.cpu().numpy()
            bp["val_eff"][t] = p_veff.cpu().numpy(); bp["val_st"][t] = p_vst.cpu().numpy()
            done_buf[t] = done
            last_done = done
            obs = next_obs

        # ---- Trajectory critic update + terminal stealth reward backfill ----
        ckl = float("nan")
        if atk_seqs_this_update:
            atk_seq_t = torch.as_tensor(np.stack(atk_seqs_this_update), device=device)
            atk_len_t = torch.as_tensor(np.array(atk_lens_this_update), device=device)
            n_clean_sample = min(len(atk_seqs_this_update), clean_seq.shape[0])
            clean_idx = np.random.choice(clean_seq.shape[0], n_clean_sample, replace=False)
            ckl, w_psi = update_trajectory_critic(
                traj_critic, traj_opt, clean_seq[clean_idx], clean_len[clean_idx],
                atk_seq_t, atk_len_t, epochs=args.traj_epoch)
            # backfill: walk done_buf again to place each completed episode's w_psi at its
            # terminal step (same order episodes were appended to atk_seqs_this_update)
            k = 0
            for t in range(T):
                for i in range(n_envs):
                    if done_buf[t, i] > 0:
                        rew_st[t, i] = float(w_psi[k]); k += 1

        # ---- bootstrap + GAE (4 advantage streams) ----
        h_boot_in = (obs / scale).astype(np.float32)
        a_v_boot = np.clip(victim_act_batch(victim, obs), lo, hi)
        p_boot_in = np.concatenate([obs / scale, a_v_boot], axis=1).astype(np.float32)
        with torch.no_grad():
            h_lv_eff = h_net.get_value(torch.as_tensor(h_boot_in, device=device)).cpu().numpy()
            h_lv_st = h_net.get_fake_value(torch.as_tensor(h_boot_in, device=device)).cpu().numpy()
            p_lv_eff = p_net.get_value(torch.as_tensor(p_boot_in, device=device)).cpu().numpy()
            p_lv_st = p_net.get_fake_value(torch.as_tensor(p_boot_in, device=device)).cpu().numpy()

        adv_h_eff, ret_h_eff = gae(rew_eff, bh["val_eff"], done_buf, h_lv_eff, args.gamma, args.gae_lambda)
        adv_h_st, ret_h_st = gae(rew_st, bh["val_st"], done_buf, h_lv_st, args.gamma_st, args.gae_lambda)
        adv_p_eff, ret_p_eff = gae(rew_eff, bp["val_eff"], done_buf, p_lv_eff, args.gamma, args.gae_lambda)
        adv_p_st, ret_p_st = gae(rew_st, bp["val_st"], done_buf, p_lv_st, args.gamma_st, args.gae_lambda)

        # Role-specific joint advantage A_lambda = A_eff + lambda*A_st (SAME lambda, both roles)
        bh["adv"] = adv_h_eff + lam * adv_h_st; bh["ret_eff"] = ret_h_eff; bh["ret_st"] = ret_h_st
        bp["adv"] = adv_p_eff + lam * adv_p_st; bp["ret_eff"] = ret_p_eff; bp["ret_st"] = ret_p_st

        # ---- Ordered actor update: Hidden first, then Performance with upstream correction ----
        plh, vleh, vlsh, enh = ppo_update_dual(h_net, h_opt, bh, args, h_in, obs_dim, device)
        with torch.no_grad():
            flat_h_obs = torch.as_tensor(bh["obs"].reshape(-1, h_in), device=device)
            flat_h_act = torch.as_tensor(bh["act"].reshape(-1, obs_dim), device=device)
            flat_h_logp_old = torch.as_tensor(bh["logp"].reshape(-1), device=device)
            new_h_logp, _, _, _ = h_net.evaluate_actions_v(flat_h_obs, flat_h_act)
            upstream_ratio = torch.exp(new_h_logp - flat_h_logp_old).clamp(0.1, 10.0)
        plp, vlep, vlsp, enp = ppo_update_dual(p_net, p_opt, bp, args, p_in, act_dim, device,
                                                upstream_ratio=upstream_ratio)

        # ---- shared dual-ascent lambda on the trajectory-level KL estimate ----
        if not np.isnan(ckl):
            ckl_hist.append(ckl)
            lam = min(max(lam + args.alpha_lambda * (ckl - args.eps_kl), 0.0), args.lambda_max)

        ret_hist.extend(returns_log)
        ret_mean = float(np.mean(ret_hist)) if ret_hist else float("nan")
        ckl_mean = float(np.mean(ckl_hist)) if ckl_hist else float("nan")
        steps = update * T * n_envs
        if update % args.log_interval == 0 or update == 1:
            sps = int(steps / (time.time() - start))
            print(f"[sc-marl] upd {update}/{num_updates} step {steps} victim_ret {ret_mean:.1f} "
                  f"C_KL {ckl_mean:.3f} lambda {lam:.3f} "
                  f"Hloss(p/ve/vs) {plh:.3f}/{vleh:.3f}/{vlsh:.3f} "
                  f"Ploss(p/ve/vs) {plp:.3f}/{vlep:.3f}/{vlsp:.3f} ({sps} sps)")
            save_ckpt()

    for e in envs:
        e.close()
    save_ckpt()
    print(f"[sc-marl] saved -> {ckpt_path}")
    return ckpt_path


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", default="Ant-v4")
    p.add_argument("--victim_run", default="")
    p.add_argument("--budget_obs", type=float, default=4.0)
    p.add_argument("--budget_act", type=float, default=4.0)
    p.add_argument("--num_env_steps", type=int, default=2_000_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99, help="discount for the effectiveness (damage) reward stream")
    p.add_argument("--gamma_st", type=float, default=0.99, help="discount for the terminal stealth reward stream")
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--ppo_epoch", type=int, default=4)
    p.add_argument("--num_mini_batch", type=int, default=4)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    # shared stealth constraint (single lambda, per explicit user spec)
    p.add_argument("--lambda_init", type=float, default=1.0)
    p.add_argument("--alpha_lambda", type=float, default=0.05)
    p.add_argument("--lambda_max", type=float, default=1e6)
    p.add_argument("--eps_kl", type=float, default=0.0, help="target trajectory-level KL estimate (dual ascent)")
    # trajectory critic
    p.add_argument("--traj_hidden", type=int, default=128)
    p.add_argument("--traj_lr", type=float, default=1e-3)
    p.add_argument("--traj_epoch", type=int, default=4, help="discriminator updates per outer iteration")
    p.add_argument("--traj_max_len", type=int, default=300, help="episode length used for the rollout AND the GRU pad/truncate length")
    p.add_argument("--n_clean_ref", type=int, default=64, help="size of the one-off clean reference episode pool")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save_dir", default="results/obs_attackers/Ant-v4")
    p.add_argument("--save_name", default="sc_marl.pt")
    p.add_argument("--log_interval", type=int, default=5)
    args = p.parse_args()
    # Rollout window == episode length so every episode that starts in a window also completes
    # in it (barring early term) -- keeps the trajectory-critic episode extraction simple.
    args.num_steps = args.traj_max_len

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cpu"
    default_victim = {"Ant-v4": "results/robust_victim/Ant-v4/mappo/*/seed-00001-*",
                       "HalfCheetah-v4": "results/robust_victim/HalfCheetah-v4/mappo/victim6m/seed-00001-*"}
    victim_run = resolve_victim_run(args.victim_run or default_victim.get(
        args.scenario, f"results/robust_victim/{args.scenario}/mappo/*/seed-00001-*"))
    probe = gym.make(args.scenario)
    obs_dim = probe.observation_space.shape[0]
    act_space = Box(probe.action_space.low, probe.action_space.high, probe.action_space.shape, np.float32)
    lo = act_space.low.astype(np.float32); hi = act_space.high.astype(np.float32)
    probe.close()
    victim = Victim(victim_run, obs_dim, act_space, device)
    obs_scale, act_scale = estimate_scales(args.scenario, victim, lo, hi)
    train(args, victim, obs_scale, act_scale, device)


if __name__ == "__main__":
    main()
