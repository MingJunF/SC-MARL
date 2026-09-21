"""Simplified SC-MAPPO for SafetyPointGoal1 (2026-09-17), per explicit user design spec: a
minimal centralized-cost-critic Lagrangian variant of the existing (more complex, ordered-update
+ importance-weighted + role-conditioned-dual-critic) `train_sc_marl_safety.py`.

The three changes from a plain synchronous MAPPO, per the user's own writeup:
  1. A trajectory-level stealth cost C(zeta) is computed once per completed episode and placed
     as a TERMINAL cost (0 everywhere else). C(zeta) = the trained TrajectoryCritic's logit on
     that reported-trajectory (o_rep/scale, a_v) sequence (HIGHER logit = more attack-like =
     worse) -- an INDEPENDENT learned discriminator, per this project's standing
     "detector-independent training, PEDM only for held-out eval" convention (see
     `train_sc_marl_safety.py`'s own docstring). **CORRECTED 2026-09-17**: the first version of
     this script used the already-FIT, held-out `pedm_detector_v23.pt` directly as C(zeta) --
     per explicit user correction ("你这样变成拟合pedm了"), that made the later "does it pass
     PEDM" check circular/meaningless, since the attacker was directly optimizing against the
     EXACT model it would then be judged by (a white-box adaptive attack against a known
     detector, not evidence of general stealth). Swapped to TrajectoryCritic so PEDM stays a
     genuinely held-out judge, matching SC-MARL's own established pattern.
  2. ONE centralized cost critic V_c(x_t) (alongside a centralized reward critic V_r(x_t)) sees
     the joint state x_t = (obs_aug, o_rep, a_v, u_dec_raw, u_dis_raw) -- richer than either
     actor's own decentralized input -- and both are trained via ordinary GAE (gamma_c=1 for the
     cost, since it's a one-shot terminal signal).
  3. BOTH attackers (deceiver h_net = obs-channel, disruptor p_net = act-channel) are updated
     with the SAME joint advantage A^r - lambda*A^c via standard synchronous PPO -- no upstream
     importance-ratio correction, no ordered update, no per-role dual critics.
  lambda is dual-ascended each update toward --eps_stealth (default 0.0 -- the discriminator's
  own "coin-flip" equilibrium logit, i.e. "indistinguishable from clean", NOT tied to any
  specific fixed detector's calibration).

Physical-effectiveness reward reuses the already-validated `haz_shaping` design from
`train_safety_attacker.py`/`train_sc_marl_safety.py` (dense hazard-approach potential + one-off
entry bonus) rather than `-victim_reward` -- Safety-Gym's raw task reward is blind to hazard
contact (reward/cost are independent channels), so `-r_v` alone taught nothing on this env
(established earlier this session); the user's own spec treats the effectiveness reward as a
slot to be filled with whatever already works, not a mandated `-r_v`.

Usage:
  python examples/train_sc_mappo_safety.py --budget_act 3.0 --budget_obs 10.0 \
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
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import safety_gymnasium  # noqa: F401
import examples.safety_hazard_patch  # noqa: F401
import examples.safety_layout_patch  # noqa: F401
from harl.attacks.adversary_ac import AdversaryActorCritic, orthogonal_init
from examples.train_marl_attacker import gae
from examples.train_safety_attacker import SafetyVictim
from examples.train_sc_marl import TrajectoryCritic, update_trajectory_critic


def dist_to_hazard(env):
    task = env.unwrapped.task
    return float(np.linalg.norm(task.agent.pos[:2] - task.hazards.pos[0][:2]))


@torch.no_grad()
def collect_clean_reference(victim, scenario, full_dim, act_dim, scale_full, n_episodes, ep_len, device, seed=9000):
    """Clean (no-attacker) reference episodes for the TrajectoryCritic, (o_aug/scale_full, a_v)
    sequences -- mirrors `train_sc_marl_safety.py`'s own `collect_clean_reference` exactly."""
    seqs = np.zeros((n_episodes, ep_len, full_dim + act_dim), np.float32)
    lens = np.zeros(n_episodes, np.int64)
    for i in range(n_episodes):
        env = gym.make(scenario)
        o, _ = env.reset(seed=seed + i)
        steps_since_goal = 0
        for t in range(ep_len):
            o_aug = victim._augment(o.astype(np.float32), steps_since_goal, env)  # noqa: SLF001
            a_v = victim.act_from_augmented(o_aug)
            env_action = np.clip(a_v, -1.0, 1.0)
            seqs[i, t] = np.concatenate([o_aug / scale_full, a_v])
            no, r, term, trunc, info = env.step(env_action)
            steps_since_goal = 0 if info.get("goal_met", False) else steps_since_goal + 1
            o = no
            if term or trunc:
                lens[i] = t + 1
                break
        else:
            lens[i] = ep_len
        env.close()
    return torch.as_tensor(seqs, device=device), torch.as_tensor(lens, device=device)


class CentralizedCritic(nn.Module):
    """V_r(x_t), V_c(x_t) from a shared trunk -- dual-head is an implementation detail (per the
    user's spec, "不是主要创新"), NOT a separate network per attacker role: the SAME joint
    (reward_adv, cost_adv) feeds both h_net's and p_net's PPO update."""

    def __init__(self, x_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            orthogonal_init(nn.Linear(x_dim, hidden)), nn.Tanh(),
            orthogonal_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.v_r = orthogonal_init(nn.Linear(hidden, 1), gain=1.0)
        self.v_c = orthogonal_init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, x):
        h = self.trunk(x)
        return self.v_r(h).squeeze(-1), self.v_c(h).squeeze(-1)


def ppo_update_actor(net, opt, obs, act, old_logp, adv, args, device):
    """Standard clipped-surrogate PPO update for ONE actor -- no value loss here (the value
    comes from the separate CentralizedCritic, updated independently below); `net`'s own
    internal critic head (from AdversaryActorCritic) is simply never read."""
    obs_t = torch.as_tensor(obs, device=device)
    act_t = torch.as_tensor(act, device=device)
    old_logp_t = torch.as_tensor(old_logp, device=device)
    adv_t = torch.as_tensor(adv, device=device)
    n = obs_t.shape[0]
    mb = max(1, n // args.num_mini_batch)
    last_pg, last_ent = 0.0, 0.0
    for _ in range(args.ppo_epoch):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            logp, ent, _ = net.evaluate_actions(obs_t[idx], act_t[idx])
            ratio = torch.exp(logp - old_logp_t[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_t[idx]
            pg_loss = -torch.min(surr1, surr2).mean()
            loss = pg_loss - args.ent_coef * ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()
            last_pg, last_ent = float(pg_loss.detach()), float(ent.mean().detach())
    return last_pg, last_ent


def update_central_critic(critic, opt, x, ret_r, ret_c, args, device):
    x_t = torch.as_tensor(x, device=device)
    ret_r_t = torch.as_tensor(ret_r, device=device)
    ret_c_t = torch.as_tensor(ret_c, device=device)
    n = x_t.shape[0]
    mb = max(1, n // args.num_mini_batch)
    last_loss = 0.0
    for _ in range(args.ppo_epoch):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            vr, vc = critic(x_t[idx])
            loss = args.vf_coef * (((vr - ret_r_t[idx]) ** 2).mean() + ((vc - ret_c_t[idx]) ** 2).mean())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            opt.step()
            last_loss = float(loss)
    return last_loss


def train(args, victim, device):
    scenario = args.scenario
    n_envs, T = args.num_envs, args.traj_max_len
    envs = [gym.make(scenario) for _ in range(n_envs)]
    act_dim = envs[0].action_space.shape[0]
    lo = envs[0].action_space.low.astype(np.float32)
    hi = envs[0].action_space.high.astype(np.float32)

    full_dim = victim.full_obs_dim
    obs_scale_full = np.maximum(np.sqrt(victim.obs_norm.var[:full_dim]).astype(np.float32), 0.01)
    scale_full = obs_scale_full[None, :]
    ascale = np.ones(act_dim, np.float32)[None, :]

    h_in = full_dim
    p_in = full_dim + act_dim
    x_dim = 3 * full_dim + 2 * act_dim  # obs_aug, o_rep, a_v, u_dec(raw), u_dis(raw)

    h_net = AdversaryActorCritic(h_in, full_dim, hidden_size=256, activation="tanh").to(device)
    p_net = AdversaryActorCritic(p_in, act_dim, hidden_size=256, activation="tanh").to(device)
    critic = CentralizedCritic(x_dim, hidden=256).to(device)
    h_opt = torch.optim.Adam(h_net.parameters(), lr=args.lr, eps=1e-5)
    p_opt = torch.optim.Adam(p_net.parameters(), lr=args.lr, eps=1e-5)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

    traj_critic = TrajectoryCritic(full_dim, act_dim, hidden=args.traj_hidden).to(device)
    traj_opt = torch.optim.Adam(traj_critic.parameters(), lr=args.traj_lr)
    print(f"[sc-mappo-safety] collecting {args.n_clean_ref} clean reference episodes for the TrajectoryCritic...")
    clean_seq, clean_len = collect_clean_reference(
        victim, scenario, full_dim, act_dim, obs_scale_full, args.n_clean_ref, T, device)

    lam = args.lambda_init
    eps_stealth = args.eps_stealth
    print(f"[sc-mappo-safety] stealth target eps={eps_stealth:.4f} (TrajectoryCritic logit, "
          f"0.0 = indistinguishable-from-clean equilibrium)")

    HAZARD_REWARD_DISTANCE = 1.0
    HAZARD_ENTRY_BONUS = 10.0

    obs = np.zeros((n_envs, 60), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=args.seed + i); obs[i] = o
    steps_since_goal = np.zeros(n_envs, np.int32)
    last_dist_hazard = np.array([dist_to_hazard(e) for e in envs], np.float32)
    was_inside_hazard = np.zeros(n_envs, dtype=bool)
    ep_seq = [[] for _ in range(n_envs)]  # persistent per-env (o_rep/scale, a_v) log for the TrajectoryCritic

    num_updates = int(args.num_env_steps // (T * n_envs))
    ret_hist, cost_hist, stealth_hist = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    start = time.time()
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)

    def save_ckpt():
        torch.save({
            "h_state_dict": h_net.state_dict(), "p_state_dict": p_net.state_dict(),
            "critic_state_dict": critic.state_dict(), "traj_critic_state_dict": traj_critic.state_dict(),
            "h_in": h_in, "p_in": p_in, "x_dim": x_dim, "full_obs_dim": full_dim, "act_dim": act_dim,
            "budget_obs": args.budget_obs, "budget_act": args.budget_act,
            "obs_scale": obs_scale_full, "lambda": float(lam), "scenario": scenario, "algo": "sc_mappo_safety",
        }, ckpt_path)

    for update in range(1, num_updates + 1):
        b_h_obs = np.zeros((T, n_envs, h_in), np.float32)
        b_h_act = np.zeros((T, n_envs, full_dim), np.float32)
        b_h_logp = np.zeros((T, n_envs), np.float32)
        b_p_obs = np.zeros((T, n_envs, p_in), np.float32)
        b_p_act = np.zeros((T, n_envs, act_dim), np.float32)
        b_p_logp = np.zeros((T, n_envs), np.float32)
        b_x = np.zeros((T, n_envs, x_dim), np.float32)
        rew_eff = np.zeros((T, n_envs), np.float32)
        cost_stealth = np.zeros((T, n_envs), np.float32)
        done_buf = np.zeros((T, n_envs), np.float32)
        ep_ret = np.zeros(n_envs, np.float32); ep_cost = np.zeros(n_envs, np.float32)
        returns_log, costs_log, stealth_log = [], [], []
        atk_seqs_this_update, atk_lens_this_update, atk_done_slots = [], [], []

        for t in range(T):
            obs_aug = victim._augment(obs, steps_since_goal, envs)  # noqa: SLF001
            h_adv_in = (obs_aug / scale_full).astype(np.float32)
            with torch.no_grad():
                u_dec, h_logp, _ = h_net.act(torch.as_tensor(h_adv_in, device=device))
            u_dec_np = u_dec.cpu().numpy()
            nrm = np.linalg.norm(u_dec_np, axis=1, keepdims=True)
            phi_h = u_dec_np * np.minimum(1.0, args.budget_obs / (nrm + 1e-8))
            o_rep = obs_aug + phi_h * scale_full

            a_v = victim.act_from_augmented(o_rep)

            p_adv_in = np.concatenate([o_rep / scale_full, a_v], axis=1).astype(np.float32)
            with torch.no_grad():
                u_dis, p_logp, _ = p_net.act(torch.as_tensor(p_adv_in, device=device))
            u_dis_np = u_dis.cpu().numpy()
            nrm = np.linalg.norm(u_dis_np, axis=1, keepdims=True)
            phi_p = u_dis_np * np.minimum(1.0, args.budget_act / (nrm + 1e-8))
            a_exec = np.clip(a_v + phi_p * ascale, lo, hi)

            x_t = np.concatenate([obs_aug, o_rep, a_v, u_dec_np, u_dis_np], axis=1).astype(np.float32)

            haz_shaping = np.zeros(n_envs, np.float32)
            next_obs = np.zeros_like(obs); done = np.zeros(n_envs, np.float32)
            for i in range(n_envs):
                no, r, term, trunc, info = envs[i].step(a_exec[i])
                c = float(info.get("cost", 0.0))
                d = term or trunc
                dh_new = dist_to_hazard(envs[i])
                haz_shaping[i] = (last_dist_hazard[i] - dh_new) * HAZARD_REWARD_DISTANCE
                if c > 0 and not was_inside_hazard[i]:
                    haz_shaping[i] += HAZARD_ENTRY_BONUS
                was_inside_hazard[i] = c > 0
                last_dist_hazard[i] = dh_new
                steps_since_goal[i] = 0 if info.get("goal_met", False) else steps_since_goal[i] + 1
                ep_ret[i] += r; ep_cost[i] += c
                ep_seq[i].append(np.concatenate([o_rep[i] / scale_full[0], a_v[i]]).astype(np.float32))
                if d:
                    returns_log.append(ep_ret[i]); costs_log.append(ep_cost[i])
                    ep_ret[i] = 0.0; ep_cost[i] = 0.0
                    seq = np.array(ep_seq[i], np.float32)
                    L = min(len(seq), T)
                    padded = np.zeros((T, full_dim + act_dim), np.float32)
                    padded[:L] = seq[:L]
                    atk_seqs_this_update.append(padded); atk_lens_this_update.append(L)
                    atk_done_slots.append((t, i))  # where to write this episode's C(zeta) once scored below
                    ep_seq[i] = []
                    no, _ = envs[i].reset()
                    last_dist_hazard[i] = dist_to_hazard(envs[i])
                    was_inside_hazard[i] = False
                    steps_since_goal[i] = 0
                next_obs[i] = no; done[i] = float(d)

            rew_eff[t] = haz_shaping
            b_h_obs[t] = h_adv_in; b_h_act[t] = u_dec_np; b_h_logp[t] = h_logp.cpu().numpy()
            b_p_obs[t] = p_adv_in; b_p_act[t] = u_dis_np; b_p_logp[t] = p_logp.cpu().numpy()
            b_x[t] = x_t
            done_buf[t] = done
            obs = next_obs

        ckl = float("nan")
        if atk_seqs_this_update:
            atk_seq_t = torch.as_tensor(np.stack(atk_seqs_this_update), device=device)
            atk_len_t = torch.as_tensor(np.array(atk_lens_this_update), device=device)
            n_clean_sample = min(len(atk_seqs_this_update), clean_seq.shape[0])
            clean_idx = np.random.choice(clean_seq.shape[0], n_clean_sample, replace=False)
            ckl, w_psi = update_trajectory_critic(
                traj_critic, traj_opt, clean_seq[clean_idx], clean_len[clean_idx],
                atk_seq_t, atk_len_t, epochs=args.traj_epoch)
            # C(zeta) = raw attacked-episode logit (HIGHER = more attack-like = worse for the
            # attacker) = -w_psi (w_psi is defined as the NEGATIVE logit, "how clean-like");
            # write each completed episode's terminal cost into its own (t,i) rollout slot.
            for k, (t_done, i_done) in enumerate(atk_done_slots):
                cost_stealth[t_done, i_done] = float(-w_psi[k])
                stealth_log.append(float(-w_psi[k]))

        with torch.no_grad():
            obs_aug_boot = victim._augment(obs, steps_since_goal, envs)  # noqa: SLF001
            h_boot_in = (obs_aug_boot / scale_full).astype(np.float32)
            a_v_boot = victim.act_from_augmented(obs_aug_boot)
            p_boot_in = np.concatenate([obs_aug_boot / scale_full, a_v_boot], axis=1).astype(np.float32)
            u_dec_boot, _, _ = h_net.act(torch.as_tensor(h_boot_in, device=device), deterministic=True)
            u_dis_boot, _, _ = p_net.act(torch.as_tensor(p_boot_in, device=device), deterministic=True)
            x_boot = np.concatenate([obs_aug_boot, obs_aug_boot, a_v_boot,
                                      u_dec_boot.cpu().numpy(), u_dis_boot.cpu().numpy()], axis=1).astype(np.float32)
            v_r_boot, v_c_boot = critic(torch.as_tensor(x_boot, device=device))
            v_r_boot = v_r_boot.cpu().numpy(); v_c_boot = v_c_boot.cpu().numpy()
            flat_x_all = torch.as_tensor(b_x.reshape(-1, x_dim), device=device)
            v_r_all, v_c_all = critic(flat_x_all)
            v_r_all = v_r_all.cpu().numpy().reshape(T, n_envs)
            v_c_all = v_c_all.cpu().numpy().reshape(T, n_envs)

        adv_r, ret_r = gae(rew_eff, v_r_all, done_buf, v_r_boot, args.gamma, args.gae_lambda)
        adv_c, ret_c = gae(cost_stealth, v_c_all, done_buf, v_c_boot, args.gamma_c, args.gae_lambda)

        joint_adv = adv_r - lam * adv_c
        joint_adv = (joint_adv - joint_adv.mean()) / (joint_adv.std() + 1e-8)
        joint_adv_flat = joint_adv.reshape(-1)

        plh, enh = ppo_update_actor(h_net, h_opt, b_h_obs.reshape(-1, h_in), b_h_act.reshape(-1, full_dim),
                                     b_h_logp.reshape(-1), joint_adv_flat, args, device)
        plp, enp = ppo_update_actor(p_net, p_opt, b_p_obs.reshape(-1, p_in), b_p_act.reshape(-1, act_dim),
                                     b_p_logp.reshape(-1), joint_adv_flat, args, device)
        vloss = update_central_critic(critic, c_opt, b_x.reshape(-1, x_dim), ret_r.reshape(-1), ret_c.reshape(-1),
                                       args, device)

        if stealth_log:
            stealth_hist.extend(stealth_log)
            stealth_mean = float(np.mean(stealth_log))
            lam = min(args.lambda_max, max(0.0, lam + args.alpha_lambda * (stealth_mean - eps_stealth)))

        ret_hist.extend(returns_log); cost_hist.extend(costs_log)
        ret_mean = float(np.mean(ret_hist)) if ret_hist else float("nan")
        cost_mean = float(np.mean(cost_hist)) if cost_hist else float("nan")
        stealth_mean_log = float(np.mean(stealth_hist)) if stealth_hist else float("nan")
        steps = update * T * n_envs
        if update % args.log_interval == 0 or update == 1:
            sps = int(steps / (time.time() - start))
            print(f"[sc-mappo-safety] upd {update}/{num_updates} step {steps} victim_ret {ret_mean:.2f} "
                  f"victim_cost/ep {cost_mean:.2f} stealth_C {stealth_mean_log:.4f} (eps {eps_stealth:.4f}) "
                  f"C_KL {ckl:.3f} lambda {lam:.3f} Hloss(p/ent) {plh:.3f}/{enh:.3f} Ploss(p/ent) {plp:.3f}/{enp:.3f} "
                  f"vloss {vloss:.3f} ({sps} sps)")
            save_ckpt()

    for e in envs:
        e.close()
    save_ckpt()
    print(f"[sc-mappo-safety] saved -> {ckpt_path}")
    return ckpt_path


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", default="SafetyPointGoal1Gymnasium-v0")
    p.add_argument("--victim_ckpt", default="results/safety_gym/SafetyPointGoal1/victim/victim.pt")
    p.add_argument("--budget_obs", type=float, default=10.0)
    p.add_argument("--budget_act", type=float, default=3.0)
    p.add_argument("--num_env_steps", type=int, default=2_000_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gamma_c", type=float, default=1.0,
                   help="discount for the cost GAE -- 1.0 since stealth cost is a one-shot terminal signal per the user's spec")
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--ppo_epoch", type=int, default=4)
    p.add_argument("--num_mini_batch", type=int, default=4)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--lambda_init", type=float, default=1.0)
    p.add_argument("--alpha_lambda", type=float, default=0.05)
    p.add_argument("--lambda_max", type=float, default=1e6)
    p.add_argument("--eps_stealth", type=float, default=0.0,
                   help="target mean stealth cost C(zeta) (TrajectoryCritic logit) for the dual-ascent lambda -- 0.0 is the discriminator's own coin-flip equilibrium (attacked episodes as classifiable as random guessing), NOT tied to any specific fixed detector's calibration (2026-09-17: switched away from the PEDM-threshold default after the PEDM-as-training-signal design was found circular -- see module docstring)")
    p.add_argument("--traj_hidden", type=int, default=128)
    p.add_argument("--traj_lr", type=float, default=1e-3)
    p.add_argument("--traj_epoch", type=int, default=4)
    p.add_argument("--n_clean_ref", type=int, default=64)
    p.add_argument("--traj_max_len", type=int, default=1000,
                   help="MUST match the real Safety-Gym episode length (1000, TimeLimit-enforced) so each rollout window aligns with one full episode per env")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save_dir", default="results/safety_gym/SafetyPointGoal1")
    p.add_argument("--save_name", default="sc_mappo_safety.pt")
    p.add_argument("--log_interval", type=int, default=5)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cpu"
    victim = SafetyVictim(args.victim_ckpt, device)
    train(args, victim, device)


if __name__ == "__main__":
    main()
