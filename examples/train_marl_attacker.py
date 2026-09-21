"""Joint MARL attacker (canonical in-loop): an observation agent + an action agent
cooperate to minimize victim reward while suppressing the detector's per-step
prediction error (the quantity CUSUM accumulates).

Canonical threat model (obeyed): the forged observation o_rep is seen by the
victim, the action agent, AND the detector. Per env step:
  o_rep = s + Proj_{Bo}(delta_o)            # obs agent forges (victim+act+detector see it)
  a_v   = pi(o_rep)                          # victim acts on forged obs
  a_app = a_v + Proj_{Ba}(delta_a)           # act agent perturbs, sees o_rep and a_v
  s'    = env.step(a_app)                    # real state evolves
  z     = || o_rep - p(o_rep_prev, a_v_prev) ||   # detectability residual (CUSUM input)

Team reward: r_att = -r_victim - lambda * ||o_rep - expected||^2  (dual-ascent lambda).
Both agents are independent PPO (IPPO) sharing this team reward. Reference: illusory
penalty, but the stealth term targets the sequential-detector (CUSUM) residual.
"""
import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.attacks.adversary_ac import AdversaryActorCritic
from examples.train_obs_attacker import (
    AntTrueDynamics, l2_project, victim_act_batch, estimate_scales, resolve_victim_run,
)
from examples.eval_robust_detector import Victim
from harl.detectors.pedm_detector import PEDMDetector
from gymnasium.spaces import Box


def pedm_batch_preds(pedm, states, actions, n_part):
    """Sampled next-state predictions, (n, n_part, obs_dim)."""
    return pedm.dyn_model.one_step_batch_preds(
        states=states.astype(np.float32), actions=actions.astype(np.float32), n_part=n_part)


def pedm_min_mean_score(preds, next_obs):
    """The detector's per-step score (min over particles of mean-abs-error)."""
    err = np.abs(preds - next_obs[:, None, :]).mean(axis=2)  # (n, n_part)
    return err.min(axis=1)  # (n,)


def ppo_update(net, optim, buf, args, in_dim, out_dim, device):
    b_obs = torch.as_tensor(buf["obs"].reshape(-1, in_dim), device=device)
    b_act = torch.as_tensor(buf["act"].reshape(-1, out_dim), device=device)
    b_logp = torch.as_tensor(buf["logp"].reshape(-1), device=device)
    b_adv = torch.as_tensor(buf["adv"].reshape(-1), device=device)
    b_ret = torch.as_tensor(buf["ret"].reshape(-1), device=device)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
    bs = b_obs.shape[0]
    mb = bs // args.num_mini_batch
    pl, vl, en = [], [], []
    for _ in range(args.ppo_epoch):
        perm = torch.randperm(bs, device=device)
        for s in range(0, bs, mb):
            idx = perm[s:s + mb]
            nlp, ent, val = net.evaluate_actions(b_obs[idx], b_act[idx])
            ratio = torch.exp(nlp - b_logp[idx])
            s1 = ratio * b_adv[idx]
            s2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * b_adv[idx]
            ploss = -torch.min(s1, s2).mean()
            vloss = 0.5 * (val - b_ret[idx]).pow(2).mean()
            loss = ploss + args.vf_coef * vloss - args.ent_coef * ent.mean()
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optim.step()
            pl.append(ploss.item()); vl.append(vloss.item()); en.append(ent.mean().item())
    return np.mean(pl), np.mean(vl), np.mean(en)


def gae(rew, val, done, last_val, gamma, lam):
    T, n = rew.shape
    adv = np.zeros((T, n), np.float32)
    g = np.zeros(n, np.float32)
    nv = last_val
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
        nv = val[t]
    return adv, adv + val


def train(args, victim, obs_scale, act_scale, device):
    scenario = args.scenario
    n_envs, T = args.num_envs, args.num_steps
    obs_dim = obs_scale.shape[0]
    scale = obs_scale[None, :]
    envs = [gym.make(scenario) for _ in range(n_envs)]
    dyn = AntTrueDynamics(scenario)
    act_dim = envs[0].action_space.shape[0]
    pedm = None
    if args.penalty == "pedm":
        pedm = PEDMDetector(obs_dim, act_dim, n_part=args.pedm_n_part, device=device)
        pedm.load(args.pedm_path)
        print(f"[marl] PEDM-driven penalty: {args.pedm_path} (n_part={args.pedm_n_part})")
    lo = envs[0].action_space.low.astype(np.float32)
    hi = envs[0].action_space.high.astype(np.float32)
    ascale = act_scale[None, :]

    obs_in = obs_dim * 3            # [s, o_rep_prev, expected]
    act_in = obs_dim + act_dim      # [o_rep, a_v]
    obs_net = AdversaryActorCritic(obs_in, obs_dim, hidden_size=256, activation="tanh").to(device)
    act_net = AdversaryActorCritic(act_in, act_dim, hidden_size=256, activation="tanh").to(device)
    obs_opt = torch.optim.Adam(obs_net.parameters(), lr=args.lr, eps=1e-5)
    act_opt = torch.optim.Adam(act_net.parameters(), lr=args.lr, eps=1e-5)
    lam = args.lambda_init

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                         entity=args.wandb_entity or None,
                         name=f"{os.path.splitext(args.save_name)[0]}_bo{args.budget_obs}_ba{args.budget_act}_seed{args.seed}",
                         config=vars(args), reinit=True)

    obs = np.zeros((n_envs, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=args.seed + i)
        obs[i] = o
    o_rep_prev = obs.copy()
    a_v_prev = np.zeros((n_envs, act_dim), np.float32)
    last_done = np.zeros(n_envs, np.float32)

    num_updates = int(args.num_env_steps // (T * n_envs))
    ret_hist, pen_hist = deque(maxlen=100), deque(maxlen=100)
    recent_pen = []
    start = time.time()

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)

    def save_ckpt():
        torch.save({"obs_state_dict": obs_net.state_dict(), "act_state_dict": act_net.state_dict(),
                    "obs_in": obs_in, "act_in": act_in, "obs_dim": obs_dim, "act_dim": act_dim,
                    "budget_obs": args.budget_obs, "budget_act": args.budget_act,
                    "penalty": args.penalty,
                    "obs_scale": obs_scale, "act_scale": act_scale}, ckpt_path)

    for update in range(1, num_updates + 1):
        bo = {"obs": np.zeros((T, n_envs, obs_in), np.float32),
              "act": np.zeros((T, n_envs, obs_dim), np.float32),
              "logp": np.zeros((T, n_envs), np.float32), "val": np.zeros((T, n_envs), np.float32)}
        ba = {"obs": np.zeros((T, n_envs, act_in), np.float32),
              "act": np.zeros((T, n_envs, act_dim), np.float32),
              "logp": np.zeros((T, n_envs), np.float32), "val": np.zeros((T, n_envs), np.float32)}
        rew_buf = np.zeros((T, n_envs), np.float32)
        done_buf = np.zeros((T, n_envs), np.float32)
        ep_ret = np.zeros(n_envs, np.float32)
        returns_log, pen_log = [], []

        for t in range(T):
            mask = last_done[:, None]
            o_rep_prev_eff = o_rep_prev * (1 - mask) + obs * mask
            a_v_prev_eff = a_v_prev * (1 - mask)
            preds = None
            if pedm is not None:
                # one PEDM forward at (o_rep_prev, a_v_prev): mean = expected feature, reused for penalty
                preds = pedm_batch_preds(pedm, o_rep_prev_eff, a_v_prev_eff, args.pedm_n_part)
                expected = preds.mean(axis=1).astype(np.float32)
            else:
                expected = np.zeros((n_envs, obs_dim), np.float32)
                for i in range(n_envs):
                    expected[i] = dyn.next_obs(o_rep_prev[i], a_v_prev[i])
            expected = expected * (1 - mask) + obs * mask

            # obs agent forges o_rep (seen by victim, act agent, detector)
            obs_adv_in = np.concatenate([obs / scale, o_rep_prev_eff / scale, expected / scale], 1).astype(np.float32)
            with torch.no_grad():
                do, lo_p, vo = obs_net.act(torch.as_tensor(obs_adv_in, device=device))
            phi_o = l2_project(do.cpu().numpy(), args.budget_obs)
            o_rep = obs + phi_o * scale

            a_v = np.clip(victim_act_batch(victim, o_rep), lo, hi)

            # act agent perturbs action, sees o_rep and a_v
            act_adv_in = np.concatenate([o_rep / scale, a_v], 1).astype(np.float32)
            with torch.no_grad():
                da, la_p, va = act_net.act(torch.as_tensor(act_adv_in, device=device))
            phi_a = l2_project(da.cpu().numpy(), args.budget_act)
            a_app = np.clip(a_v + phi_a * ascale, lo, hi)

            # detectability residual (CUSUM input)
            if pedm is not None:
                # the detector's OWN min-mean score for transition (o_rep_prev, a_v_prev) -> o_rep
                penalty = pedm_min_mean_score(preds, o_rep) * (1 - last_done)
            else:
                penalty = np.sum(((o_rep - expected) / scale) ** 2, axis=1) * (1 - last_done)

            next_obs = np.zeros_like(obs); rew = np.zeros(n_envs, np.float32); done = np.zeros(n_envs, np.float32)
            for i in range(n_envs):
                no, r, term, trunc, _ = envs[i].step(a_app[i])
                d = term or trunc
                ep_ret[i] += r
                if d:
                    returns_log.append(ep_ret[i]); ep_ret[i] = 0.0
                    no, _ = envs[i].reset()
                next_obs[i] = no; rew[i] = r; done[i] = float(d)

            r_att = -rew - lam * penalty

            bo["obs"][t] = obs_adv_in; bo["act"][t] = do.cpu().numpy()
            bo["logp"][t] = lo_p.cpu().numpy(); bo["val"][t] = vo.cpu().numpy()
            ba["obs"][t] = act_adv_in; ba["act"][t] = da.cpu().numpy()
            ba["logp"][t] = la_p.cpu().numpy(); ba["val"][t] = va.cpu().numpy()
            rew_buf[t] = r_att; done_buf[t] = done
            pen_log.append(penalty[last_done == 0].mean() if np.any(last_done == 0) else 0.0)

            o_rep_prev = np.where(done[:, None] > 0, next_obs, o_rep)
            a_v_prev = np.where(done[:, None] > 0, np.zeros_like(a_v), a_v)
            last_done = done
            obs = next_obs

        # bootstrap values
        expected = np.zeros((n_envs, obs_dim), np.float32)
        for i in range(n_envs):
            expected[i] = dyn.next_obs(o_rep_prev[i], a_v_prev[i])
        obs_adv_in = np.concatenate([obs / scale, o_rep_prev / scale, expected / scale], 1).astype(np.float32)
        a_v_boot = np.clip(victim_act_batch(victim, obs), lo, hi)
        act_adv_in = np.concatenate([obs / scale, a_v_boot], 1).astype(np.float32)
        with torch.no_grad():
            lv_o = obs_net.get_value(torch.as_tensor(obs_adv_in, device=device)).cpu().numpy()
            lv_a = act_net.get_value(torch.as_tensor(act_adv_in, device=device)).cpu().numpy()

        adv_o, ret_o = gae(rew_buf, bo["val"], done_buf, lv_o, args.gamma, args.gae_lambda)
        adv_a, ret_a = gae(rew_buf, ba["val"], done_buf, lv_a, args.gamma, args.gae_lambda)
        bo["adv"], bo["ret"] = adv_o, ret_o
        ba["adv"], ba["ret"] = adv_a, ret_a
        plo, vlo, eno = ppo_update(obs_net, obs_opt, bo, args, obs_in, obs_dim, device)
        pla, vla, ena = ppo_update(act_net, act_opt, ba, args, act_in, act_dim, device)

        mean_pen = float(np.mean(pen_log)) if pen_log else 0.0
        recent_pen.append(mean_pen)
        d_pen = float(np.mean(recent_pen[-10:]))
        lam = max(lam + args.alpha_lambda * (d_pen - args.eps_pen), 0.0)
        lam = min(lam, args.lambda_max)  # cap prevents dual-ascent runaway at large budgets

        ret_hist.extend(returns_log); pen_hist.append(mean_pen)
        steps = update * T * n_envs
        ret_mean = float(np.mean(ret_hist)) if ret_hist else float("nan")
        if run is not None:
            run.log({"victim_return": ret_mean, "detectability_penalty": mean_pen, "lambda": lam,
                     "obs_ploss": plo, "act_ploss": pla, "update": update}, step=steps)
        if update % args.log_interval == 0 or update == 1:
            sps = int(steps / (time.time() - start))
            print(f"[marl] upd {update}/{num_updates} step {steps} victim_ret {ret_mean:.1f} "
                  f"penalty {mean_pen:.4f} lambda {lam:.3f} ({sps} sps)")
            save_ckpt()  # periodic checkpoint so the model is loadable mid-run

    for e in envs:
        e.close()
    if run is not None:
        run.finish()
    save_ckpt()
    print(f"[marl] saved -> {ckpt_path}")
    return ckpt_path


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", default="Ant-v4")
    p.add_argument("--victim_run", default="")
    p.add_argument("--budget_obs", type=float, default=0.4)
    p.add_argument("--budget_act", type=float, default=0.4)
    p.add_argument("--num_env_steps", type=int, default=400_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--ppo_epoch", type=int, default=4)
    p.add_argument("--num_mini_batch", type=int, default=4)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--lambda_init", type=float, default=1.0)
    p.add_argument("--alpha_lambda", type=float, default=0.1)
    p.add_argument("--lambda_max", type=float, default=1e9)
    p.add_argument("--eps_pen", type=float, default=0.05, help="target detectability penalty (dual ascent)")
    p.add_argument("--penalty", choices=["truedyn", "pedm"], default="truedyn",
                   help="truedyn: ||o_rep-p_true||^2 (scaled); pedm: detector min-mean residual (white-box)")
    p.add_argument("--pedm_path", default="results/obs_attackers/Ant-v4/pedm_detector.pt")
    p.add_argument("--pedm_n_part", type=int, default=20, help="particles for the training penalty (eval uses 100)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save_dir", default="results/obs_attackers/Ant-v4")
    p.add_argument("--save_name", default="marl_attacker.pt")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="illusory-marl")
    p.add_argument("--wandb_group", default="")
    p.add_argument("--wandb_entity", default="")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cpu"
    victim_run = resolve_victim_run(args.victim_run or
                                    "results/robust_victim/Ant-v4/mappo/*/seed-00001-*")
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
