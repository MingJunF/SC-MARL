"""Runner for MAPPO + a stealth-cost constraint (soft Lagrangian or hard bang-bang), 2026-09-17.

This is the HARL-native replacement for the earlier hand-rolled SC-MARL/standard-MAPPO
scripts. It reuses the unmodified `MAPPO` actor class and `VCritic` critic class as-is;
the only change versus plain `OnPolicyMARunner` is that a second ("cost") critic is
trained on a stealth cost, and PER-AGENT advantages are handed to `actor.train()` instead
of one shared `adv_r` (role-split, 2026-09-18 standing instruction -- see `SafetyMARLEnv`'s
module docstring): Disruptor (agent 0, the only channel affecting reward) gets the full
`adv_r - lam*adv_c`; Concealment (agent 1, zero causal path to reward, purely judged by the
stealth cost) gets ONLY `-adv_c`, unweighted by lambda -- lambda represents a real reward-vs-
cost trade-off for Disruptor, but not for a single-objective agent like Concealment (this
project's own earlier-established "lambda确实不负责权衡" finding).

Three variants share this one class, switched by `algo_args["algo"]["constraint_mode"]`:
- "illu" (2026-09-18, added after tracing the ORIGINAL Illu-Attacks-Jax reference
          implementation's actual mechanism): lam is a FIXED constant (`lambda_init`), never
          updated during training -- exactly matches `train_adversary.py`'s own
          `reward = -victim_reward + lambda_illu * illu_reward` (lambda_illu swept OFFLINE via
          hyperparameter search, no dual-ascent, no eps_cost/constraint target at all). This is
          NOT a constraint of any kind (soft or hard) -- it's plain fixed-weight scalarization.
- "soft":  lam is a dual-ascent variable, smoothly adapted each update
           (lam <- clip(lam + alpha_lambda * (mean_cost - eps_cost), 0, lambda_max)), where
           `mean_cost` is the rolling average of PER-EPISODE stealth-cost TOTALS (not a flat
           per-step average -- see `_stealth_ep_history` / class docstring below for why that
           distinction matters). This is "SC-MARL", a genuine Lagrangian-relaxation constraint.
- "hard":  lam is a bang-bang switch: lambda_max if mean_cost > eps_cost else 0.
           This is the "standard" hard-constraint baseline.

Stealth-cost source is normally the env's per-step `infos[i]["cost"]` (see `SafetyMARLEnv`'s
"wm"/"pedm_or" modes). When `env_args["stealth_source"] == "traj_critic"` (2026-09-18, per
explicit user request "跑一个不拟合pedm的，而是自己trajectory的" -- a detector-INDEPENDENT
alternative to both the frozen world-model and the frozen PEDM detector), this runner instead
hosts a `TrajectoryCritic` (GRU discriminator, reused verbatim from `examples/train_sc_marl.py`,
this project's own established design from before the HARL-native pivot): a binary classifier
adversarially co-trained EVERY update to distinguish CLEAN reference (o_rep/scale, a_v)
episode sequences from this update's freshly-completed ATTACKED ones. The env contributes
ZERO per-step cost in this mode (always 0) and just exposes each step's `(o_rep_scaled, a_v)`
via info; THIS runner assembles per-thread episode sequences from that info stream, and once
an episode completes, patches its terminal `-w_psi` (negative discriminator logit on that
episode, i.e. "how clean-like") into the cost-critic buffer's placeholder-0 slot at the exact
(step, thread) index where the episode ended -- a one-shot terminal cost, matching this
project's established terminal-cost + gamma_c=1 convention for trajectory-level signals.
REQUIRES `episode_length == env's max_cycles` so every rollout window aligns with exactly one
full episode per thread (this project's own established simplifying convention, see
`train_sc_mappo_safety.py`'s `traj_max_len` docstring) -- asserted in `__init__`.
"""
from collections import deque

import numpy as np
import torch
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.algorithms.critics.v_critic import VCritic
from harl.utils.trans_tools import _t2n


class OnPolicyLagrRunner(OnPolicyMARunner):
    """MAPPO with a second cost critic and a soft/hard constraint on the advantage."""

    def __init__(self, args, algo_args, env_args):
        # 2026-09-19: the base runner's __init__ calls self.restore() (if train.model_dir is
        # set) BEFORE this subclass has created self.cost_critic below -- restore() is
        # overridden here to also load the cost critic, so calling it that early crashes with
        # AttributeError. Defer: temporarily hide model_dir from the base class, build
        # everything (including cost_critic), then call the full restore() ourselves once
        # every attribute it needs actually exists.
        model_dir = algo_args["train"]["model_dir"]
        algo_args["train"]["model_dir"] = None
        super().__init__(args, algo_args, env_args)
        algo_args["train"]["model_dir"] = model_dir

        lagr_cfg = algo_args["algo"]
        self.constraint_mode = lagr_cfg.get("constraint_mode", "soft")
        self.eps_cost = float(lagr_cfg.get("eps_cost", 0.1))
        self.gamma_c = float(lagr_cfg.get("gamma_c", 1.0))
        self.alpha_lambda = float(lagr_cfg.get("alpha_lambda", 0.05))
        self.lambda_max = float(lagr_cfg.get("lambda_max", 20.0))
        self.lam = float(lagr_cfg.get("lambda_init", 0.0))

        share_observation_space = self.envs.share_observation_space[0]
        cost_critic_args = {**algo_args["model"], **algo_args["algo"]}
        cost_critic_args["gamma"] = self.gamma_c
        self.cost_critic = VCritic(cost_critic_args, share_observation_space, device=self.device)
        self.cost_critic_buffer = OnPolicyCriticBufferEP(
            {**algo_args["train"], **algo_args["model"], **algo_args["algo"], "gamma": self.gamma_c},
            share_observation_space,
        )

        n_threads = algo_args["train"]["n_rollout_threads"]
        self._true_cost_window = []  # per-step true_cost, cleared every train() call
        self._true_cost_ep_sum = np.zeros(n_threads, dtype=np.float32)  # running per-thread total
        self._true_cost_completed = []  # completed episodes' total true_cost this window

        # Per-EPISODE stealth-cost totals (2026-09-18, replaces a flat per-step average that
        # diluted rare-but-severe spikes: one bad step among ~1200 per-step samples barely moved
        # a flat mean, so lambda's dual-ascent update barely reacted no matter how badly that one
        # step would fail the real OR/peak-based detection criterion. Averaging PER-EPISODE totals
        # instead means a spike shows up fully in ITS OWN episode's total, and only THEN gets
        # averaged across several episodes -- the same "Monte-Carlo estimate of an expectation"
        # averaging every Lagrangian/RCPO-style method needs, just done over the right unit.)
        self._stealth_cost_ep_sum = np.zeros(n_threads, dtype=np.float32)
        # 2026-09-21 bug fix (user-caught, "会不会是hopper实际上meancost要小于10...你和hc一样设置
        # 成100了" -- diagnosed further to a variable-episode-length issue, not just a wrong
        # number): for envs whose episodes can terminate EARLY (Hopper/Ant fall over; HC/
        # SafetyPointGoal1 never do), the per-episode SUM this class was built around is not a
        # portable quantity -- a more aggressively-attacked Hopper episode ends sooner (verified
        # directly: attacked episode_length=97, not 1000), so its cost SUM shrinks even though
        # the per-step anomaly rate is unchanged or worse, letting `eps_cost` (calibrated
        # assuming ~1000-step episodes) become far too loose without lambda ever reacting --
        # confirmed empirically: Hopper's SC-MARL run converged mean_cost to ~0.35-0.40 (SUM
        # over ~97 steps) against `eps_cost=100`, lambda relaxed to 0 almost immediately, yet the
        # per-step cost rate (0.0036-0.0041) was never actually driven down by the constraint --
        # it just never had to be, since the SUM already looked tiny purely from the short
        # episode. New optional `cost_aggregation` ("sum", the default, unchanged behavior for
        # every existing safety_marl calibration; or "mean", dividing by the completed episode's
        # own step count) makes `eps_cost` a genuine PER-STEP target regardless of how long an
        # episode actually runs -- portable across environments with different/variable episode
        # lengths, unlike the SUM convention's implicit "assume ~1000 steps" dependency.
        self.cost_aggregation = str(lagr_cfg.get("cost_aggregation", "sum"))
        assert self.cost_aggregation in ("sum", "mean")
        self._stealth_cost_ep_steps = np.zeros(n_threads, dtype=np.float32)
        # 2026-09-20, user-request ("50个好像有点多了...以后改成10个"): window length is now
        # configurable (`mean_cost_window`, default 10, was hardcoded 50) -- a shorter window
        # makes mean_cost react faster to recent episodes instead of being dragged out by a
        # spike still sitting in a long rolling history (see the 2026-09-19 lag discussion: a
        # true_cost spike had already visibly declined for several updates while mean_cost kept
        # climbing, because the spike episodes were still inside the window).
        self.mean_cost_window = int(lagr_cfg.get("mean_cost_window", 10))
        self._stealth_ep_history = deque(maxlen=self.mean_cost_window)  # rolling window across
        # MULTIPLE updates, since one 200-step rollout window rarely contains a full 1000-step
        # episode boundary
        # 2026-09-19 bug fix (user-caught): UNCAPPED list, alongside the maxlen deque above.
        # The alternating runner needs "exactly which episodes completed during THIS SPECIFIC
        # rollout call" (for C_post) -- it used to snapshot len(_stealth_ep_history) before/after
        # and slice the difference, but once total completions exceed the deque's maxlen (happens
        # within the first few outer cycles at 6 threads x ~1 completion/thread/rollout), len()
        # stays PINNED at maxlen forever (the deque only evicts-and-appends, never grows past it)
        # -- so before==after ALWAYS from that point on, making the "new completions" slice
        # permanently EMPTY. C_post's window then silently stops receiving any new data almost
        # immediately after training starts. This list is reset to [] by whoever wants a clean
        # "since last mark" read (see OnPolicyAlternatingLagrRunner._collect_rollout) and is
        # never capped, so it can't suffer the same saturation.
        self._stealth_ep_recent = []

        # 2026-09-19: an intervention-cost regularizer (penalizing Concealment's AND Disruptor's
        # own raw action magnitude) was implemented and then reverted here per explicit user
        # redirect ("不要加正则化了没意义") in favor of an ATLA-inspired fix that targets the
        # DISTRIBUTION mechanics directly: env-side tanh-squash (replacing hard clip) + a smaller
        # log_std init/clamp range, applied right below. Both agents were independently confirmed
        # (via direct actor-network inspection) to have drifted their raw Gaussian mean far
        # outside their own clip window, causing near-100% clip-saturation into a
        # state-independent, non-adaptive constant action.
        self._act_dim = 2
        self._full_dim = self.envs.action_space[1].shape[0] - self._act_dim
        # ATLA-inspired std control: keep the raw (pre-squash) output naturally within a range
        # where tanh's gradient is still meaningful.
        self._hidden_log_std_init = float(lagr_cfg.get("hidden_log_std_init", -1.0))
        self._hidden_log_std_min = float(lagr_cfg.get("hidden_log_std_min", -3.0))
        self._hidden_log_std_max = float(lagr_cfg.get("hidden_log_std_max", 0.0))
        # 2026-09-20, reverted per user diagnosis: entropy_coef was ALSO force-set to 0.0 for
        # both agents here, on the theory that entropy's upward pressure on std would fight the
        # log_std clamp. But v36/v37 (both agents at entropy_coef=0, one dual-ascent, one fixed
        # lambda) showed the IDENTICAL "attacks once, then reverts to near-total passivity"
        # pattern regardless of constraint_mode -- since illu-mode's fixed lambda never depends
        # on mean_cost at all, that pattern can't be explained by the lambda/window dynamics, but
        # DOES apply equally to a shared entropy_coef=0 change: with zero entropy bonus, once a
        # single PPO update reacts to one costly attack and pulls the mean back, nothing pushes
        # exploration back toward attacking again. The log_std clamp below already provides an
        # INDEPENDENT safety net against std runaway, so entropy_coef no longer needs to be
        # zeroed for that purpose -- restored to the project's normal per-agent value (whatever
        # algo_args["algo"]["entropy_coef"] already is, currently 0.01) by simply NOT overriding
        # it here anymore.
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                self.actor[agent_id].actor.act.action_out.log_std.fill_(self._hidden_log_std_init)

        self.traj_mode = env_args.get("stealth_source") == "traj_critic"
        if self.traj_mode:
            self._init_traj_critic(env_args, algo_args)

        if model_dir is not None:
            self.restore()

    def _init_traj_critic(self, env_args, algo_args):
        from examples.train_sc_marl import TrajectoryCritic, update_trajectory_critic
        from examples.train_sc_mappo_safety import collect_clean_reference
        from examples.train_safety_attacker import SafetyVictim

        episode_length = algo_args["train"]["episode_length"]
        max_cycles = int(env_args.get("max_cycles", 1000))
        assert episode_length == max_cycles, (
            f"traj_critic mode requires episode_length ({episode_length}) == env max_cycles "
            f"({max_cycles}) so every rollout window aligns with exactly one full episode per "
            f"thread -- pass --episode_length {max_cycles}"
        )
        self._update_trajectory_critic = update_trajectory_critic

        victim = SafetyVictim(env_args["victim_ckpt"], "cpu")
        full_dim = victim.full_obs_dim
        act_dim = 2
        scale_full = np.maximum(np.sqrt(victim.obs_norm.var[:full_dim]).astype(np.float32), 0.01)

        traj_hidden = int(algo_args["algo"].get("traj_hidden", 128))
        traj_lr = float(algo_args["algo"].get("traj_lr", 1e-3))
        self.traj_epoch = int(algo_args["algo"].get("traj_epoch", 4))
        n_clean_ref = int(algo_args["algo"].get("n_clean_ref", 64))

        self.traj_critic = TrajectoryCritic(full_dim, act_dim, hidden=traj_hidden).to(self.device)
        self.traj_opt = torch.optim.Adam(self.traj_critic.parameters(), lr=traj_lr)
        print(f"[traj_critic] collecting {n_clean_ref} clean reference episodes...")
        self.clean_seq, self.clean_len = collect_clean_reference(
            victim, env_args.get("scenario", "SafetyPointGoal1Gymnasium-v0"),
            full_dim, act_dim, scale_full, n_clean_ref, episode_length, self.device,
        )

        n_threads = algo_args["train"]["n_rollout_threads"]
        self.ep_seq_buffers = [[] for _ in range(n_threads)]
        self.pending_episodes = []  # list of (step, thread, seq_array, length)
        self.last_ckl = float("nan")

    def warmup(self):
        super().warmup()
        self.cost_critic_buffer.share_obs[0] = self.critic_buffer.share_obs[0].copy()

    @torch.no_grad()
    def collect(self, step):
        values, actions, action_log_probs, rnn_states, rnn_states_critic = super().collect(step)
        cost_value, cost_rnn_state_critic = self.cost_critic.get_values(
            self.cost_critic_buffer.share_obs[step],
            self.cost_critic_buffer.rnn_states_critic[step],
            self.cost_critic_buffer.masks[step],
        )
        self._cost_values = _t2n(cost_value)
        self._cost_rnn_states_critic = _t2n(cost_rnn_state_critic)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        super().insert(data)
        (
            obs, share_obs, rewards, dones, infos, available_actions,
            values, actions, action_log_probs, rnn_states, rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)
        cost_rnn_states_critic = self._cost_rnn_states_critic
        cost_rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.algo_args["train"]["n_rollout_threads"], 1), dtype=np.float32)
        masks[dones_env == True] = 0.0
        bad_masks = np.array(
            [[0.0] if info[0].get("bad_transition", False) else [1.0] for info in infos]
        )
        costs = np.array([[info[0].get("cost", 0.0)] for info in infos], dtype=np.float32)
        true_costs = np.array([info[0].get("true_cost", 0.0) for info in infos], dtype=np.float32)
        self._true_cost_window.append(true_costs)
        self._true_cost_ep_sum += true_costs
        stealth_costs_flat = costs[:, 0]
        self._stealth_cost_ep_sum += stealth_costs_flat
        self._stealth_cost_ep_steps += 1.0
        for i in range(len(true_costs)):
            if dones_env[i]:
                self._true_cost_completed.append(float(self._true_cost_ep_sum[i]))
                self._true_cost_ep_sum[i] = 0.0
                if self.cost_aggregation == "mean":
                    ep_cost = float(self._stealth_cost_ep_sum[i]) / max(1.0, float(self._stealth_cost_ep_steps[i]))
                else:
                    ep_cost = float(self._stealth_cost_ep_sum[i])
                self._stealth_ep_history.append(ep_cost)
                self._stealth_ep_recent.append(ep_cost)
                self._stealth_cost_ep_sum[i] = 0.0
                self._stealth_cost_ep_steps[i] = 0.0

        cur_step = self.cost_critic_buffer.step  # slot this insert() call is about to write
        self.cost_critic_buffer.insert(
            share_obs[:, 0], cost_rnn_states_critic, self._cost_values, costs, masks, bad_masks
        )

        if self.traj_mode:
            for i, info in enumerate(infos):
                # 2026-09-19: audit Concealment's own forged action (a_rep), never the honest
                # a_v -- see safety_marl_env.py's module docstring for why pairing a fake obs
                # with the honest action left a structural hole.
                self.ep_seq_buffers[i].append(
                    np.concatenate([info[0]["o_rep_scaled"], info[0]["a_rep"]]).astype(np.float32)
                )
                if dones_env[i]:
                    seq = np.array(self.ep_seq_buffers[i], np.float32)
                    self.pending_episodes.append((cur_step, i, seq))
                    self.ep_seq_buffers[i] = []

    @torch.no_grad()
    def compute(self):
        super().compute()
        next_cost_value, _ = self.cost_critic.get_values(
            self.cost_critic_buffer.share_obs[-1],
            self.cost_critic_buffer.rnn_states_critic[-1],
            self.cost_critic_buffer.masks[-1],
        )
        next_cost_value = _t2n(next_cost_value)
        if self.traj_mode:
            # Defer compute_returns(): the terminal costs for this window's just-completed
            # episodes haven't been patched in yet (that needs a gradient-enabled
            # TrajectoryCritic update, which can't run inside this no_grad-decorated method) --
            # see train()/_run_traj_critic_update(), which patches costs THEN calls this.
            self._next_cost_value = next_cost_value
        else:
            self.cost_critic_buffer.compute_returns(next_cost_value, None)

    def _run_traj_critic_update(self):
        """Score this update's completed episodes against the co-trained TrajectoryCritic,
        patch each episode's terminal `-w_psi` into its (step, thread) slot in the cost
        buffer (replacing the placeholder 0 every env step wrote), and return the mean
        terminal cost over just these completed episodes (for the dual-ascent step -- the
        full per-step buffer average would be diluted ~1000x by the zero-filled non-terminal
        steps, unlike a genuinely dense signal)."""
        if not self.pending_episodes:
            return None
        max_len = self.algo_args["train"]["episode_length"]
        feat_dim = self.pending_episodes[0][2].shape[-1]
        atk_seqs = np.zeros((len(self.pending_episodes), max_len, feat_dim), np.float32)
        atk_lens = np.zeros(len(self.pending_episodes), np.int64)
        for k, (_, _, seq) in enumerate(self.pending_episodes):
            L = min(len(seq), max_len)
            atk_seqs[k, :L] = seq[:L]
            atk_lens[k] = L
        atk_seq_t = torch.as_tensor(atk_seqs, device=self.device)
        atk_len_t = torch.as_tensor(atk_lens, device=self.device)

        n_sample = min(len(self.pending_episodes), self.clean_seq.shape[0])
        clean_idx = np.random.choice(self.clean_seq.shape[0], n_sample, replace=False)
        ckl, w_psi = self._update_trajectory_critic(
            self.traj_critic, self.traj_opt,
            self.clean_seq[clean_idx], self.clean_len[clean_idx],
            atk_seq_t, atk_len_t, epochs=self.traj_epoch,
        )
        self.last_ckl = ckl

        terminal_costs = []
        for k, (step, thread, _) in enumerate(self.pending_episodes):
            cost = float(-w_psi[k])
            self.cost_critic_buffer.rewards[step, thread, 0] = cost
            terminal_costs.append(cost)
        self.pending_episodes = []
        return float(np.mean(terminal_costs))

    def train(self):
        mean_cost_override = None
        if self.traj_mode:
            mean_cost_override = self._run_traj_critic_update()
            self.cost_critic_buffer.compute_returns(self._next_cost_value, None)

        # 2026-09-19 bug fix (user-caught): reward critic uses ValueNorm (use_valuenorm=True),
        # so critic_buffer.value_preds is stored in NORMALIZED space while returns is already
        # denormalized (see OnPolicyCriticBufferEP.compute_returns). The unmodified HARL
        # reference (on_policy_ma_runner.py's train()) denormalizes value_preds before
        # subtracting; this custom override had silently skipped that branch since v14,
        # computing RAW_returns - NORMALIZED_value_preds -- a real scale mismatch, not just a
        # style difference from the reference. Cost critic has no value_normalizer (trained
        # with None), so its own value_preds are already raw -- no denormalization needed there.
        if self.value_normalizer is not None:
            adv_r = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1]
            )
        else:
            adv_r = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
        adv_c = self.cost_critic_buffer.returns[:-1] - self.cost_critic_buffer.value_preds[:-1]

        # `combine_then_normalize` (2026-09-19, per user's isolated ablation request -- "先隔离
        # 归一化问题"): if True, combine adv_r/adv_c RAW (adv_r - lam*adv_c) and normalize the
        # COMBINED result ONCE; if False (default, original behavior), normalize adv_r and adv_c
        # to unit variance SEPARATELY before combining. The separate-normalization scheme means
        # lambda's effective weight silently drifts every update as each stream's OWN batch
        # variance changes -- even a constant lambda can correspond to a different effective
        # penalty from one update to the next, which is a scale-mismatch relative to lambda
        # being dual-ascended against RAW (unnormalized) per-episode cost. combine_then_normalize
        # keeps lambda's numeric value tied to a stable meaning in the raw cost scale throughout
        # training. This flag exists so the normalization fix can be evaluated in ISOLATION
        # (synchronous updates unchanged) before/separately from the alternating-schedule runner.
        combine_then_normalize = bool(self.algo_args["algo"].get("combine_then_normalize", False))
        if not combine_then_normalize:
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-5)
            adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-5)

        # traj_critic mode: the buffer's own per-step mean is diluted ~episode_length-to-1 by
        # the zero-filled non-terminal steps (only one real value per thread per window) -- use
        # the mean over just this window's completed episodes' terminal costs instead. For
        # "soft"/"hard", use the rolling per-EPISODE-total average (self._stealth_ep_history)
        # instead of a flat per-step average over the buffer -- averaging per-episode totals
        # means a single severe spike shows up fully in its own episode's total before any
        # cross-episode averaging happens, rather than being diluted 1-in-~1200 against calm
        # per-step samples from the same episode (see class docstring / memory).
        if mean_cost_override is not None:
            mean_cost = mean_cost_override
        elif self._stealth_ep_history:
            mean_cost = float(np.mean(self._stealth_ep_history))
        else:
            mean_cost = self.cost_critic_buffer.get_mean_rewards()  # no episode completed yet anywhere

        # freeze_lambda (2026-09-19, "hidden catch-up test" diagnostic: resume from a checkpoint
        # via model_dir, hold lambda fixed at whatever value was restored, and see whether
        # Concealment alone -- with the Disruptor and the constraint target both frozen -- can
        # still drive the stealth cost down further. If it can, the synchronous baseline's
        # Concealment was lagging behind a moving Disruptor, not at its own optimum.)
        freeze_lambda = bool(self.algo_args["algo"].get("freeze_lambda", False))
        if freeze_lambda:
            pass
        elif self.constraint_mode == "illu":
            # Fixed lambda (2026-09-18, matches Illu-Attacks-Jax's own `train_adversary.py`
            # convention exactly: `lambda_illu` is a constant hyperparameter, swept OFFLINE via
            # search, never dual-ascended during training -- no eps_cost/constraint target
            # exists in that method at all). self.lam stays at whatever `lambda_init` was set to.
            pass
        elif self.constraint_mode == "hard":
            self.lam = self.lambda_max if mean_cost > self.eps_cost else 0.0
        else:  # soft
            self.lam = float(
                np.clip(
                    self.lam + self.alpha_lambda * (mean_cost - self.eps_cost),
                    0.0,
                    self.lambda_max,
                )
            )

        # Role-split (2026-09-18 standing instruction) cleanly separates responsibility: under
        # `a_v = victim.act_from_augmented(obs_aug)` (clean), Disruptor's action is the ONLY
        # channel affecting reward, and has zero direct bearing on the (o_rep,a_v) record the
        # stealth cost scores -- so it gets the full constrained objective. Concealment has zero
        # causal path to reward at all, so giving it any adv_r term would be spurious credit;
        # per this project's own earlier-established finding ("lambda确实不负责权衡" for a
        # single-objective agent), it gets ONLY the (unweighted, un-lambda'd) stealth advantage.
        #
        # 2026-09-19: an intervention-cost regularizer (penalizing raw action magnitude) was
        # tried and reverted here per explicit user redirect ("不要加正则化了没意义") -- see
        # `_apply_saturation_fix()` / the ATLA-inspired tanh-squash + std-control approach
        # instead (env-side squash + log_std init/clamp + per-agent entropy_coef=0), which
        # targets the DISTRIBUTION mechanics directly rather than adding a competing objective
        # term. See module-level comment / memory for the full before/after reasoning.
        adv_disruptor = adv_r - self.lam * adv_c
        adv_concealment = -adv_c
        if combine_then_normalize:
            adv_disruptor = (adv_disruptor - adv_disruptor.mean()) / (adv_disruptor.std() + 1e-5)
            adv_concealment = (adv_concealment - adv_concealment.mean()) / (adv_concealment.std() + 1e-5)
        per_agent_adv = [adv_disruptor, adv_concealment]

        # freeze_disruptor (2026-09-19, "hidden catch-up test" diagnostic companion to
        # freeze_lambda): skip agent 0's (Disruptor's) actor update entirely, so a checkpoint's
        # performance policy stays exactly fixed while only Concealment (agent 1) keeps adapting.
        freeze_disruptor = bool(self.algo_args["algo"].get("freeze_disruptor", False))
        actor_train_infos = []
        for agent_id in range(self.num_agents):
            if freeze_disruptor and agent_id == 0:
                actor_train_infos.append({})
                continue
            actor_train_info = self.actor[agent_id].train(
                self.actor_buffer[agent_id], per_agent_adv[agent_id].copy(), "EP"
            )
            actor_train_infos.append(actor_train_info)

        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        cost_critic_train_info = self.cost_critic.train(self.cost_critic_buffer, None)
        critic_train_info["cost_value_loss"] = cost_critic_train_info["value_loss"]
        critic_train_info["mean_cost"] = mean_cost
        critic_train_info["lambda"] = self.lam
        if self.traj_mode:
            critic_train_info["traj_critic_ckl"] = self.last_ckl

        # Real safety_gym hazard cost -- NOT the stealth-cost signal above. Tracks whether a
        # genuine physical attack is happening at all, independent of what's being optimized
        # against; per-step mean is always available, per-episode mean only when >=1 episode
        # completed within this window (episode_length may not align to the env's own episode
        # boundary outside traj_critic mode, so a window can complete zero, one, or several).
        critic_train_info["true_cost_per_step"] = float(np.mean(self._true_cost_window))
        if self._true_cost_completed:
            critic_train_info["true_cost_per_episode"] = float(np.mean(self._true_cost_completed))
        self._true_cost_window = []
        self._true_cost_completed = []

        # ATLA-inspired std control (2026-09-19): clamp log_std back into range after every
        # gradient step -- gradient descent could otherwise push it back out (e.g. entropy
        # pressure from OTHER loss terms, or just drift), undoing the fix over time. Also log
        # both agents' current raw std so the fix's effect is directly observable in training,
        # not just inferred after the fact from a one-off checkpoint inspection.
        for agent_id in range(self.num_agents):
            log_std_param = self.actor[agent_id].actor.act.action_out.log_std
            with torch.no_grad():
                log_std_param.clamp_(self._hidden_log_std_min, self._hidden_log_std_max)
            std_x_coef = self.actor[agent_id].actor.act.action_out.std_x_coef
            std_y_coef = self.actor[agent_id].actor.act.action_out.std_y_coef
            actual_std = torch.sigmoid(log_std_param / std_x_coef) * std_y_coef
            critic_train_info[f"agent{agent_id}_mean_std"] = float(actual_std.mean().item())

        return actor_train_infos, critic_train_info

    def after_update(self):
        super().after_update()
        self.cost_critic_buffer.after_update()

    def prep_rollout(self):
        super().prep_rollout()
        self.cost_critic.prep_rollout()

    def prep_training(self):
        super().prep_training()
        self.cost_critic.prep_training()

    def save(self):
        super().save()
        torch.save(
            self.cost_critic.critic.state_dict(),
            str(self.save_dir) + "/cost_critic_agent.pt",
        )
        if self.traj_mode:
            torch.save(
                self.traj_critic.state_dict(),
                str(self.save_dir) + "/traj_critic.pt",
            )

    def restore(self):
        super().restore()
        cost_critic_state_dict = torch.load(
            str(self.algo_args["train"]["model_dir"]) + "/cost_critic_agent.pt"
        )
        self.cost_critic.critic.load_state_dict(cost_critic_state_dict)
        if self.traj_mode:
            traj_critic_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"]) + "/traj_critic.pt"
            )
            self.traj_critic.load_state_dict(traj_critic_state_dict)
