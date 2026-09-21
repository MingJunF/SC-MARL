"""Alternating SC-MAPPO (2026-09-19), per the user's own detailed design spec responding to
the diagnosed instability of the SYNCHRONOUS `OnPolicyLagrRunner` ("SC-MAPPO baseline", kept
unmodified -- registered as `mappo_lagr`/`mappo_hard`/`mappo_illu` for comparison, per explicit
instruction "旧算法应完整保留，作为同步MAPPO-Lagrangian baseline").

Diagnosis this addresses: synchronously updating both attackers every cycle means the JOINT
policy distribution keeps moving under both agents at once, so (a) lambda's dual-ascent chases
a constantly-shifting cost distribution instead of converging, and (b) Concealment is
perpetually adapting to a Disruptor that changed AGAIN before it caught up. Observed empirically
in `harl_native_v17_scmarl_loose`: lambda drifted continuously downward for >1M steps without
settling, with a prior low-lambda regime (~2.2) immediately preceding a true_cost blowup to
200+ -- consistent with (not proof of) a longer-period oscillation driven by this
policy-induced non-stationarity, not just noisy dual-ascent.

Design (outer cycle, repeated):
  1. HIDDEN ADAPTATION: freeze the performance (Disruptor) policy, freeze lambda, freeze the
     cost estimator (TrajectoryCritic, if in traj_mode). Run `K_h` FRESH rollout-collect-update
     cycles of ONLY the hidden (Concealment) actor, using `A_hidden = -adv_c` (already this
     project's existing per-agent formula -- unchanged). K_h default 2-3: representing multiple
     fresh on-policy rollout-update cycles (not repeated PPO epochs over stale data, which would
     just re-optimize against an already-seen trajectory distribution rather than genuinely
     adapting to what the CURRENT performance policy is doing).
  2. POST-ADAPTATION ROLLOUT: with the now-adapted hidden policy and the STILL-frozen
     performance policy, collect one more fresh rollout. This trajectory's stealth cost
     (`C_post`) is what lambda should observe -- not a blend with the hidden-adaptation-phase's
     own (still-adapting, not yet representative) rollouts.
  3. PERFORMANCE UPDATE: freeze the (now-adapted) hidden policy, update Disruptor ONCE on the
     post-adaptation rollout's data using `A_perf = adv_r - lam*adv_c` (combined RAW, THEN
     normalized ONCE -- see below).
  4. SLOW DUAL UPDATE: every `M` (`lambda_update_period`) outer cycles (not every cycle),
     update lambda from the plain mean of the last `M` cycles' own `C_post` values (RAW
     per-episode-total cost, never a normalized advantage) -- no separate EMA coefficient
     (2026-09-19, simplified per user request: `M` alone controls both the dual-ascent
     frequency and the cost-averaging window, one fewer hyperparameter to search over):
     `lam <- clip(lam + alpha_lambda*(mean(last M C_post values) - eps_cost), 0, lambda_max)`.

Advantage normalization fix: the baseline runner separately normalizes adv_r and adv_c to unit
variance EVERY update before combining, which means lambda's effective weight silently drifts
as each stream's own batch variance changes -- even a CONSTANT lambda could correspond to a
very different effective penalty from one update to the next. Here, Disruptor's advantage is
combined RAW first (`adv_r - lam*adv_c`, both un-normalized GAE advantages) and normalized ONCE
as a whole, so lambda's numeric value keeps a stable meaning relative to the RAW cost scale
throughout training (per the user's "方案B" recommendation, chosen for simplicity). Concealment's
own advantage (`-adv_c`, a single term, no lambda) is normalized on its own as usual -- no
drift risk there since there's nothing for it to drift relative to.

This subclasses `OnPolicyLagrRunner` and reuses ALL of its collection/buffer/cost-critic/
traj_critic machinery unchanged -- only `run()` (the outer training loop) and the update
mechanics (`_train_selected`, lambda's own update rule) are new.
"""
from collections import deque

import numpy as np
import torch

from harl.runners.on_policy_lagr_runner import OnPolicyLagrRunner


class OnPolicyAlternatingLagrRunner(OnPolicyLagrRunner):
    """Alternating (hidden-adapts-first, then performance-updates-once) SC-MAPPO."""

    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        # 2026-09-19 bug fix (user-caught, P0): a rollout window SHORTER than the env's own
        # episode (max_cycles) means an episode completing during the post-adaptation rollout
        # may have STARTED many rollout-windows (even outer-cycles) earlier, under a MIX of
        # older hidden-policy (and possibly older performance-policy) versions -- so its cost
        # total is not actually "the cost of (frozen performance, fully-adapted hidden)" at all,
        # contaminating C_post's entire meaning. Require exact alignment (same convention
        # already enforced for traj_critic mode in the base OnPolicyLagrRunner) so every
        # `_collect_rollout()` call is exactly one full episode per thread, generated ENTIRELY
        # under whichever single (performance, hidden) policy pair was frozen/adapting that
        # phase -- no cross-phase or cross-cycle policy-version mixing within one episode.
        episode_length = algo_args["train"]["episode_length"]
        max_cycles = int(env_args.get("max_cycles", 1000))
        assert episode_length == max_cycles, (
            f"Alternating SC-MAPPO requires episode_length ({episode_length}) == env max_cycles "
            f"({max_cycles}) so every rollout is exactly one full episode per thread, generated "
            f"entirely under one frozen (performance, hidden) policy pair -- otherwise C_post is "
            f"contaminated by episodes that started under older policy versions. Pass "
            f"--episode_length {max_cycles}."
        )
        alt_cfg = algo_args["algo"]
        self.K_h = int(alt_cfg.get("k_hidden", 2))
        # 2026-09-20, per explicit user pushback ("Disruptor更新才1次相比conceal5次？感觉有点少了，
        # 至少5/2把" -- Disruptor getting exactly ONE update per outer cycle vs Concealment's K_h
        # is too lopsided): number of (post-adaptation rollout, one performance update) rounds
        # per outer cycle, default 1 (the original design) -- set >1 to let Disruptor catch up
        # more before the next hidden-adaptation phase freezes it again.
        self.K_p = int(alt_cfg.get("k_perf", 1))
        # 2026-09-19, simplified per user request (drop the extra ema_coef hyperparameter):
        # lambda updates every `lambda_update_period` outer cycles from the plain mean of the
        # last `lambda_update_period` cycles' own post-adaptation costs -- `M` alone decides
        # both the dual-ascent frequency AND the cost-averaging window, no separate EMA needed.
        self.lambda_update_period = int(alt_cfg.get("lambda_update_period", 5))
        self._c_post_window = deque(maxlen=self.lambda_update_period)
        self._outer_cycle_count = 0

    def _collect_rollout(self):
        """One full rollout window (episode_length steps): collect+step+insert, then compute()
        returns/advantages. Returns the list of per-episode stealth-cost totals that completed
        DURING this specific call, for C_post.

        2026-09-19 bug fix (user-caught): previously snapshotted len(_stealth_ep_history)
        before/after and sliced the difference -- but that deque has maxlen=50, so once total
        completions exceed 50 (happens within the first few outer cycles), len() stays pinned at
        50 forever and the "new completions" slice is permanently empty. Use the UNCAPPED
        `_stealth_ep_recent` list instead: clear it right before collecting, read it right after
        -- immune to any maxlen saturation regardless of how many episodes complete."""
        self.prep_rollout()
        self._stealth_ep_recent = []
        for step in range(self.algo_args["train"]["episode_length"]):
            data = self.collect(step)
            values, actions, action_log_probs, rnn_states, rnn_states_critic = data
            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
            full_data = (
                obs, share_obs, rewards, dones, infos, available_actions,
                values, actions, action_log_probs, rnn_states, rnn_states_critic,
            )
            self.logger.per_step(full_data)
            self.insert(full_data)
        self.compute()
        completed_this_rollout = list(self._stealth_ep_recent)
        return completed_this_rollout

    def _combined_advantages(self):
        """RAW (unnormalized, but correctly DEnormalized where ValueNorm applies) adv_r, adv_c --
        role-specific normalization happens AFTER combining, in _train_selected(). 2026-09-19
        bug fix (user-caught): reward critic uses ValueNorm, so value_preds lives in normalized
        space while returns is already raw -- must denormalize value_preds before subtracting,
        matching the unmodified HARL reference (on_policy_ma_runner.py's train()) and the same
        fix applied to OnPolicyLagrRunner.train()."""
        if self.value_normalizer is not None:
            adv_r = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1]
            )
        else:
            adv_r = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
        adv_c = self.cost_critic_buffer.returns[:-1] - self.cost_critic_buffer.value_preds[:-1]
        return adv_r, adv_c

    @staticmethod
    def _normalize(x):
        return (x - x.mean()) / (x.std() + 1e-5)

    def _train_selected(self, train_disruptor, train_concealment):
        """Train the value critics as usual (fresh data always benefits both), plus whichever
        actor(s) are enabled this phase. Returns (actor_train_infos, critic_train_info) in the
        same shape `OnPolicyBaseRunner.run()`'s logging expects."""
        mean_cost_override = None
        if self.traj_mode:
            mean_cost_override = self._run_traj_critic_update()
            self.cost_critic_buffer.compute_returns(self._next_cost_value, None)

        adv_r, adv_c = self._combined_advantages()
        assert np.isfinite(adv_r).all(), "adv_r has non-finite values"
        assert np.isfinite(adv_c).all(), "adv_c has non-finite values"

        actor_train_infos = [{}, {}]
        if train_disruptor:
            assert np.isfinite(self.lam), f"self.lam is non-finite: {self.lam}"
            adv_perf_raw = adv_r - self.lam * adv_c  # combine RAW
            assert np.isfinite(adv_perf_raw).all(), "adv_perf_raw (pre-normalize) has non-finite values"
            adv_perf = self._normalize(adv_perf_raw)  # normalize ONCE
            assert np.isfinite(adv_perf).all(), "adv_perf (post-normalize) has non-finite values"
            actor_train_infos[0] = self.actor[0].train(self.actor_buffer[0], adv_perf.copy(), "EP")
        if train_concealment:
            adv_hidden = self._normalize(-adv_c)
            assert np.isfinite(adv_hidden).all(), "adv_hidden has non-finite values"
            actor_train_infos[1] = self.actor[1].train(self.actor_buffer[1], adv_hidden.copy(), "EP")

        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        cost_critic_train_info = self.cost_critic.train(self.cost_critic_buffer, None)
        critic_train_info["cost_value_loss"] = cost_critic_train_info["value_loss"]
        critic_train_info["lambda"] = self.lam

        if mean_cost_override is not None and self._stealth_ep_history:
            pass  # traj_mode's own per-window mean already folded into _stealth_ep_history via insert()
        critic_train_info["mean_cost"] = (
            float(np.mean(self._stealth_ep_history)) if self._stealth_ep_history else 0.0
        )
        critic_train_info["true_cost_per_step"] = float(np.mean(self._true_cost_window)) if self._true_cost_window else 0.0
        if self._true_cost_completed:
            critic_train_info["true_cost_per_episode"] = float(np.mean(self._true_cost_completed))
        self._true_cost_window = []
        self._true_cost_completed = []

        # Per user's suggestion: check every logged scalar is finite BEFORE it reaches
        # tensorboard, so a future non-finite value fails loudly with its exact tag/source
        # instead of surfacing only as tensorboardX's generic, tag-less "NaN or Inf found in
        # input tensor" warning.
        def _to_numpy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64)

        for key, value in critic_train_info.items():
            arr = _to_numpy(value)
            assert np.all(np.isfinite(arr)), f"critic_train_info[{key!r}] is non-finite: {value}"
        for agent_id, info in enumerate(actor_train_infos):
            for key, value in info.items():
                arr = _to_numpy(value)
                assert np.all(np.isfinite(arr)), (
                    f"actor_train_infos[{agent_id}][{key!r}] is non-finite: {value}"
                )

        # ATLA-inspired std control (2026-09-19, see OnPolicyLagrRunner.__init__/train() for the
        # full rationale): clamp log_std back into range after every gradient step, for whichever
        # actor(s) were actually trained this phase (entropy_coef=0 and the initial log_std fill
        # are already applied to both agents in the shared __init__).
        for agent_id, trained in [(0, train_disruptor), (1, train_concealment)]:
            if not trained:
                continue
            log_std_param = self.actor[agent_id].actor.act.action_out.log_std
            with torch.no_grad():
                log_std_param.clamp_(self._hidden_log_std_min, self._hidden_log_std_max)

        return actor_train_infos, critic_train_info

    def run(self):
        self.warmup()
        episode_length = self.algo_args["train"]["episode_length"]
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        steps_per_rollout = episode_length * n_threads
        steps_per_cycle = steps_per_rollout * (self.K_h + self.K_p)
        outer_cycles = int(self.algo_args["train"]["num_env_steps"]) // steps_per_cycle
        total_episodes = outer_cycles * (self.K_h + self.K_p)
        self.logger.init(total_episodes)

        episode_idx = 0
        for cycle in range(1, outer_cycles + 1):
            self._outer_cycle_count += 1

            # --- Phase 1: hidden adaptation, K_h fresh rollout-update cycles ---
            for _ in range(self.K_h):
                episode_idx += 1
                self.logger.episode_init(episode_idx)
                self._collect_rollout()
                self.prep_training()
                actor_train_infos, critic_train_info = self._train_selected(
                    train_disruptor=False, train_concealment=True
                )
                if episode_idx % self.algo_args["train"]["log_interval"] == 0:
                    self.logger.episode_log(
                        actor_train_infos, critic_train_info, self.actor_buffer, self.critic_buffer
                    )
                self.after_update()

            # --- Phase 2+3: K_p rounds of (post-adaptation rollout, ONE performance update).
            # 2026-09-20, per explicit user pushback ("Disruptor更新才1次相比conceal5次？感觉有点
            # 少了，至少5/2把" -- a single Disruptor update per cycle vs K_h Concealment updates
            # is too lopsided, Disruptor needs more than one shot to actually catch up before
            # being frozen again): K_p defaults to 1 (the original design) but can be raised so
            # Disruptor gets multiple fresh-rollout updates per cycle too, still always AFTER
            # Concealment has fully finished adapting for this cycle (so each of these rounds'
            # rollout is still against a frozen, no-longer-moving Concealment target -- only the
            # COUNT of Disruptor's own updates changes, not the alternation structure itself).
            # Only the LAST round's actor_train_infos/critic_train_info get logged (below,
            # after Phase 4 adds c_post/lambda to them) -- matches the original design's single
            # end-of-cycle log call; intermediate rounds (when K_p>1) update the policy but
            # don't get their own separate tensorboard point, avoiding a duplicate write at the
            # same step for the last round.
            c_post_episodes = []
            for _ in range(self.K_p):
                episode_idx += 1
                self.logger.episode_init(episode_idx)
                c_post_episodes.extend(self._collect_rollout())
                self.prep_training()
                actor_train_infos, critic_train_info = self._train_selected(
                    train_disruptor=True, train_concealment=False
                )
                self.after_update()

            # --- Phase 4: slow dual update, every `lambda_update_period` (M) outer cycles,
            # from the plain mean of the last M cycles' own post-adaptation costs (no EMA
            # coefficient needed -- M alone controls both the update frequency and the
            # averaging window, per user's simplification request). Uses the LAST of this
            # cycle's K_p rounds' `critic_train_info`/`actor_train_infos` for the cycle-level
            # c_post/lambda logging line below, but the cost itself is pooled across ALL K_p
            # rounds' completed episodes.
            if not c_post_episodes:
                # Rare (post-fix, expected only if genuinely zero threads finished an episode
                # during this specific rollout) -- print instead of silently writing NaN to
                # tensorboard with no explanation, per user's "不要直接过滤掉NaN,先打印tag和来源".
                print(f"[c_post] outer_cycle={self._outer_cycle_count}: 0 episodes completed "
                      f"this post-adaptation rollout -- c_post logged as NaN, window NOT updated.")
            c_post = float(np.mean(c_post_episodes)) if c_post_episodes else None
            if c_post is not None:
                assert np.isfinite(c_post), f"c_post is non-finite: {c_post}"
                self._c_post_window.append(c_post)
            critic_train_info["c_post"] = c_post if c_post is not None else float("nan")
            window_mean = float(np.mean(self._c_post_window)) if self._c_post_window else None
            critic_train_info["c_window_mean"] = window_mean if window_mean is not None else float("nan")
            if self._outer_cycle_count % self.lambda_update_period == 0 and window_mean is not None:
                assert np.isfinite(window_mean), f"window_mean is non-finite: {window_mean}"
                # 2026-09-20 bug fix (found via "无合作" baseline: constraint_mode="illu" was
                # silently ignored here -- this method never had an "illu" branch, unlike
                # OnPolicyLagrRunner.train()'s own (correct) illu/hard/soft handling, so passing
                # illu to the ALTERNATING runner fell through to the "soft" else-branch and kept
                # dual-ascending lambda anyway (observed: lambda drifted 0.0 -> 0.565 over a
                # "lambda_init=0, constraint_mode=illu" run that was supposed to keep it fixed at
                # exactly 0 for a no-Lagrangian-coupling ablation).
                if self.constraint_mode == "hard":
                    self.lam = self.lambda_max if window_mean > self.eps_cost else 0.0
                elif self.constraint_mode == "illu":
                    pass  # lambda stays fixed at lambda_init, exactly matching OnPolicyLagrRunner
                else:  # soft
                    self.lam = float(np.clip(
                        self.lam + self.alpha_lambda * (window_mean - self.eps_cost),
                        0.0, self.lambda_max,
                    ))
                assert np.isfinite(self.lam), f"self.lam became non-finite after update: {self.lam}"
            critic_train_info["lambda"] = self.lam

            if episode_idx % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos, critic_train_info, self.actor_buffer, self.critic_buffer
                )
            if cycle % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()
            self.after_update()
