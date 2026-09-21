"""Runner for training an illusory-attack adversary against a HARL victim.

This runner implements the illusory-attack algorithm (Franzmeyer et al., ICLR
2024) for HARL multi-agent policies. A frozen, pre-trained victim policy is
attacked by perturbing each agent's observation. The adversary is trained with
PPO to minimise the victim's return while keeping the perturbed observations
consistent with a learned forward dynamics model (the *illusory* /
detectability term). The same probabilistic ensemble dynamics model powers a
PEDM anomaly detector that flags perturbed transitions.

Workflow:
    1. Load and freeze a pre-trained victim policy.
    2. Collect clean victim rollouts and fit a per-agent PEDM dynamics model.
    3. Calibrate the PEDM detector threshold on clean trajectories.
    4. Train the adversary with PPO using the illusory reward.
    5. Periodically evaluate attack strength (victim return drop) and detector
       detection rate / false-positive rate.
"""

import json
import os
import time

import numpy as np
import torch
import setproctitle

from harl.algorithms.actors import ALGO_REGISTRY
from harl.attacks.adversary_ac import AdversaryActorCritic
from harl.detectors.pedm_detector import PEDMDetector
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    set_seed,
    get_num_agents,
    get_shape_from_act_space,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config
from harl.utils.trans_tools import _t2n


class AdversaryRunner:
    """Train an illusory-attack adversary against a frozen HARL victim."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the AdversaryRunner.

        Args:
            args: dict with keys 'algo' (== "illusory"), 'env', 'exp_name'.
            algo_args: adversary/detector config (from illusory.yaml).
            env_args: environment config (must match the victim's env).
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        set_seed(algo_args["seed"])
        self.seed = algo_args["seed"]["seed"]
        self.device = init_device(algo_args["device"])

        self.n_rollout_threads = algo_args["train"]["n_rollout_threads"]
        self.episode_length = algo_args["train"]["episode_length"]

        # ---- load victim configuration ----
        self.victim_cfg = self._load_victim_config()
        self.victim_algo = self.victim_cfg["main_args"]["algo"]
        self.victim_algo_args = self.victim_cfg["algo_args"]
        self.victim_model_dir = algo_args["victim"]["model_dir"]

        # ---- output dir ----
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"],
            env_args,
            args["algo"],
            args["exp_name"],
            self.seed,
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(args, algo_args, env_args, self.run_dir)
        setproctitle.setproctitle(
            f"{args['algo']}-{args['env']}-{args['exp_name']}"
        )

        # ---- envs ----
        self.envs = make_train_env(
            args["env"],
            self.seed,
            self.n_rollout_threads,
            env_args,
        )
        self.eval_envs = make_eval_env(
            args["env"],
            self.seed,
            algo_args["eval"]["n_eval_rollout_threads"],
            env_args,
        )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        # victim recurrent bookkeeping
        v_model = self.victim_algo_args["model"]
        self.recurrent_n = v_model["recurrent_n"]
        self.rnn_hidden_size = v_model["hidden_sizes"][-1]

        # per-agent dimensions
        self.obs_dims = [
            self.envs.observation_space[i].shape[0] for i in range(self.num_agents)
        ]
        self.act_dims = [
            get_shape_from_act_space(self.envs.action_space[i])
            for i in range(self.num_agents)
        ]
        self.act_spaces = [self.envs.action_space[i] for i in range(self.num_agents)]

        # which agents to attack
        target = algo_args["attack"]["target_agents"]
        if target == "all":
            self.target_agents = list(range(self.num_agents))
        else:
            self.target_agents = list(target)

        self.epsilon = algo_args["attack"]["epsilon"]
        self.lambda_illu = algo_args["attack"]["lambda_illu"]
        self.victim_deterministic = algo_args["attack"]["victim_deterministic"]

        self._build_victim()
        self._build_adversary()
        self._build_dynamics_and_detector()

    # ------------------------------------------------------------------ setup
    def _load_victim_config(self):
        cfg_path = self.algo_args["victim"]["config"]
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_victim(self):
        """Reconstruct and freeze the victim actors."""
        self.victim = []
        model_algo = {
            **self.victim_algo_args["model"],
            **self.victim_algo_args["algo"],
        }
        share_param = self.victim_algo_args["algo"]["share_param"]
        for agent_id in range(self.num_agents):
            if share_param and agent_id > 0:
                self.victim.append(self.victim[0])
                continue
            agent = ALGO_REGISTRY[self.victim_algo](
                model_algo,
                self.envs.observation_space[agent_id],
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.victim.append(agent)

        # load weights
        for agent_id in range(self.num_agents):
            if share_param and agent_id > 0:
                continue
            state_dict = torch.load(
                os.path.join(self.victim_model_dir, f"actor_agent{agent_id}.pt"),
                map_location=self.device,
            )
            self.victim[agent_id].actor.load_state_dict(state_dict)
            self.victim[agent_id].prep_rollout()

    def _build_adversary(self):
        """Build one adversary actor-critic per attacked agent."""
        adv_cfg = self.algo_args["adversary"]
        self.adversary = {}
        self.adv_optimizer = {}
        for agent_id in self.target_agents:
            obs_dim = self.obs_dims[agent_id]
            net = AdversaryActorCritic(
                obs_dim=obs_dim * 3,
                action_dim=obs_dim,
                hidden_size=adv_cfg["hidden_size"],
                activation=adv_cfg["activation"],
            ).to(self.device)
            self.adversary[agent_id] = net
            self.adv_optimizer[agent_id] = torch.optim.Adam(
                net.parameters(), lr=adv_cfg["lr"], eps=1e-5
            )

    def _build_dynamics_and_detector(self):
        dyn_cfg = self.algo_args["dynamics"]
        det_cfg = self.algo_args["detector"]
        self.detector = {}
        for agent_id in range(self.num_agents):
            det = PEDMDetector(
                obs_dim=self.obs_dims[agent_id],
                action_dim=self.act_dims[agent_id],
                ens_size=dyn_cfg["ens_size"],
                hidden_sizes=tuple(dyn_cfg["hidden_sizes"]),
                lr=dyn_cfg["lr"],
                n_part=det_cfg["n_part"],
                criterion=det_cfg["criterion"],
                aggregation_function=det_cfg["aggregation_function"],
                device=self.device,
            )
            self.detector[agent_id] = det

    # ------------------------------------------------------------- victim step
    @torch.no_grad()
    def _victim_act(self, obs, rnn_states, masks):
        """Compute victim actions for all agents given (perturbed) obs.

        Args:
            obs: (n_threads, n_agents, obs_dim)
            rnn_states: (n_threads, n_agents, recurrent_n, rnn_hidden_size)
            masks: (n_threads, n_agents, 1)
        Returns:
            actions: (n_threads, n_agents, act_dim)
            rnn_states: updated rnn states
        """
        action_collector = []
        for agent_id in range(self.num_agents):
            action, temp_rnn = self.victim[agent_id].act(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                masks[:, agent_id],
                None,
                deterministic=self.victim_deterministic,
            )
            rnn_states[:, agent_id] = _t2n(temp_rnn)
            action_collector.append(_t2n(action))
        actions = np.array(action_collector).transpose(1, 0, 2)
        return actions, rnn_states

    def _apply_perturbation(self, agent_id, true_obs, delta):
        """Map raw adversary output to a bounded perturbed observation.

        Args:
            true_obs: (n_threads, obs_dim) numpy
            delta: (n_threads, obs_dim) numpy raw perturbation
        Returns:
            perturbed_obs: (n_threads, obs_dim) numpy
        """
        if self.epsilon is not None and self.epsilon > 0:
            perturbed = true_obs + self.epsilon * np.tanh(delta)
        else:
            perturbed = true_obs + delta
        return perturbed

    # ------------------------------------------------------- dynamics pretrain
    def pretrain_dynamics(self):
        """Collect clean victim rollouts and fit the PEDM dynamics models."""
        dyn_cfg = self.algo_args["dynamics"]
        n_episodes = dyn_cfg["n_pretrain_episodes"]
        print(f"[illusory] collecting {n_episodes} clean episodes for dynamics fit")

        # per-agent transition buffers
        obs_buf = [[] for _ in range(self.num_agents)]
        act_buf = [[] for _ in range(self.num_agents)]
        nobs_buf = [[] for _ in range(self.num_agents)]
        # per-agent per-episode sequences for threshold calibration
        clean_seqs = [[] for _ in range(self.num_agents)]

        episodes = int(np.ceil(n_episodes / self.n_rollout_threads))
        for _ in range(episodes):
            obs, _, _ = self.envs.reset()
            rnn_states = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.recurrent_n,
                 self.rnn_hidden_size),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            ep_obs = [[obs[:, a].copy()] for a in range(self.num_agents)]
            ep_act = [[] for _ in range(self.num_agents)]

            for _step in range(self.episode_length):
                actions, rnn_states = self._victim_act(obs, rnn_states, masks)
                next_obs, _, _, dones, _, _ = self.envs.step(actions)
                for a in range(self.num_agents):
                    obs_buf[a].append(obs[:, a].copy())
                    act_buf[a].append(actions[:, a].copy())
                    nobs_buf[a].append(next_obs[:, a].copy())
                    ep_obs[a].append(next_obs[:, a].copy())
                    ep_act[a].append(actions[:, a].copy())

                dones_env = np.all(dones, axis=1)
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones_env] = 0.0
                rnn_states[dones_env] = 0.0
                obs = next_obs

            # store per-thread clean sequences (thread 0 is enough & cheap)
            for a in range(self.num_agents):
                seq_obs = np.stack([o[0] for o in ep_obs[a]], axis=0)  # (T+1, obs)
                seq_act = np.stack([ac[0] for ac in ep_act[a]], axis=0)  # (T, act)
                clean_seqs[a].append((seq_obs, seq_act))

        # fit dynamics + calibrate detector per agent
        for a in range(self.num_agents):
            obs_arr = np.concatenate(obs_buf[a], axis=0)
            act_arr = np.concatenate(act_buf[a], axis=0)
            nobs_arr = np.concatenate(nobs_buf[a], axis=0)
            print(
                f"[illusory] fitting PEDM for agent {a} on {len(obs_arr)} transitions"
            )
            self.detector[a].fit(
                obs_arr,
                act_arr,
                nobs_arr,
                n_train_epochs=dyn_cfg["n_train_epochs"],
                batch_size=dyn_cfg["batch_size"],
                verbose=True,
            )
            clean_scores = [
                self.detector[a].predict_scores(seq_obs, seq_act)
                for seq_obs, seq_act in clean_seqs[a]
            ]
            thr = self.detector[a].calibrate_threshold(
                clean_scores, quantile=self.algo_args["detector"]["threshold_quantile"]
            )
            print(f"[illusory] agent {a} detector threshold = {thr:.5f}")
            self.detector[a].dyn_model.save(
                os.path.join(self.save_dir, f"pedm_agent{a}.pt")
            )

    # -------------------------------------------------------------- adversary
    def run(self):
        """Full pipeline: dynamics pretrain, adversary training, evaluation."""
        self.pretrain_dynamics()

        num_updates = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.episode_length
            // self.n_rollout_threads
        )
        print(f"[illusory] training adversary for {num_updates} updates")
        for update in range(1, num_updates + 1):
            rollout = self._collect_rollout()
            train_info = self._update_adversary(rollout)

            if update % self.algo_args["train"]["log_interval"] == 0:
                step = update * self.episode_length * self.n_rollout_threads
                self._log(update, step, rollout, train_info)

            if update % self.algo_args["train"]["eval_interval"] == 0:
                self.evaluate(update)
                self._save()

        self._save()

    def _init_rollout_state(self):
        obs, _, _ = self.envs.reset()
        rnn_states = np.zeros(
            (self.n_rollout_threads, self.num_agents, self.recurrent_n,
             self.rnn_hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # last perturbed obs / last victim action per agent
        last_pert = {a: obs[:, a].copy() for a in range(self.num_agents)}
        last_vact = {
            a: np.zeros((self.n_rollout_threads, self.act_dims[a]), dtype=np.float32)
            for a in range(self.num_agents)
        }
        last_done = np.zeros(self.n_rollout_threads, dtype=np.float32)
        return obs, rnn_states, masks, last_pert, last_vact, last_done

    def _collect_rollout(self):
        """Collect one PPO rollout while attacking the victim."""
        T, N = self.episode_length, self.n_rollout_threads
        # storage per attacked agent
        store = {
            a: {
                "adv_obs": np.zeros((T, N, self.obs_dims[a] * 3), dtype=np.float32),
                "action": np.zeros((T, N, self.obs_dims[a]), dtype=np.float32),
                "logp": np.zeros((T, N), dtype=np.float32),
                "value": np.zeros((T, N), dtype=np.float32),
                "reward": np.zeros((T, N), dtype=np.float32),
                "done": np.zeros((T, N), dtype=np.float32),
                "illu": np.zeros((T, N), dtype=np.float32),
            }
            for a in self.target_agents
        }
        victim_return = np.zeros(N, dtype=np.float32)

        obs, rnn_states, masks, last_pert, last_vact, last_done = (
            self._init_rollout_state()
        )

        for t in range(T):
            perturbed_obs = obs.copy()
            step_delta = {}
            for a in self.target_agents:
                true_obs = obs[:, a]
                expected = self.detector[a].dyn_model.predict_next_obs_mean(
                    last_pert[a], last_vact[a]
                )
                # reset memory at episode boundaries
                mask_col = last_done[:, None]
                exp_masked = expected * (1 - mask_col) + true_obs * mask_col
                pert_prev = last_pert[a] * (1 - mask_col) + true_obs * mask_col

                adv_obs = np.concatenate([true_obs, pert_prev, exp_masked], axis=1)
                adv_obs_t = torch.as_tensor(
                    adv_obs, dtype=torch.float32, device=self.device
                )
                action, logp, value = self.adversary[a].act(adv_obs_t)
                delta = _t2n(action)
                perturbed = self._apply_perturbation(a, true_obs, delta)
                perturbed_obs[:, a] = perturbed

                illu = -np.linalg.norm(perturbed - exp_masked, axis=1) * (1 - last_done)

                store[a]["adv_obs"][t] = adv_obs
                store[a]["action"][t] = delta
                store[a]["logp"][t] = _t2n(logp)
                store[a]["value"][t] = _t2n(value)
                store[a]["illu"][t] = illu
                step_delta[a] = perturbed

            # victim acts on perturbed observations
            actions, rnn_states = self._victim_act(perturbed_obs, rnn_states, masks)
            next_obs, _, rewards, dones, _, _ = self.envs.step(actions)

            dones_env = np.all(dones, axis=1)
            victim_return += rewards[:, :, 0].mean(axis=1)

            for a in self.target_agents:
                r_victim = rewards[:, a, 0]
                adv_reward = -r_victim + self.lambda_illu * store[a]["illu"][t]
                store[a]["reward"][t] = adv_reward
                store[a]["done"][t] = dones_env.astype(np.float32)
                last_pert[a] = step_delta[a]
                last_vact[a] = actions[:, a].astype(np.float32)

            masks = np.ones((N, self.num_agents, 1), dtype=np.float32)
            masks[dones_env] = 0.0
            rnn_states[dones_env] = 0.0
            last_done = dones_env.astype(np.float32)
            obs = next_obs

        # bootstrap value at the end
        last_values = {}
        for a in self.target_agents:
            true_obs = obs[:, a]
            expected = self.detector[a].dyn_model.predict_next_obs_mean(
                last_pert[a], last_vact[a]
            )
            adv_obs = np.concatenate([true_obs, last_pert[a], expected], axis=1)
            adv_obs_t = torch.as_tensor(adv_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                last_values[a] = _t2n(self.adversary[a].get_value(adv_obs_t))

        store["_victim_return"] = victim_return
        store["_last_values"] = last_values
        return store

    def _compute_gae(self, reward, value, done, last_value):
        """Generalized advantage estimation.

        Args shapes: reward/value/done (T, N); last_value (N,).
        """
        adv_cfg = self.algo_args["adversary"]
        gamma, lam = adv_cfg["gamma"], adv_cfg["gae_lambda"]
        T, N = reward.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)
        next_value = last_value
        for t in reversed(range(T)):
            nonterminal = 1.0 - done[t]
            delta = reward[t] + gamma * next_value * nonterminal - value[t]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae
            next_value = value[t]
        returns = advantages + value
        return advantages, returns

    def _update_adversary(self, rollout):
        """PPO update for every adversary."""
        adv_cfg = self.algo_args["adversary"]
        T, N = self.episode_length, self.n_rollout_threads
        info = {"policy_loss": [], "value_loss": [], "entropy": []}

        for a in self.target_agents:
            s = rollout[a]
            advantages, returns = self._compute_gae(
                s["reward"], s["value"], s["done"], rollout["_last_values"][a]
            )
            b_obs = torch.as_tensor(
                s["adv_obs"].reshape(T * N, -1), dtype=torch.float32, device=self.device
            )
            b_act = torch.as_tensor(
                s["action"].reshape(T * N, -1), dtype=torch.float32, device=self.device
            )
            b_logp = torch.as_tensor(
                s["logp"].reshape(T * N), dtype=torch.float32, device=self.device
            )
            b_adv = torch.as_tensor(
                advantages.reshape(T * N), dtype=torch.float32, device=self.device
            )
            b_ret = torch.as_tensor(
                returns.reshape(T * N), dtype=torch.float32, device=self.device
            )
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            batch_size = T * N
            minibatch_size = batch_size // adv_cfg["num_mini_batch"]
            net = self.adversary[a]
            optimizer = self.adv_optimizer[a]

            for _ in range(adv_cfg["ppo_epoch"]):
                perm = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, minibatch_size):
                    idx = perm[start : start + minibatch_size]
                    new_logp, entropy, value = net.evaluate_actions(
                        b_obs[idx], b_act[idx]
                    )
                    ratio = torch.exp(new_logp - b_logp[idx])
                    surr1 = ratio * b_adv[idx]
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - adv_cfg["clip_eps"],
                        1.0 + adv_cfg["clip_eps"],
                    ) * b_adv[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * (value - b_ret[idx]).pow(2).mean()
                    ent = entropy.mean()
                    loss = (
                        policy_loss
                        + adv_cfg["vf_coef"] * value_loss
                        - adv_cfg["ent_coef"] * ent
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), adv_cfg["max_grad_norm"]
                    )
                    optimizer.step()

                    info["policy_loss"].append(policy_loss.item())
                    info["value_loss"].append(value_loss.item())
                    info["entropy"].append(ent.item())

        return {k: float(np.mean(v)) for k, v in info.items()}

    # ------------------------------------------------------------------- logs
    def _log(self, update, step, rollout, train_info):
        victim_return = float(rollout["_victim_return"].mean())
        illu = float(
            np.mean([rollout[a]["illu"].mean() for a in self.target_agents])
        )
        print(
            f"[illusory] update {update} step {step} "
            f"victim_return {victim_return:.3f} illu_reward {illu:.3f} "
            f"policy_loss {train_info['policy_loss']:.4f}"
        )
        self.writter.add_scalar("adversary/victim_return", victim_return, step)
        self.writter.add_scalar("adversary/illu_reward", illu, step)
        for k, v in train_info.items():
            self.writter.add_scalar(f"adversary/{k}", v, step)

    # -------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self, update):
        """Evaluate attack strength and detector performance.

        Runs clean and attacked episodes on the eval envs, reporting the victim
        return in both settings and, if the detector is enabled, the detection
        rate on attacked trajectories and false-positive rate on clean ones.
        """
        n_eval = self.algo_args["eval"]["eval_episodes"]
        clean_ret = self._rollout_eval(attack=False, n_episodes=n_eval)
        attack_ret, det_rate, fp_rate = self._rollout_eval(
            attack=True, n_episodes=n_eval, detect=True
        )
        step = update * self.episode_length * self.n_rollout_threads
        print(
            f"[illusory][eval] update {update} clean_return {clean_ret:.3f} "
            f"attacked_return {attack_ret:.3f} detection_rate {det_rate:.3f} "
            f"false_positive_rate {fp_rate:.3f}"
        )
        self.writter.add_scalar("eval/clean_return", clean_ret, step)
        self.writter.add_scalar("eval/attacked_return", attack_ret, step)
        self.writter.add_scalar("eval/detection_rate", det_rate, step)
        self.writter.add_scalar("eval/false_positive_rate", fp_rate, step)

    @torch.no_grad()
    def _rollout_eval(self, attack, n_episodes, detect=False):
        n_threads = self.algo_args["eval"]["n_eval_rollout_threads"]
        episodes = int(np.ceil(n_episodes / n_threads))
        returns = []
        flags = []  # per-transition anomaly flags on attacked agents

        for _ in range(episodes):
            obs, _, _ = self.eval_envs.reset()
            rnn_states = np.zeros(
                (n_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
            masks = np.ones((n_threads, self.num_agents, 1), dtype=np.float32)
            last_pert = {a: obs[:, a].copy() for a in range(self.num_agents)}
            last_vact = {
                a: np.zeros((n_threads, self.act_dims[a]), dtype=np.float32)
                for a in range(self.num_agents)
            }
            last_done = np.zeros(n_threads, dtype=np.float32)
            ep_ret = np.zeros(n_threads, dtype=np.float32)
            # per-agent observed sequences (thread 0) for detection
            seq_obs = {a: [obs[0, a].copy()] for a in self.target_agents}
            seq_act = {a: [] for a in self.target_agents}

            for _t in range(self.episode_length):
                perturbed_obs = obs.copy()
                for a in self.target_agents:
                    if attack:
                        true_obs = obs[:, a]
                        expected = self.detector[a].dyn_model.predict_next_obs_mean(
                            last_pert[a], last_vact[a]
                        )
                        mask_col = last_done[:, None]
                        exp_masked = expected * (1 - mask_col) + true_obs * mask_col
                        pert_prev = last_pert[a] * (1 - mask_col) + true_obs * mask_col
                        adv_obs = np.concatenate(
                            [true_obs, pert_prev, exp_masked], axis=1
                        )
                        adv_obs_t = torch.as_tensor(
                            adv_obs, dtype=torch.float32, device=self.device
                        )
                        action, _, _ = self.adversary[a].act(
                            adv_obs_t, deterministic=True
                        )
                        perturbed_obs[:, a] = self._apply_perturbation(
                            a, true_obs, _t2n(action)
                        )

                actions, rnn_states = self._victim_act(
                    perturbed_obs, rnn_states, masks
                )
                next_obs, _, rewards, dones, _, _ = self.eval_envs.step(actions)
                ep_ret += rewards[:, :, 0].mean(axis=1)
                dones_env = np.all(dones, axis=1)

                for a in self.target_agents:
                    # detector observes the (perturbed) obs the victim acted on
                    seq_obs[a].append(perturbed_obs[0, a].copy())
                    seq_act[a].append(actions[0, a].astype(np.float32))
                    last_pert[a] = perturbed_obs[:, a]
                    last_vact[a] = actions[:, a].astype(np.float32)

                masks = np.ones((n_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones_env] = 0.0
                rnn_states[dones_env] = 0.0
                last_done = dones_env.astype(np.float32)
                obs = next_obs

            returns.append(ep_ret.mean())

            if detect and self.algo_args["detector"]["enable"]:
                for a in self.target_agents:
                    so = np.stack(seq_obs[a], axis=0)
                    sa = np.stack(seq_act[a], axis=0)
                    flag = self.detector[a].detect(so, sa)
                    flags.append(flag)

        mean_return = float(np.mean(returns))
        if not detect:
            return mean_return

        # detection rate: fraction of attacked transitions flagged as anomalous
        if flags and self.algo_args["detector"]["enable"]:
            det_rate = float(np.mean(np.concatenate(flags)))
        else:
            det_rate = 0.0
        # false-positive rate is measured separately on clean rollouts
        fp_rate = self._clean_false_positive_rate(n_episodes=max(1, n_episodes // 2))
        return mean_return, det_rate, fp_rate

    @torch.no_grad()
    def _clean_false_positive_rate(self, n_episodes):
        if not self.algo_args["detector"]["enable"]:
            return 0.0
        n_threads = self.algo_args["eval"]["n_eval_rollout_threads"]
        episodes = int(np.ceil(n_episodes / n_threads))
        flags = []
        for _ in range(episodes):
            obs, _, _ = self.eval_envs.reset()
            rnn_states = np.zeros(
                (n_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
            masks = np.ones((n_threads, self.num_agents, 1), dtype=np.float32)
            seq_obs = {a: [obs[0, a].copy()] for a in self.target_agents}
            seq_act = {a: [] for a in self.target_agents}
            for _t in range(self.episode_length):
                actions, rnn_states = self._victim_act(obs, rnn_states, masks)
                next_obs, _, _, dones, _, _ = self.eval_envs.step(actions)
                dones_env = np.all(dones, axis=1)
                for a in self.target_agents:
                    seq_obs[a].append(obs[0, a].copy())
                    seq_act[a].append(actions[0, a].astype(np.float32))
                masks = np.ones((n_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones_env] = 0.0
                rnn_states[dones_env] = 0.0
                obs = next_obs
            for a in self.target_agents:
                so = np.stack(seq_obs[a], axis=0)
                sa = np.stack(seq_act[a], axis=0)
                flags.append(self.detector[a].detect(so, sa))
        return float(np.mean(np.concatenate(flags))) if flags else 0.0

    # ------------------------------------------------------------------- save
    def _save(self):
        for a in self.target_agents:
            torch.save(
                self.adversary[a].state_dict(),
                os.path.join(self.save_dir, f"adversary_agent{a}.pt"),
            )

    def close(self):
        self.envs.close()
        if self.eval_envs is not self.envs:
            self.eval_envs.close()
        self.writter.export_scalars_to_json(str(self.log_dir) + "/summary.json")
        self.writter.close()
