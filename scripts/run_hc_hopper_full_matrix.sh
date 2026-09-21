#!/bin/bash
# Full HC + Ant experiment matrix (2026-09-22), 4 seeds each, max 3 concurrent jobs.
# HC includes both ablations (hard-constraint, not-alternating); Ant does not (matches this
# project's own established pattern of scoping ablations to HC only, per explicit user
# instruction). Uses the CURRENT best-validated settings from this project's debugging history
# -- see harl/runners/on_policy_lagr_runner.py's `cost_aggregation` docstring and
# harl/envs/mujoco_marl/mujoco_marl_env.py's `project_budget` usage for the full rationale:
#   - budget_norm=l2, with the FIXED project_budget (scale-then-project, so `budget` bounds the
#     REAL physical L2 perturbation identically regardless of the environment's own obs_scale
#     spread).
#   - cost_aggregation=mean (per-step target, portable across environments whose episodes can
#     terminate early e.g. from falling -- Ant -- unlike the old per-episode-SUM convention).
#     This key MUST exist in harl/configs/algos_cfgs/mappo_alt.yaml / mappo_lagr.yaml for the
#     CLI override to actually take effect -- HARL's update_args() only overrides EXISTING yaml
#     keys (verify with get_defaults_yaml_args+update_args if in doubt).
#   - eps_cost is DIFFERENT per environment (calibrated from each env's own clean truedyn
#     per-step floor, ~10x above it): HalfCheetah -> eps_cost=0.1; Ant -> eps_cost=0.04. A
#     tighter eps_cost=0.005 was tried for Ant and did NOT improve OR/CUSUM outcomes (in fact
#     slightly worse at a comparable training stage) -- 0.04 is the current best-known value,
#     not further tuned in this script.
#   - lambda_max=20 (NOT unbounded): an unbounded lambda was tried on Hopper and made results
#     WORSE (more erratic, some CUSUM peaks went UP) without ever converging.
#
# Variants:
#   HC (7 variants x 4 seeds = 28 runs):
#     1. act-only          -- pure Disruptor, no Concealment, no stealth objective at all
#        (mappo_lagr, constraint_mode=illu, lambda_init=0)
#     2. obs-only           -- pure single-agent OBSERVATION attacker, NO illusory-consistency
#        term (train_obs_attacker.py --mode ppo --attack obs) -- physically perturbs what the
#        victim itself sees, reward-only objective, no stealth consideration at all (NEW, per
#        explicit user request "baseline多加一个obsattackonly的，无illusoryde")
#     3. illusory            -- single-agent OBSERVATION attacker WITH illusory-consistency
#        constraint (train_obs_attacker.py --mode illusory --attack obs)
#     4. scmarl_alt          -- the main method: role-split Disruptor+Concealment, alternating
#        training, soft Lagrangian stealth constraint (mappo_alt)
#     5. [ablation] hardconstraint -- same as scmarl_alt but constraint_mode=hard (bang-bang
#        lambda, no continuous dual ascent)
#     6. concealonly         -- pure Concealment, Disruptor absent (disruptor_eps=0)
#     7. [ablation] notalt    -- same budgets/eps_cost as scmarl_alt but synchronous mappo_lagr
#        instead of the alternating scheme
#   Ant (5 variants x 4 seeds = 20 runs): same as HC variants 1-4 and 6 (no ablations 5/7).
#
# Usage:
#   bash scripts/run_hc_ant_full_matrix.sh [n_rollout_threads] [max_concurrent_jobs]
#   (defaults: 8 threads/job, 3 concurrent jobs)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate harl_marl 2>/dev/null || conda activate mujoko 2>/dev/null || {
    echo "no 'harl_marl' or 'mujoko' conda env found -- run environment/install.sh first" >&2
    exit 1
}

N_THREADS="${1:-8}"
MAX_JOBS="${2:-3}"
LOGDIR=/tmp/hc_ant_full_matrix_logs
mkdir -p "$LOGDIR"

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 20
    done
}
run_job() {
    local name="$1"; shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $name"
    "$@" > "$LOGDIR/${name}.log" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $name (exit $?)"
}
queue_job() {
    wait_for_slot
    run_job "$@" &
}

VICTIM_HC=results/robust_victim/HalfCheetah-v4/mappo/victim_halfcheetah/seed-00010-2026-06-24-22-04-33
VICTIM_ANT=results/robust_victim/Ant-v4/mappo/victim6m/seed-00001-2026-06-28-01-25-45
# Both must exist BEFORE running this script -- see environment/README.md for how to obtain/
# regenerate victim checkpoints and PEDM detectors (neither ships in the git repo).

SEEDS="1 2 3 4"

echo "=================== HC + Ant FULL MATRIX START $(date) ==================="

# --- HC: 1. act-only ---
for seed in $SEEDS; do
    queue_job "hc_actonly_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_hc_actonly_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0 --hidden_act_eps 0 --budget_norm l2 \
        --constraint_mode illu --lambda_init 0.0 --seed ${seed}
done

# --- HC: 2. obs-only (NEW, no illusory constraint) ---
for seed in $SEEDS; do
    queue_job "hc_obsonly_s${seed}" python -u -m examples.train_obs_attacker --scenario HalfCheetah-v4 \
        --mode ppo --attack obs --budget 0.2 --budget_norm l2 \
        --victim_run "$VICTIM_HC" --num_env_steps 4000000 --log_interval 20 \
        --save_dir "results/obs_attackers/HalfCheetah-v4/obsonly_matrix_s${seed}" --seed ${seed}
done

# --- HC: 3. illusory ---
for seed in $SEEDS; do
    queue_job "hc_illusory_s${seed}" python -u -m examples.train_obs_attacker --scenario HalfCheetah-v4 \
        --mode illusory --attack obs --budget 0.2 --budget_norm l2 \
        --victim_run "$VICTIM_HC" --num_env_steps 4000000 --log_interval 20 \
        --save_dir "results/obs_attackers/HalfCheetah-v4/illusory_matrix_s${seed}" --seed ${seed}
done

# --- HC: 4. SC-MARL alternating (main method) ---
for seed in $SEEDS; do
    queue_job "hc_scmarl_alt_s${seed}" python -u -m examples.train --algo mappo_alt --env mujoco_marl \
        --exp_name harl_native_hc_scmarl_alt_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 1000 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.1 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 \
        --k_hidden 5 --k_perf 2 --lambda_update_period 5 --seed ${seed}
done

# --- HC: 5. [ablation] hard-constraint ---
for seed in $SEEDS; do
    queue_job "hc_hardconstraint_s${seed}" python -u -m examples.train --algo mappo_alt --env mujoco_marl \
        --exp_name harl_native_hc_hardconstraint_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 1000 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode hard --cost_aggregation mean --eps_cost 0.1 --lambda_max 20.0 \
        --k_hidden 5 --k_perf 2 --lambda_update_period 5 --seed ${seed}
done

# --- HC: 6. concealer-only ---
for seed in $SEEDS; do
    queue_job "hc_concealonly_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_hc_concealonly_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.1 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 --seed ${seed}
done

# --- HC: 7. [ablation] not-alternating (synchronous) ---
for seed in $SEEDS; do
    queue_job "hc_notalt_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_hc_notalt_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.1 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 --seed ${seed}
done

# --- Ant: 1. act-only ---
for seed in $SEEDS; do
    queue_job "ant_actonly_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_ant_actonly_s${seed} \
        --scenario Ant-v4 --victim_run "$VICTIM_ANT" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0 --hidden_act_eps 0 --budget_norm l2 \
        --constraint_mode illu --lambda_init 0.0 --seed ${seed}
done

# --- Ant: 2. obs-only (NEW, no illusory constraint) ---
for seed in $SEEDS; do
    queue_job "ant_obsonly_s${seed}" python -u -m examples.train_obs_attacker --scenario Ant-v4 \
        --mode ppo --attack obs --budget 0.2 --budget_norm l2 \
        --victim_run "$VICTIM_ANT" --num_env_steps 4000000 --log_interval 20 \
        --save_dir "results/obs_attackers/Ant-v4/obsonly_matrix_s${seed}" --seed ${seed}
done

# --- Ant: 3. illusory ---
for seed in $SEEDS; do
    queue_job "ant_illusory_s${seed}" python -u -m examples.train_obs_attacker --scenario Ant-v4 \
        --mode illusory --attack obs --budget 0.2 --budget_norm l2 \
        --victim_run "$VICTIM_ANT" --num_env_steps 4000000 --log_interval 20 \
        --save_dir "results/obs_attackers/Ant-v4/illusory_matrix_s${seed}" --seed ${seed}
done

# --- Ant: 4. SC-MARL alternating (main method) ---
for seed in $SEEDS; do
    queue_job "ant_scmarl_alt_s${seed}" python -u -m examples.train --algo mappo_alt --env mujoco_marl \
        --exp_name harl_native_ant_scmarl_alt_s${seed} \
        --scenario Ant-v4 --victim_run "$VICTIM_ANT" \
        --n_rollout_threads "$N_THREADS" --episode_length 1000 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.04 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 \
        --k_hidden 5 --k_perf 2 --lambda_update_period 5 --seed ${seed}
done

# --- Ant: 5. concealer-only ---
for seed in $SEEDS; do
    queue_job "ant_concealonly_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_ant_concealonly_s${seed} \
        --scenario Ant-v4 --victim_run "$VICTIM_ANT" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.04 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 --seed ${seed}
done

echo "waiting for all jobs to finish..."
wait
echo "=================== HC + Ant FULL MATRIX COMPLETE $(date) ==================="
echo "ALL DONE"
echo "Evaluate mujoco_marl-based variants with:"
echo "  python -m examples.eval_mujoco_marl_vs_pedm --run_dir results/mujoco_marl/<Scenario-v4>/<mappo_alt|mappo_lagr>/<exp_name> --episodes 15"
echo "Evaluate train_obs_attacker.py-based variants (obs-only, illusory) with:"
echo "  python -m examples.eval_illusory_matrix --ckpt results/obs_attackers/<Scenario-v4>/<obsonly|illusory>_matrix_s<seed>/attacker_obs_<ppo|illusory>.pt --scenario <Scenario-v4> --victim_run <victim dir> --pedm_ckpt results/obs_attackers/<Scenario-v4>/pedm_detector.pt --episodes 15"
