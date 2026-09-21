#!/bin/bash
# Full HC + Hopper experiment matrix (2026-09-22), 4 seeds each, max 3 concurrent jobs.
# HC includes both ablations (hard-constraint, not-alternating); Hopper does not (matches this
# project's own established pattern of scoping ablations to HC only, per explicit user
# instruction). Uses the CURRENT best-validated settings from this project's debugging history
# -- see harl/runners/on_policy_lagr_runner.py's `cost_aggregation` docstring and
# harl/envs/mujoco_marl/mujoco_marl_env.py's `project_budget` usage for the full rationale:
#   - budget_norm=l2, with the FIXED project_budget (scale-then-project, so `budget` bounds the
#     REAL physical L2 perturbation identically regardless of the environment's own obs_scale
#     spread -- this mattered MOST for Hopper, whose obs_scale spans a 45x range, 5-6x blowing
#     past the nominal budget under the old buggy projection order).
#   - cost_aggregation=mean (per-step target, portable across environments whose episodes can
#     terminate early e.g. from falling -- Hopper's attacked episodes run ~97 steps, not 1000 --
#     unlike the old per-episode-SUM convention, which silently assumed ~1000-step episodes).
#     This key MUST exist in harl/configs/algos_cfgs/mappo_alt.yaml / mappo_lagr.yaml for the
#     CLI override to actually take effect -- HARL's update_args() only overrides EXISTING yaml
#     keys (verify with get_defaults_yaml_args+update_args if in doubt).
#   - eps_cost is DIFFERENT per environment (calibrated from each env's own clean truedyn
#     per-step floor, ~10x above it): HalfCheetah -> eps_cost=0.1; Hopper -> eps_cost=0.03.
#   - lambda_max=20 (NOT unbounded): an unbounded lambda was tried on Hopper and made results
#     WORSE (more erratic, some CUSUM peaks went UP) without ever converging.
#
# KNOWN RESULT, not yet overturned by any variant tried (7 distinct Hopper configs across this
# project's debugging history, always with a real attack present): Hopper's SC-MARL checkpoints
# have NEVER passed CUSUM, not even a single held-out episode out of 15 -- this script re-runs
# the matrix anyway (more seeds = a cleaner reported negative result, not an attempt to "fix" it
# further) rather than to chase a pass that has not appeared under any setting so far.
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
#   Hopper (5 variants x 4 seeds = 20 runs): same as HC variants 1-4 and 6 (no ablations 5/7).
#
# Usage:
#   bash scripts/run_hc_hopper_full_matrix.sh [n_rollout_threads] [max_concurrent_jobs]
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
LOGDIR=/tmp/hc_hopper_full_matrix_logs
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
VICTIM_HOPPER=results/robust_victim/Hopper-v4/mappo/victim6m/seed-00001-2026-06-29-12-56-28
# Both must exist BEFORE running this script -- both already checked into this repo (see
# results/robust_victim/), no retraining needed. Hopper also needs a PEDM detector for eval
# (results/obs_attackers/Hopper-v4/pedm_detector.pt -- NOT checked in, regenerate with
# `python -m examples.train_pedm_detector --scenario Hopper-v4 --victim_run "$VICTIM_HOPPER"`
# if missing).

echo "=================== HC + Hopper FULL MATRIX START $(date) ==================="
echo "(2026-09-22: pruned to skip seeds that already have valid, matching-config results on"
echo " disk from this project's 2026-09-20/21 HC/Hopper debugging history -- see the per-block"
echo " comments below for the exact directory each skip reuses.)"

# --- HC: 1. act-only --- seeds 1-3 already done+valid: results/mujoco_marl/HalfCheetah-v4/mappo_lagr/harl_native_hc_actonly_l2_s{1,2,3} (0/15,0/15,0/15 both OR+CUSUM). Only seed 4 needed.
for seed in 4; do
    queue_job "hc_actonly_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_hc_actonly_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0 --hidden_act_eps 0 --budget_norm l2 \
        --constraint_mode illu --lambda_init 0.0 --seed ${seed}
done

# --- HC: 2. obs-only (NEW, no illusory constraint) --- never run before, all 4 seeds needed.
for seed in 1 2 3 4; do
    queue_job "hc_obsonly_s${seed}" python -u -m examples.train_obs_attacker --scenario HalfCheetah-v4 \
        --mode ppo --attack obs --budget 0.2 --budget_norm l2 \
        --victim_run "$VICTIM_HC" --num_env_steps 4000000 --log_interval 20 \
        --save_dir "results/obs_attackers/HalfCheetah-v4/obsonly_matrix_s${seed}" --seed ${seed}
done

# --- HC: 3. illusory --- seeds 1-3 already done+valid: results/obs_attackers/HalfCheetah-v4/illusory_l2_matrix_s{1,2,3}. Only seed 4 needed.
for seed in 4; do
    queue_job "hc_illusory_s${seed}" python -u -m examples.train_obs_attacker --scenario HalfCheetah-v4 \
        --mode illusory --attack obs --budget 0.2 --budget_norm l2 \
        --victim_run "$VICTIM_HC" --num_env_steps 4000000 --log_interval 20 \
        --save_dir "results/obs_attackers/HalfCheetah-v4/illusory_matrix_s${seed}" --seed ${seed}
done

# --- HC: 4. SC-MARL alternating (main method) --- seeds 1-3 already done+valid: harl_native_hc_scmarl_alt_fixed_s1, _s2, _s3 (eps_cost=100/alpha_lambda=0.1 under "sum" aggregation -- mathematically IDENTICAL to eps_cost=0.1/alpha_lambda=100 under "mean" here, since HC episodes are always exactly 1000 steps, a constant scale factor). Only seed 4 needed.
for seed in 4; do
    queue_job "hc_scmarl_alt_s${seed}" python -u -m examples.train --algo mappo_alt --env mujoco_marl \
        --exp_name harl_native_hc_scmarl_alt_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 1000 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.1 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 \
        --k_hidden 5 --k_perf 2 --lambda_update_period 5 --seed ${seed}
done

# --- HC: 5. [ablation] hard-constraint --- seeds 1-3 already done+valid: harl_native_hc_hardconstraint_s{1,2,3} (same sum/mean equivalence as above). Only seed 4 needed.
for seed in 4; do
    queue_job "hc_hardconstraint_s${seed}" python -u -m examples.train --algo mappo_alt --env mujoco_marl \
        --exp_name harl_native_hc_hardconstraint_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 1000 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode hard --cost_aggregation mean --eps_cost 0.1 --lambda_max 20.0 \
        --k_hidden 5 --k_perf 2 --lambda_update_period 5 --seed ${seed}
done

# --- HC: 6. concealer-only --- seeds 1-3 already done+valid: harl_native_hc_concealonly_l2_s{1,2,3} (same sum/mean equivalence). Only seed 4 needed.
for seed in 4; do
    queue_job "hc_concealonly_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_hc_concealonly_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.1 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 --seed ${seed}
done

# --- HC: 7. [ablation] not-alternating (synchronous) --- seeds 2,3 already done+valid: harl_native_hc_notalt_l2_s{2,3}. Seed 1 must be REDONE (the old seed1, harl_native_hc_v8_sync_ablation, used L-infinity not L2 -- a norm mismatch, CUSUM 4/15 vs seeds 2/3's 0/15). Seed 4 also needed.
for seed in 1 4; do
    queue_job "hc_notalt_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_hc_notalt_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.1 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 --seed ${seed}
done

# --- Hopper: 1. act-only --- seeds 1,2 already done+valid (L2-budget-fix applied; cost_aggregation is irrelevant here since constraint_mode=illu/lambda_init=0 never updates lambda regardless): results/mujoco_marl/Hopper-v4/mappo_lagr/harl_native_hopper_actonly_l2_fixed_s{1,2}. Seeds 3,4 needed.
for seed in 3 4; do
    queue_job "hopper_actonly_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_hopper_actonly_s${seed} \
        --scenario Hopper-v4 --victim_run "$VICTIM_HOPPER" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0 --hidden_act_eps 0 --budget_norm l2 \
        --constraint_mode illu --lambda_init 0.0 --seed ${seed}
done

# --- Hopper: 2. obs-only (NEW, no illusory constraint) --- never run before, all 4 seeds needed.
for seed in 1 2 3 4; do
    queue_job "hopper_obsonly_s${seed}" python -u -m examples.train_obs_attacker --scenario Hopper-v4 \
        --mode ppo --attack obs --budget 0.2 --budget_norm l2 \
        --victim_run "$VICTIM_HOPPER" --num_env_steps 4000000 --log_interval 20 \
        --save_dir "results/obs_attackers/Hopper-v4/obsonly_matrix_s${seed}" --seed ${seed}
done

# --- Hopper: 3. illusory --- seed 1 already done+valid (post L2-fix, created 13:44 vs the invalid pre-fix s1/s2/s3 at 08:09): results/obs_attackers/Hopper-v4/illusory_l2_matrix_fixed_s1. Seeds 2-4 needed.
for seed in 2 3 4; do
    queue_job "hopper_illusory_s${seed}" python -u -m examples.train_obs_attacker --scenario Hopper-v4 \
        --mode illusory --attack obs --budget 0.2 --budget_norm l2 \
        --victim_run "$VICTIM_HOPPER" --num_env_steps 4000000 --log_interval 20 \
        --save_dir "results/obs_attackers/Hopper-v4/illusory_matrix_s${seed}" --seed ${seed}
done

# --- Hopper: 4. SC-MARL alternating (main method) --- seed 1 already done+valid: harl_native_hopper_scmarl_alt_meancost_real_s1 (config-verified: L2-fixed budget AND cost_aggregation=mean genuinely applied, eps_cost=0.03/alpha_lambda=100/lambda_init=10/lambda_max=20/k_hidden=5/k_perf=2 -- every earlier Hopper scmarl_alt dir, "_fixed_s1" included, predates the cost_aggregation yaml fix and was secretly still "sum"). Seeds 2-4 needed. KNOWN RESULT: this checkpoint still fails CUSUM (15/15) -- expected, not a bug, see header comment.
for seed in 2 3 4; do
    queue_job "hopper_scmarl_alt_s${seed}" python -u -m examples.train --algo mappo_alt --env mujoco_marl \
        --exp_name harl_native_hopper_scmarl_alt_s${seed} \
        --scenario Hopper-v4 --victim_run "$VICTIM_HOPPER" \
        --n_rollout_threads "$N_THREADS" --episode_length 1000 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.03 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 \
        --k_hidden 5 --k_perf 2 --lambda_update_period 5 --seed ${seed}
done

# --- Hopper: 5. concealer-only --- the only existing dirs (harl_native_hopper_concealonly_l2_s{1,2,3}) predate the L2-budget fix (mtime 09:58 on 2026-09-21, before the ~13:xx fix) -- no "_fixed" variant was ever run. All 4 seeds needed fresh.
for seed in 1 2 3 4; do
    queue_job "hopper_concealonly_s${seed}" python -u -m examples.train --algo mappo_lagr --env mujoco_marl \
        --exp_name harl_native_hopper_concealonly_s${seed} \
        --scenario Hopper-v4 --victim_run "$VICTIM_HOPPER" \
        --n_rollout_threads "$N_THREADS" --episode_length 200 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.03 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 --seed ${seed}
done

echo "waiting for all jobs to finish..."
wait
echo "=================== HC + Hopper FULL MATRIX COMPLETE $(date) ==================="
echo "ALL DONE"
echo "Evaluate mujoco_marl-based variants with:"
echo "  python -m examples.eval_mujoco_marl_vs_pedm --run_dir results/mujoco_marl/<Scenario-v4>/<mappo_alt|mappo_lagr>/<exp_name> --episodes 15"
echo "Evaluate train_obs_attacker.py-based variants (obs-only, illusory) with:"
echo "  python -m examples.eval_illusory_matrix --ckpt results/obs_attackers/<Scenario-v4>/<obsonly|illusory>_matrix_s<seed>/attacker_obs_<ppo|illusory>.pt --scenario <Scenario-v4> --victim_run <victim dir> --pedm_ckpt results/obs_attackers/<Scenario-v4>/pedm_detector.pt --episodes 15"
