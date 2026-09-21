#!/bin/bash
# 3-seed SC-MARL alternating (main method) runs for HalfCheetah-v4 and Ant-v4 (2026-09-22).
# Uses the FINAL, validated settings from this project's own debugging history -- see
# harl/runners/on_policy_lagr_runner.py's `cost_aggregation` docstring and
# harl/envs/mujoco_marl/mujoco_marl_env.py's `project_budget` usage for the full rationale:
#   - budget_norm=l2, with the FIXED project_budget (scale-then-project, so `budget` bounds the
#     REAL physical L2 perturbation identically regardless of the environment's own obs_scale
#     spread -- do not use a version of this codebase predating that fix, results will not be
#     comparable).
#   - cost_aggregation=mean (per-step target, portable across environments whose episodes can
#     terminate early e.g. from falling -- Ant/Hopper -- unlike the old per-episode-SUM
#     convention, which silently assumed ~1000-step episodes). This key MUST exist in
#     harl/configs/algos_cfgs/mappo_alt.yaml for the CLI override below to actually take effect
#     -- HARL's update_args() only overrides EXISTING yaml keys (confirmed the hard way: this
#     override was silently a no-op for a while before the yaml key was added -- verify with
#     `get_defaults_yaml_args`+`update_args` if you ever add a new key here).
#   - eps_cost is DIFFERENT per environment (calibrated from each env's own clean truedyn
#     per-step floor, ~10x above it, matching how HC's own eps_cost=0.1 was originally derived
#     from illusory's reference eps_kl=0.1): HalfCheetah clean floor ~0.01 -> eps_cost=0.1;
#     Ant clean floor ~0.0043 -> eps_cost=0.04 (Ant's own value is still under active
#     investigation as of this script's writing -- a tighter eps_cost=0.005 was being tested to
#     see if forcing lambda to saturate EARLIER in training, closer to how an accidentally-
#     mis-scaled earlier run behaved, improves the OR/CUSUM outcome without reintroducing the
#     units bug; re-check this project's own notes/results before trusting 0.04 as final for
#     Ant). Do NOT reuse HalfCheetah's eps_cost for Ant (or any other env) without re-measuring
#     that env's own clean floor first (see `examples/train_pedm_detector.py`'s sibling
#     truedyn-floor measurement pattern in this project's own history, not yet a standalone
#     tool -- roll a few thousand steps with BOTH agents' actions forced to zero and read
#     info["cost"]'s mean).
#   - lambda_max=20 (NOT unbounded): an unbounded lambda was tried and made results WORSE on
#     Hopper (episode return became more erratic, some CUSUM peaks went UP, not down) without
#     ever converging -- capping at 20 is the validated choice.
#
# Usage:
#   bash scripts/run_hc_ant_3seed.sh [n_rollout_threads] [max_concurrent_jobs]
#   (defaults: 8 threads/job, 2 concurrent jobs -- tune both up on a bigger server)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate harl_marl 2>/dev/null || conda activate mujoko 2>/dev/null || {
    echo "no 'harl_marl' or 'mujoko' conda env found -- run environment/install.sh first" >&2
    exit 1
}

N_THREADS="${1:-8}"
MAX_JOBS="${2:-2}"
LOGDIR=/tmp/hc_ant_3seed_logs
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

echo "=================== HC + Ant, 3 seeds each START $(date) ==================="

for seed in 1 2 3; do
    queue_job "hc_scmarl_alt_s${seed}" python -u -m examples.train --algo mappo_alt --env mujoco_marl \
        --exp_name harl_native_hc_scmarl_alt_3seed_s${seed} \
        --scenario HalfCheetah-v4 --victim_run "$VICTIM_HC" \
        --n_rollout_threads "$N_THREADS" --episode_length 1000 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.1 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 \
        --k_hidden 5 --k_perf 2 --lambda_update_period 5 --seed ${seed}
done

for seed in 1 2 3; do
    queue_job "ant_scmarl_alt_s${seed}" python -u -m examples.train --algo mappo_alt --env mujoco_marl \
        --exp_name harl_native_ant_scmarl_alt_3seed_s${seed} \
        --scenario Ant-v4 --victim_run "$VICTIM_ANT" \
        --n_rollout_threads "$N_THREADS" --episode_length 1000 --num_env_steps 4000000 --log_interval 5 --use_eval False \
        --disruptor_eps 0.4 --hidden_eps 0.2 --hidden_act_eps 0.2 --budget_norm l2 \
        --constraint_mode soft --cost_aggregation mean --eps_cost 0.04 --alpha_lambda 100 --lambda_init 10.0 --lambda_max 20.0 \
        --k_hidden 5 --k_perf 2 --lambda_update_period 5 --seed ${seed}
done

echo "waiting for all jobs to finish..."
wait
echo "=================== HC + Ant, 3 seeds each COMPLETE $(date) ==================="
echo "ALL DONE"
echo "Evaluate each with: python -m examples.eval_mujoco_marl_vs_pedm --run_dir results/mujoco_marl/<Scenario-v4>/mappo_alt/<exp_name> --episodes 15"
