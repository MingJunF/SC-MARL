#!/bin/bash
# Evaluate every already-trained, config-valid HC/Hopper attacker checkpoint against a given
# detector (2026-09-22, --detector pedm|cotd, see harl/detectors/{pedm,cotd}_detector.py) -- the
# same checkpoint set reported throughout this project's history (see
# scripts/run_hc_hopper_full_matrix.sh's per-block comments for why each one is considered
# valid: L2-budget-fix applied, and cost_aggregation genuinely mean where that matters).
#
# Requires results/obs_attackers/{HalfCheetah-v4,Hopper-v4}/<detector>_detector.pt to already
# exist (python -m examples.train_{pedm,cotd}_detector ...) -- this script does not train them.
#
# Usage: bash scripts/eval_existing_checkpoints_cotd.sh [episodes] [pedm|cotd]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate harl_marl 2>/dev/null || conda activate mujoko 2>/dev/null || {
    echo "no 'harl_marl' or 'mujoko' conda env found" >&2
    exit 1
}

EPISODES="${1:-15}"
DETECTOR="${2:-pedm}"
OUT="/tmp/eval_existing_checkpoints_${DETECTOR}.log"
: > "$OUT"

log() { echo "$@" | tee -a "$OUT"; }

eval_mujoco() {
    local label="$1" run_dir="$2"
    log "=== $label ($run_dir) ==="
    python -u -m examples.eval_mujoco_marl_vs_pedm --detector "$DETECTOR" --episodes "$EPISODES" \
        --run_dir "$run_dir" 2>&1 | grep -E "^(clean|attacked|return drop)" | tee -a "$OUT"
}

eval_illusory() {
    local label="$1" ckpt="$2" scenario="$3" victim="$4" detector_ckpt="$5"
    log "=== $label ($ckpt) ==="
    python -u -m examples.eval_illusory_matrix --detector "$DETECTOR" --episodes "$EPISODES" \
        --ckpt "$ckpt" --scenario "$scenario" --victim_run "$victim" --pedm_ckpt "$detector_ckpt" \
        2>&1 | grep -E "^(clean|attacked|return drop)" | tee -a "$OUT"
}

VICTIM_HC=results/robust_victim/HalfCheetah-v4/mappo/victim_halfcheetah/seed-00010-2026-06-24-22-04-33
VICTIM_HOPPER=results/robust_victim/Hopper-v4/mappo/victim6m/seed-00001-2026-06-29-12-56-28

log "==================== HC ===================="
for s in 1 2 3; do
    eval_mujoco "hc_actonly_s${s}" "results/mujoco_marl/HalfCheetah-v4/mappo_lagr/harl_native_hc_actonly_l2_s${s}"
done
eval_mujoco "hc_scmarl_alt_s1" "results/mujoco_marl/HalfCheetah-v4/mappo_alt/harl_native_hc_scmarl_alt_fixed_s1"
eval_mujoco "hc_scmarl_alt_s2" "results/mujoco_marl/HalfCheetah-v4/mappo_alt/harl_native_hc_scmarl_alt_s2"
eval_mujoco "hc_scmarl_alt_s3" "results/mujoco_marl/HalfCheetah-v4/mappo_alt/harl_native_hc_scmarl_alt_s3"
for s in 1 2 3; do
    eval_mujoco "hc_hardconstraint_s${s}" "results/mujoco_marl/HalfCheetah-v4/mappo_alt/harl_native_hc_hardconstraint_s${s}"
done
for s in 1 2 3; do
    eval_mujoco "hc_concealonly_s${s}" "results/mujoco_marl/HalfCheetah-v4/mappo_lagr/harl_native_hc_concealonly_l2_s${s}"
done
for s in 2 3; do
    eval_mujoco "hc_notalt_s${s}" "results/mujoco_marl/HalfCheetah-v4/mappo_lagr/harl_native_hc_notalt_l2_s${s}"
done
for s in 1 2 3; do
    eval_illusory "hc_illusory_s${s}" "results/obs_attackers/HalfCheetah-v4/illusory_l2_matrix_s${s}/attacker_obs_illusory.pt" \
        HalfCheetah-v4 "$VICTIM_HC" "results/obs_attackers/HalfCheetah-v4/${DETECTOR}_detector.pt"
done

log "==================== Hopper ===================="
for s in 1 2; do
    eval_mujoco "hopper_actonly_s${s}" "results/mujoco_marl/Hopper-v4/mappo_lagr/harl_native_hopper_actonly_l2_fixed_s${s}"
done
eval_mujoco "hopper_scmarl_alt_s1" "results/mujoco_marl/Hopper-v4/mappo_alt/harl_native_hopper_scmarl_alt_meancost_real_s1"
eval_illusory "hopper_illusory_s1" "results/obs_attackers/Hopper-v4/illusory_l2_matrix_fixed_s1/attacker_obs_illusory.pt" \
    Hopper-v4 "$VICTIM_HOPPER" "results/obs_attackers/Hopper-v4/${DETECTOR}_detector.pt"

log "==================== DONE ===================="
echo "full log: $OUT"
