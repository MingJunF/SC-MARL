#!/bin/bash
# Auto-install script for this project's HARL-native SC-MARL environment (2026-09-21).
# Creates a conda env with everything needed to run examples/train.py (mujoco_marl /
# safety_marl envs, mappo_lagr / mappo_alt algos) and the eval_*_vs_pedm.py scripts.
#
# Usage:
#   bash environment/install.sh [env_name] [--cpu]
#
#   env_name   conda env name to create (default: harl_marl)
#   --cpu      install the CPU-only torch build instead of the CUDA one (default: CUDA,
#              matches this project's own dev machine: torch 2.11.0+cu130 on an RTX 4080S --
#              adjust TORCH_CUDA_INDEX below if the server's driver needs an older CUDA build)
set -euo pipefail

ENV_NAME="${1:-harl_marl}"
CPU_ONLY=false
for arg in "$@"; do
    if [ "$arg" == "--cpu" ]; then CPU_ONLY=true; fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# `conda activate` needs the shell FUNCTION that conda.sh installs, not just the `conda`
# binary on PATH -- a non-interactive script's shell never gets that function just because
# `command -v conda` succeeds (this bit a fresh container whose base env was already active
# via `conda`/PATH but had never run `conda init`: `conda activate` failed with "CondaError:
# Run 'conda init' before 'conda activate'", `set -e` then killed the script BEFORE any pip
# install ran, silently leaving the env with nothing but bare python in it). So: always
# locate and source conda.sh, regardless of whether `conda` is already callable.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniforge3"
elif [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/miniforge3"
else
    echo "conda not found -- install Miniconda/Anaconda/Miniforge first: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi
if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "conda found on PATH but ${CONDA_BASE}/etc/profile.d/conda.sh is missing -- unusual install layout, source your conda.sh manually before re-running this script" >&2
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[install.sh] conda env '${ENV_NAME}' already exists -- reusing it (delete it first with 'conda env remove -n ${ENV_NAME}' for a clean reinstall)."
else
    echo "[install.sh] creating conda env '${ENV_NAME}' (python 3.10)..."
    conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

if [ "$CPU_ONLY" = true ]; then
    echo "[install.sh] installing CPU-only torch..."
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0
else
    echo "[install.sh] installing CUDA torch (cu130 -- edit TORCH_CUDA_INDEX in this script if the server needs a different CUDA build)..."
    TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu130"
    pip install --index-url "$TORCH_CUDA_INDEX" torch==2.11.0 torchvision==0.26.0
fi

echo "[install.sh] installing the rest of requirements.txt..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "[install.sh] installing this repo in editable mode (for the harl package itself)..."
pip install -e "${REPO_ROOT}" --no-deps 2>/dev/null || echo "  (no setup.py/pyproject at repo root -- skipping, harl/ is imported via PYTHONPATH=repo root instead, e.g. 'python -m examples.train ...' from ${REPO_ROOT})"

echo "[install.sh] verifying the install..."
python - <<'PYEOF'
import torch, gymnasium, mujoco, safety_gymnasium, numpy
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
print("gymnasium", gymnasium.__version__)
print("mujoco", mujoco.__version__)
print("safety_gymnasium", safety_gymnasium.__version__)
print("numpy", numpy.__version__, "(must stay 1.23.x -- see requirements.txt's own comment)")
import gymnasium as gym
for scenario in ["HalfCheetah-v4", "Ant-v4", "Hopper-v4"]:
    e = gym.make(scenario)
    e.close()
    print(f"  {scenario}: OK")
PYEOF

echo "[install.sh] done. Activate with: conda activate ${ENV_NAME}"
echo "[install.sh] then run experiments from the repo root, e.g.:"
echo "    cd ${REPO_ROOT}"
echo "    python -m examples.train --algo mappo_alt --env mujoco_marl --exp_name my_run ..."
