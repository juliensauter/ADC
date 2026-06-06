#!/bin/bash
# Train one ADC preset across k folds on SLURM.
#
# Usage:
#   PRESET=scratch sbatch slurm/train_kfold.sh
#   PRESET=paper_faithful_polyp NUM_FOLDS=5 sbatch slurm/train_kfold.sh
#   PRESET=scratch FOLDS=0,2,4 sbatch slurm/train_kfold.sh

#SBATCH --partition=workstations
#SBATCH --qos=students_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=adc_kfold
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s-mtschi@haw-landshut.de
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resolve project dir in a robust way:
# 1) PROJECT_DIR override, 2) sbatch submit dir, 3) legacy $HOME/ADC fallback.
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/ADC}}
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "[ERROR] PROJECT_DIR does not exist: $PROJECT_DIR"
    exit 2
fi
cd "$PROJECT_DIR"

if [[ ! -f "run_kfold.py" ]]; then
    echo "[ERROR] run_kfold.py not found in $PROJECT_DIR"
    echo "        Submit from ADC repo root or set PROJECT_DIR explicitly."
    exit 2
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found, installing for user..."
    curl -LsSf https://astral.sh/uv/install.sh | bash
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv &>/dev/null; then
    echo "[ERROR] uv is still unavailable after install attempt."
    exit 2
fi

export PRESET=${PRESET:-scratch}
export TRAINING_TARGET=${TRAINING_TARGET:-workstation}
export NUM_FOLDS=${NUM_FOLDS:-5}

ARGS=()
if [[ -n "${FOLDS:-}" ]]; then
    ARGS+=(--folds "$FOLDS")
fi

echo "K-fold job $SLURM_JOB_ID on $(hostname)"
echo "Project dir: $PROJECT_DIR"
echo "Working dir: $(pwd)"
echo "uv version: $(uv --version)"
echo "Python: $(uv run python -c 'import sys; print(sys.executable)')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo unavailable)"
echo "Preset: $PRESET"
echo "Training target: $TRAINING_TARGET"
echo "Num folds: $NUM_FOLDS"
if [[ -n "${FOLDS:-}" ]]; then
    echo "Requested folds: $FOLDS"
fi

uv run python run_kfold.py --preset "$PRESET" --training-target "$TRAINING_TARGET" --num-folds "$NUM_FOLDS" "${ARGS[@]}"

echo "Done at $(date)."
