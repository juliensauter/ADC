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
cd "$HOME/ADC"

export PRESET=${PRESET:-scratch}
export TRAINING_TARGET=${TRAINING_TARGET:-workstation}
export NUM_FOLDS=${NUM_FOLDS:-5}

ARGS=()
if [[ -n "${FOLDS:-}" ]]; then
    ARGS+=(--folds "$FOLDS")
fi

echo "K-fold job $SLURM_JOB_ID on $(hostname)"
echo "Preset: $PRESET"
echo "Training target: $TRAINING_TARGET"
echo "Num folds: $NUM_FOLDS"
if [[ -n "${FOLDS:-}" ]]; then
    echo "Requested folds: $FOLDS"
fi

uv run python run_kfold.py --preset "$PRESET" --training-target "$TRAINING_TARGET" --num-folds "$NUM_FOLDS" "${ARGS[@]}"

echo "Done at $(date)."
