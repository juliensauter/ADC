#!/bin/bash
# Train ADC on liver data. Requires setup.sh to have been run first.
#
# Usage:
#   PRESET=scratch sbatch slurm/train.sh          # single preset
#   PRESET=all sbatch slurm/train.sh              # all presets, autodetect completion
#   sbatch slurm/train.sh                         # default: all presets
#
#SBATCH --partition=workstations
#SBATCH --qos=students_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=adc_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s-jsaute@haw-landshut.de
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/ADC"

# ── Preset selection ──
export PRESET=${PRESET:-all}

echo "Job $SLURM_JOB_ID on $(hostname) — $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Preset: $PRESET  |  Starting at $(date)"

if [[ "$PRESET" == "all" ]]; then
    # Run all presets sequentially — autodetects completion, skips done presets
    TRAINING_TARGET=workstation uv run python run_all.py
else
    # Run a single preset
    TRAINING_TARGET=workstation uv run python tutorial_train_single_gpu.py
fi

echo "Done at $(date). Preset: $PRESET  |  Output: runs/"
