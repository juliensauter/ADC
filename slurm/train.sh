#!/bin/bash
# Train ADC on liver data. Requires setup.sh to have been run first.
# Usage: sbatch slurm/train.sh
#SBATCH --partition=workstations
#SBATCH --qos=students_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=adc_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s-jsaute@haw-landshut.de
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/ADC"

# ── Preset selection (override with:  PRESET=polyp_transfer sbatch slurm/train.sh) ──
export PRESET=${PRESET:-scratch}

echo "Job $SLURM_JOB_ID on $(hostname) — $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Preset: $PRESET  |  Starting training at $(date)"

TRAINING_TARGET=workstation uv run python tutorial_train_single_gpu.py

echo "Done at $(date). Preset: $PRESET  |  Checkpoints: runs/$PRESET/"
