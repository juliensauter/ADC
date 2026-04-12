#!/bin/bash
# Train ADC on liver data. Requires setup.sh to have been run first.
# Usage: sbatch slurm/train.sh
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
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
cd "$HOME/ADC"

echo "Job $SLURM_JOB_ID on $(hostname) — $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Starting training at $(date)"

TRAINING_TARGET=dgx_single uv run python tutorial_train_single_gpu.py

echo "Done at $(date). Checkpoints: lightning_logs/"
