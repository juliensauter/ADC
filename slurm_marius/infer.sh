#!/bin/bash
# Run inference with a trained checkpoint.
# Usage: CKPT_PATH=./lightning_logs/.../last.ckpt sbatch slurm/infer.sh
#SBATCH --partition=workstations
#SBATCH --qos=students_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=adc_infer
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s-mtschi@haw-landshut.de
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/ADC"

echo "Inference job $SLURM_JOB_ID — $(date)"
uv run python tutorial_inference_local.py

echo "Done at $(date). Results: generated_results/"
