#!/bin/bash
# Run inference with a trained checkpoint.
# Usage: CKPT=./lightning_logs/.../epoch=0-step=3000.ckpt sbatch slurm/infer.sh
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=adc_infer
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julien.sauter@haw-landshut.de
set -euo pipefail
cd "$HOME/ADC"
source .venv/bin/activate

echo "Inference job $SLURM_JOB_ID — $(date)"
python tutorial_inference_local.py

echo "Done at $(date). Results: generated_results/"
