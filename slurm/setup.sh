#!/bin/bash
# One-time setup: install deps, download weights, create control checkpoint.
# Usage: sbatch slurm/setup.sh
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=adc_setup
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s-jsaute@haw-landshut.de
set -euo pipefail
cd "$HOME/ADC"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | bash
    export PATH="$HOME/.cargo/bin:$PATH"
fi

uv run python setup_adc.py "$@"

echo "Setup done. Next: sbatch slurm/train.sh"
