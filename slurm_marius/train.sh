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
#SBATCH --mail-user=s-mtschi@haw-landshut.de
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/ADC"

START_TIME=$(date)
EXIT_CODE=0

# ── Preset selection ──
export PRESET=${PRESET:-all}

echo "Job $SLURM_JOB_ID on $(hostname) — $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Preset: $PRESET  |  Starting at $START_TIME"

if [[ "$PRESET" == "all" ]]; then
    TRAINING_TARGET=workstation uv run python run_all.py || EXIT_CODE=$?
else
    TRAINING_TARGET=workstation uv run python tutorial_train_single_gpu.py || EXIT_CODE=$?
fi

# ── Job summary (appears in .out file and SLURM email) ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  JOB SUMMARY — $SLURM_JOB_ID"
echo "═══════════════════════════════════════════════════════════"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo "  Started:  $START_TIME"
echo "  Finished: $(date)"
echo "  Exit:     $EXIT_CODE"
echo ""

# Show disk usage per preset
echo "  Disk usage:"
for d in runs/*/; do
    if [[ -d "$d" ]]; then
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        ckpts=$(find "$d" -name "*.ckpt" 2>/dev/null | wc -l | tr -d ' ')
        imgs=$(find "$d" -path "*/image_log/train/*.png" 2>/dev/null | wc -l | tr -d ' ')
        echo "    $(basename "$d"): ${size} (${ckpts} ckpts, ${imgs} images)"
    fi
done
echo ""

# Show training report summary if available
if [[ -f runs/training_report.md ]]; then
    echo "  Training report (first 25 lines):"
    head -25 runs/training_report.md | sed 's/^/    /'
fi

echo "═══════════════════════════════════════════════════════════"
exit $EXIT_CODE
