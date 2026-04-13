#!/bin/bash
# Submit all training presets as a SLURM dependency chain.
# Each preset runs after the previous one finishes (afterany = run even if prior failed).
#
# Usage:
#   bash slurm/train_all.sh              # submit all 3 presets sequentially
#   bash slurm/train_all.sh scratch polyp_transfer   # submit only selected presets
#
# To submit a single preset directly:
#   PRESET=polyp_transfer sbatch slurm/train.sh
set -euo pipefail

# Default preset sequence:
#   Phase 1 base presets → Phase 1 chain presets → Phase 2
# Chain presets depend on their source preset's checkpoint, so ordering matters.
if [[ $# -gt 0 ]]; then
    PRESETS=("$@")
else
    PRESETS=(scratch polyp_transfer scratch_unlocked polyp_unlocked polyp_stage2)
fi

echo "=== ADC train_all: submitting ${#PRESETS[@]} presets as dependency chain ==="
echo "Presets: ${PRESETS[*]}"
echo ""

PREV_JOB=""

for preset in "${PRESETS[@]}"; do
    if [[ -z "$PREV_JOB" ]]; then
        # First job — no dependency
        JOB_ID=$(sbatch --parsable --export=ALL,PRESET="$preset" slurm/train.sh)
    else
        # Chain: run after previous job finishes (regardless of exit code)
        JOB_ID=$(sbatch --parsable --dependency=afterany:"$PREV_JOB" --export=ALL,PRESET="$preset" slurm/train.sh)
    fi
    echo "  Submitted preset=$preset → Job $JOB_ID${PREV_JOB:+ (after $PREV_JOB)}"
    PREV_JOB="$JOB_ID"
done

echo ""
echo "All jobs submitted. Monitor with:  squeue -u \$USER"
echo "Cancel chain with:                 scancel $PREV_JOB  (cancels last; earlier jobs still run)"
