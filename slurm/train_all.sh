#!/bin/bash
# Submit ALL training presets as a single SLURM job.
# Preferred method: runs all presets sequentially, autodetects completion.
#
# Usage:
#   bash slurm/train_all.sh              # submit all presets as one job
#   bash slurm/train_all.sh --split      # submit as separate dependent jobs
#
# The default (recommended) mode submits a single job that runs run_all.py,
# which handles sequencing, completion detection, and post-training analysis.
#
# The --split mode submits separate SLURM jobs with dependency chaining
# (each preset waits for the previous one to finish). Use this if you
# need separate job tracking per preset.
set -euo pipefail

if [[ "${1:-}" == "--split" ]]; then
    shift
    # Split mode: separate jobs with dependencies
    if [[ $# -gt 0 ]]; then
        PRESETS=("$@")
    else
        PRESETS=(scratch polyp_transfer scratch_unlocked polyp_unlocked polyp_stage2 scratch_stage2 polyp_stage2_from_unlocked)
    fi

    echo "=== ADC train_all (split mode): submitting ${#PRESETS[@]} presets ==="
    echo "Presets: ${PRESETS[*]}"
    echo ""

    PREV_JOB=""
    for preset in "${PRESETS[@]}"; do
        if [[ -z "$PREV_JOB" ]]; then
            JOB_ID=$(sbatch --parsable --export=ALL,PRESET="$preset" slurm/train.sh)
        else
            JOB_ID=$(sbatch --parsable --dependency=afterany:"$PREV_JOB" --export=ALL,PRESET="$preset" slurm/train.sh)
        fi
        echo "  Submitted preset=$preset → Job $JOB_ID${PREV_JOB:+ (after $PREV_JOB)}"
        PREV_JOB="$JOB_ID"
    done

    echo ""
    echo "All jobs submitted. Monitor with:  squeue -u \$USER"
else
    # Default: single job running all presets
    echo "=== ADC train_all: submitting single job for all presets ==="
    JOB_ID=$(sbatch --parsable --export=ALL,PRESET=all slurm/train.sh)
    echo "  Submitted job $JOB_ID — runs all presets sequentially"
    echo "  Monitor with:  squeue -u \$USER"
    echo "  Output:        adc_train_${JOB_ID}.out"
fi
