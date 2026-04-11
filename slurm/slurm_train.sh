#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# slurm_train.sh — ADC training on the LRZ SLURM cluster (DGX H100 partition)
# ══════════════════════════════════════════════════════════════════════════════
#
# Submits a training job for tutorial_train_single_gpu.py on the dgx_01
# partition (NVIDIA H100 GPUs).
#
# Usage:
#   sbatch slurm_train.sh                     # 1 GPU, default
#   sbatch --gres=gpu:2 slurm_train.sh        # 2 GPUs (overrides default)
#   sbatch --job-name=liver_v2 slurm_train.sh  # custom job name
#
# Before first run:
#   1. Clone repo to /mnt/home/<user>/ADC/
#   2. Run:  sbatch slurm_setup.sh   (installs deps + downloads weights)
#   3. Place your data in /mnt/home/<user>/ADC/data/
#   4. Submit:  sbatch slurm_train.sh
#
# Logs:    /mnt/home/<user>/ADC/logs/<jobname>_<jobid>.out
# Checkpoints: /mnt/home/<user>/ADC/lightning_logs/
# ══════════════════════════════════════════════════════════════════════════════

# ── SBATCH directives ────────────────────────────────────────────────────────
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=250G
#SBATCH --job-name=adc_train
#SBATCH --output=/mnt/home/%u/ADC/logs/%x_%j.out
#SBATCH --error=/mnt/home/%u/ADC/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julien.sauter@haw-landshut.de

# ── Setup ─────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJDIR="/mnt/home/${USER}/ADC"
VENVDIR="${PROJDIR}/.venv"
LOGDIR="${PROJDIR}/logs"

echo "════════════════════════════════════════════════════════════"
echo "  ADC Training Job"
echo "  User:       ${USER}"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Partition:  ${SLURM_JOB_PARTITION}"
echo "  GPUs:       ${SLURM_GPUS_ON_NODE:-1}"
echo "  Node:       $(hostname)"
echo "  Start:      $(date)"
echo "════════════════════════════════════════════════════════════"

# Create log directory (must exist before SLURM writes to it)
mkdir -p "${LOGDIR}"

cd "${PROJDIR}"

# ── Optional: use scratch for data I/O ────────────────────────────────────────
# Uncomment this block if training data is large and I/O-bound.
# Scratch is local SSD on the worker — much faster than NFS, but EPHEMERAL.
#
# SCRATCH="/scratch/${USER}_${SLURM_JOB_ID}"
# echo "Copying data to scratch: ${SCRATCH}"
# mkdir -p "${SCRATCH}/data"
# rsync -a "${PROJDIR}/data/" "${SCRATCH}/data/"
# # Point DATA_ROOT in the training script to scratch path,
# # or symlink: ln -sf "${SCRATCH}/data" "${PROJDIR}/data_scratch"

# ── Activate virtual environment ──────────────────────────────────────────────
if [ ! -d "${VENVDIR}" ]; then
    echo "ERROR: Virtual environment not found at ${VENVDIR}"
    echo "Run 'sbatch slurm_setup.sh' first to create it."
    exit 1
fi
source "${VENVDIR}/bin/activate"

# ── Verify GPU access ────────────────────────────────────────────────────────
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Determine training target ────────────────────────────────────────────────
GPU_COUNT="${SLURM_GPUS_ON_NODE:-1}"

if [ "${GPU_COUNT}" -gt 1 ]; then
    TRAINING_TARGET="dgx_multi"
else
    TRAINING_TARGET="dgx_single"
fi
echo "Training target: ${TRAINING_TARGET} (${GPU_COUNT} GPU(s))"

# ── Patch the training script's TRAINING_TARGET on the fly ────────────────────
# Instead of editing the file, we use sed to create a temporary patched copy.
# This way the original file is never modified.
TRAIN_SCRIPT="${PROJDIR}/tutorial_train_single_gpu.py"
PATCHED_SCRIPT="${PROJDIR}/.slurm_train_patched.py"

sed "s/^TRAINING_TARGET = .*/TRAINING_TARGET = \"${TRAINING_TARGET}\"/" \
    "${TRAIN_SCRIPT}" > "${PATCHED_SCRIPT}"

echo "Running patched training script: ${PATCHED_SCRIPT}"
echo ""

# ── Run training ──────────────────────────────────────────────────────────────
python "${PATCHED_SCRIPT}"

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm -f "${PATCHED_SCRIPT}"

# If using scratch, copy results back:
# rsync -a "${SCRATCH}/lightning_logs/" "${PROJDIR}/lightning_logs/"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Training complete"
echo "  End: $(date)"
echo "  Checkpoints: ${PROJDIR}/lightning_logs/"
echo "════════════════════════════════════════════════════════════"
