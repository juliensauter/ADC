#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# slurm_inference.sh — ADC inference on the LRZ SLURM cluster
# ══════════════════════════════════════════════════════════════════════════════
#
# Generates synthetic images from segmentation masks using a trained ADC model.
# Single GPU, short job — typically finishes in minutes.
#
# Usage:
#   sbatch slurm_inference.sh
#   sbatch --gres=gpu:1 slurm_inference.sh
#
# Configure checkpoint and DDIM steps via environment variables:
#   CKPT=./lightning_logs/version_0/checkpoints/epoch=0-step=3000.ckpt \
#   DDIM_STEPS=50 sbatch slurm_inference.sh
#
# Results: /mnt/home/<user>/ADC/generated_results/
# ══════════════════════════════════════════════════════════════════════════════

# ── SBATCH directives ────────────────────────────────────────────────────────
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=adc_infer
#SBATCH --output=/mnt/home/%u/ADC/logs/%x_%j.out
#SBATCH --error=/mnt/home/%u/ADC/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julien.sauter@haw-landshut.de

# ── Setup ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# Cleanup patched scripts on exit (success or failure)
cleanup() { rm -f "${PROJDIR:-/tmp}/.slurm_infer_patched.py"; }
trap cleanup EXIT

PROJDIR="/mnt/home/${USER}/ADC"
VENVDIR="${PROJDIR}/.venv"

echo "════════════════════════════════════════════════════════════"
echo "  ADC Inference Job"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Node:    $(hostname)"
echo "  Start:   $(date)"
echo "════════════════════════════════════════════════════════════"

mkdir -p "${PROJDIR}/logs"
cd "${PROJDIR}"

# ── Activate virtual environment ──────────────────────────────────────────────
if [ ! -d "${VENVDIR}" ]; then
    echo "ERROR: Virtual environment not found at ${VENVDIR}"
    echo "Run 'sbatch slurm_setup.sh' first."
    exit 1
fi
source "${VENVDIR}/bin/activate"

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Configure via env vars (with defaults) ────────────────────────────────────
CKPT="${CKPT:-./adc_weights/merged_pytorch_model.pth}"
DDIM_STEPS="${DDIM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-9.0}"
RESULT_DIR="${RESULT_DIR:-./generated_results/slurm_${SLURM_JOB_ID}/}"

echo "Checkpoint:  ${CKPT}"
echo "DDIM steps:  ${DDIM_STEPS}"
echo "CFG scale:   ${CFG_SCALE}"
echo "Output dir:  ${RESULT_DIR}"

# ── Create a patched inference script with the chosen settings ────────────────
INFER_SCRIPT="${PROJDIR}/tutorial_inference_local.py"
PATCHED="${PROJDIR}/.slurm_infer_patched.py"

sed \
    -e "s|^CKPT_PATH = .*|CKPT_PATH = \"${CKPT}\"|" \
    -e "s|^DDIM_STEPS = .*|DDIM_STEPS = ${DDIM_STEPS}|" \
    -e "s|^CFG_SCALE = .*|CFG_SCALE = ${CFG_SCALE}|" \
    -e "s|^RESULT_DIR = .*|RESULT_DIR = \"${RESULT_DIR}\"|" \
    "${INFER_SCRIPT}" > "${PATCHED}"

echo ""
echo "Running inference..."
python "${PATCHED}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Inference complete"
echo "  End:     $(date)"
echo "  Results: ${RESULT_DIR}"
echo "════════════════════════════════════════════════════════════"
