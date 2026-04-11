#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# slurm_setup.sh — One-time ADC environment setup on the LRZ SLURM cluster
# ══════════════════════════════════════════════════════════════════════════════
#
# Creates a persistent virtual environment in /mnt/home/<user>/ADC/.venv
# and downloads all model weights. Run this ONCE before training/inference.
#
# Usage:
#   sbatch slurm_setup.sh                        # full setup
#   sbatch slurm_setup.sh --weights-only          # skip deps
#   sbatch slurm_setup.sh --deps-only             # skip weights
#   sbatch slurm_setup.sh --no-control-ckpt       # skip control_sd15.ckpt
#
# The .venv is on NFS (/mnt/home/) and persists across jobs — subsequent
# training/inference jobs just activate it. Delete with:
#   rm -rf /mnt/home/<user>/ADC/.venv
#
# WARNING: This downloads ~17 GB of weights and creates ~35 GB on disk.
#          Make sure you have enough NFS quota.
# ══════════════════════════════════════════════════════════════════════════════

# ── SBATCH directives ────────────────────────────────────────────────────────
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=adc_setup
#SBATCH --output=/mnt/home/%u/ADC/logs/%x_%j.out
#SBATCH --error=/mnt/home/%u/ADC/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julien.sauter@haw-landshut.de

# ── Setup ─────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJDIR="/mnt/home/${USER}/ADC"
VENVDIR="${PROJDIR}/.venv"

# Pass any extra arguments to setup_adc.py (e.g. --weights-only, --deps-only)
SETUP_ARGS="${@}"

echo "════════════════════════════════════════════════════════════"
echo "  ADC Cluster Setup"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Node:    $(hostname)"
echo "  Args:    ${SETUP_ARGS:-<none>}"
echo "  Start:   $(date)"
echo "════════════════════════════════════════════════════════════"

mkdir -p "${PROJDIR}/logs"
cd "${PROJDIR}"

# ── Install uv if not available ───────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! command -v uv &> /dev/null; then
    echo ""
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | bash
    export PATH="${HOME}/.cargo/bin:${PATH}"
    echo "uv installed: $(uv --version)"
fi

# ── Create persistent virtual environment ─────────────────────────────────────
if [ ! -d "${VENVDIR}" ]; then
    echo ""
    echo "Creating virtual environment at ${VENVDIR}..."
    uv venv "${VENVDIR}"
fi
source "${VENVDIR}/bin/activate"

echo ""
echo "Python:  $(python --version)"
echo "uv:      $(uv --version)"
echo ""

# ── Verify GPU is accessible (needed for control_sd15.ckpt creation) ──────────
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU — weight download still works)"
echo ""

# ── Run setup_adc.py ──────────────────────────────────────────────────────────
# setup_adc.py handles:
#   1. Installing dependencies via uv
#   2. Downloading SD v1.5 weights (~7.7 GB)
#   3. Downloading ADC pretrained weights (~9.6 GB)
#   4. Creating control_sd15.ckpt from SD v1.5 (~9 GB)
echo "Running setup_adc.py ${SETUP_ARGS}..."
echo ""

python setup_adc.py ${SETUP_ARGS}

# ── Verify installation ──────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'  torch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import pytorch_lightning
print(f'  pytorch-lightning {pytorch_lightning.__version__}')
print('  ✓ All imports successful')
"

echo ""
echo "Checking weights..."
for f in stable-diffusion-v1-5/v1-5-pruned.ckpt adc_weights/merged_pytorch_model.pth; do
    if [ -f "$f" ]; then
        echo "  ✓ $f ($(du -sh "$f" | cut -f1))"
    else
        echo "  ✗ $f MISSING"
    fi
done

CTRL="stable-diffusion-v1-5/control_sd15.ckpt"
if [ -f "$CTRL" ]; then
    echo "  ✓ $CTRL ($(du -sh "$CTRL" | cut -f1))"
else
    echo "  ○ $CTRL not created (use --no-control-ckpt was passed, or it failed)"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo "  End:  $(date)"
echo ""
echo "  Next steps:"
echo "    1. Place your data in ${PROJDIR}/data/"
echo "       (images/, masks/, prompt.json)"
echo "    2. Submit training:  sbatch slurm_train.sh"
echo "    3. Submit inference: sbatch slurm_inference.sh"
echo "════════════════════════════════════════════════════════════"
