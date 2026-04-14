"""
tutorial_train_single_gpu.py
============================
Training script for ADC with two env-var switches:

  TRAINING_TARGET  — hardware target (mps | workstation | dgx_single | dgx_multi)
  PRESET           — training config  (scratch | polyp_transfer | scratch_unlocked |
                                       polyp_unlocked | polyp_stage2 | scratch_stage2 |
                                       polyp_stage2_from_unlocked)

Usage:
    # Local (MPS):
    uv run python tutorial_train_single_gpu.py

    # Cluster single preset:
    TRAINING_TARGET=workstation PRESET=polyp_transfer uv run python tutorial_train_single_gpu.py

    # Cluster via SLURM (single):
    PRESET=polyp_transfer sbatch slurm/train.sh

    # Cluster via SLURM (all presets chained):
    bash slurm/train_all.sh

Each preset stores output in runs/{preset_name}/ (checkpoints, images, metrics).
Auto-resumes from last.ckpt if found.  Override with RESUME_PATH env var.

Key differences vs original tutorial_train.py:
  - DeepSpeed removed → standard Lightning training
  - Gradient accumulation for simulating larger batch size on limited VRAM
  - Float32 enforced automatically for MPS
  - Lightning checkpoints are directly loadable with load_state_dict()
"""

import os
import sys

# Always run from ADC project directory (so relative paths like ./data work correctly)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from share import *

import glob
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def find_source_checkpoint(source_preset: str) -> str:
    """Find the latest last.ckpt for a source preset.

    Searches (in priority order):
      1. runs/{source_preset}/*/checkpoints/last.ckpt   (new preset system)
      2. lightning_logs/*/checkpoints/last.ckpt          (legacy, scratch only)

    Returns the path to the checkpoint, or raises FileNotFoundError.
    """
    candidates = sorted(glob.glob(f'runs/{source_preset}/*/checkpoints/last.ckpt'))
    if not candidates and source_preset == "scratch":
        candidates = sorted(glob.glob('lightning_logs/*/checkpoints/last.ckpt'))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for source preset '{source_preset}'.\n"
            f"  Searched: runs/{source_preset}/*/checkpoints/last.ckpt\n"
            f"  Has '{source_preset}' been trained? Run it first."
        )
    return candidates[-1]


# ──────────────────────────────────────────────────────────────────────────────
# ★ TWO SWITCHES — hardware target + training preset
#   Set via env vars:   TRAINING_TARGET=workstation PRESET=polyp_transfer uv run python ...
#   Or edit defaults below.
# ──────────────────────────────────────────────────────────────────────────────
TRAINING_TARGET = os.environ.get("TRAINING_TARGET", "mps")  # "mps" | "dgx_single" | "dgx_multi" | "workstation"

# ──────────────────────────────────────────────────────────────────────────────
# ★ TRAINING PRESETS — all config in one place for easy switching
#   Each preset defines: starting checkpoint, weights locking, learning rate, etc.
#   PRESET=scratch           → train ControlNets from SD v1.5 (baseline)
#   PRESET=polyp_transfer    → transfer from ADC polyp weights (closer domain)
#   PRESET=scratch_unlocked  → continue scratch + unlock last 3 decoder blocks
#   PRESET=polyp_unlocked    → continue polyp_transfer + unlock last 3 decoder blocks
#   PRESET=polyp_stage2      → full ADC Phase 2: image CN + distillation + decoder
#   PRESET=scratch_stage2    → Phase 2 from scratch path
#   PRESET=polyp_stage2_from_unlocked → Phase 2 from polyp progressive unfreezing
#
# Chain presets use "$source_preset" as ckpt_path — resolved at runtime via
# find_source_checkpoint().  This auto-finds the latest checkpoint regardless
# of whether it's in runs/ or legacy lightning_logs/.
# ──────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "scratch": {
        "ckpt_path": "./stable-diffusion-v1-5/control_sd15.ckpt",
        "strict_load": True,
        "sd_locked": True,         # only train ControlNets
        "unlock_last_n": 0,        # decoder fully frozen
        "train_image_cn": False,   # image CN unused in Phase 1 — save it from weight decay
        "image_loss": 0.0,         # Phase 1: mask-only
        "distill_loss": 0.0,
        "decoder_lr_scale": 0.1,
        "lr": 1e-5,
        "max_steps": 20000,
        "desc": "SD v1.5 base → liver ControlNets from scratch",
    },
    "polyp_transfer": {
        "ckpt_path": "./adc_weights/merged_pytorch_model.pth",
        "strict_load": False,      # ADC polyp ckpt has different key names
        "sd_locked": True,
        "unlock_last_n": 0,
        "train_image_cn": False,
        "image_loss": 0.0,
        "distill_loss": 0.0,
        "decoder_lr_scale": 0.1,
        "lr": 1e-5,
        "max_steps": 20000,
        "desc": "ADC polyp weights → liver (closer medical domain)",
    },
    "scratch_unlocked": {
        "ckpt_path": "$scratch",           # auto-resolved via find_source_checkpoint()
        "strict_load": False,      # Lightning ckpt format
        "sd_locked": False,        # unlock decoder
        "unlock_last_n": 3,        # only last 3 blocks (37.6M params, fine texture)
        "train_image_cn": False,   # still Phase 1
        "image_loss": 0.0,
        "distill_loss": 0.0,
        "decoder_lr_scale": 0.1,   # decoder LR = 0.1 × base LR
        "lr": 5e-6,
        "max_steps": 10000,
        "desc": "Scratch 20k → unlock last 3 decoder blocks (progressive unfreezing)",
    },
    "polyp_unlocked": {
        "ckpt_path": "$polyp_transfer",    # auto-resolved
        "strict_load": False,
        "sd_locked": False,
        "unlock_last_n": 3,
        "train_image_cn": False,
        "image_loss": 0.0,
        "distill_loss": 0.0,
        "decoder_lr_scale": 0.1,
        "lr": 5e-6,
        "max_steps": 10000,
        "desc": "Polyp transfer 20k → unlock last 3 decoder blocks",
    },
    "polyp_stage2": {
        "ckpt_path": "$polyp_transfer",    # auto-resolved — Phase 2 directly from Phase 1
        "strict_load": False,
        "sd_locked": False,
        "unlock_last_n": 3,
        "train_mask_cn": False,    # Phase 2: mask CN frozen (teacher)
        "train_image_cn": True,    # Phase 2: enable image ControlNet training
        "image_loss": 1.0,         # image decoder denoising loss
        "distill_loss": 0.5,       # anatomy-aware distillation (mask→image)
        "decoder_lr_scale": 0.1,
        "lr": 5e-6,
        "max_steps": 10000,
        "desc": "Full ADC Phase 2: image CN + distillation (mask CN frozen as teacher)",
    },
    "scratch_stage2": {
        "ckpt_path": "$scratch_unlocked",  # auto-resolved — Phase 2 from unfrozen scratch
        "strict_load": False,
        "sd_locked": False,
        "unlock_last_n": 3,
        "train_mask_cn": False,
        "train_image_cn": True,
        "image_loss": 1.0,
        "distill_loss": 0.5,
        "decoder_lr_scale": 0.1,
        "lr": 5e-6,
        "max_steps": 10000,
        "desc": "Scratch path Phase 2: image CN + distillation (mask CN frozen)",
    },
    "polyp_stage2_from_unlocked": {
        "ckpt_path": "$polyp_unlocked",    # auto-resolved — Phase 2 from progressive unfreeze
        "strict_load": False,
        "sd_locked": False,
        "unlock_last_n": 3,
        "train_mask_cn": False,
        "train_image_cn": True,
        "image_loss": 1.0,
        "distill_loss": 0.5,
        "decoder_lr_scale": 0.1,
        "lr": 5e-6,
        "max_steps": 10000,
        "desc": "Phase 2 from progressively unfrozen polyp weights (mask CN frozen)",
    },
}

PRESET_NAME = os.environ.get("PRESET", "scratch")
assert PRESET_NAME in PRESETS, f"Unknown PRESET={PRESET_NAME!r}. Options: {list(PRESETS.keys())}"
preset = PRESETS[PRESET_NAME]

# Resolve $source_preset markers → actual checkpoint paths
CKPT_PATH = preset["ckpt_path"]
if CKPT_PATH.startswith("$"):
    source_preset = CKPT_PATH[1:]
    CKPT_PATH = find_source_checkpoint(source_preset)
    print(f"  Resolved ${source_preset} → {CKPT_PATH}")
STRICT_LOAD  = preset["strict_load"]
SD_LOCKED    = preset["sd_locked"]
LR           = preset["lr"]
MAX_STEPS    = int(os.environ.get('MAX_STEPS', str(preset["max_steps"])))
RESUME_PATH  = os.environ.get("RESUME_PATH", None)
LOG_DIR      = f"runs/{PRESET_NAME}"
ONLY_MID_CTRL = False

# New preset parameters for fine-grained control
UNLOCK_LAST_N   = preset.get("unlock_last_n", 0)
TRAIN_MASK_CN   = preset.get("train_mask_cn", True)
TRAIN_IMAGE_CN  = preset.get("train_image_cn", True)
IMAGE_LOSS      = preset.get("image_loss", 0.0)
DISTILL_LOSS    = preset.get("distill_loss", 0.0)
DECODER_LR_SCALE = preset.get("decoder_lr_scale", 0.1)

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
pl.seed_everything(42, workers=True)

# ──────────────────────────────────────────────────────────────────────────────
# Data config — set DATA_ROOT to your prepared liver data folder
# Run: uv run python prepare_liver_data.py --src /path/to/raw --out ./data
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT    = './data/train/prompt.json'   # train split
LOGGER_FREQ  = 400

print(f"\n{'='*60}")
print(f"  PRESET: {PRESET_NAME}")
print(f"  {preset['desc']}")
print(f"  ckpt:       {CKPT_PATH}")
print(f"  sd_locked:  {SD_LOCKED}  |  unlock_last_n: {UNLOCK_LAST_N}")
print(f"  mask_cn:    {TRAIN_MASK_CN}  |  image_cn: {TRAIN_IMAGE_CN}")
print(f"  lr: {LR}  |  decoder_lr: {LR * DECODER_LR_SCALE}  |  max_steps: {MAX_STEPS}")
print(f"  image_loss: {IMAGE_LOSS}  |  distill: {DISTILL_LOSS}")
print(f"  log_dir:    {LOG_DIR}")
print(f"{'='*60}")

# ──────────────────────────────────────────────────────────────────────────────
# Hardware-specific settings derived from TRAINING_TARGET
# ──────────────────────────────────────────────────────────────────────────────
if TRAINING_TARGET == "mps":
    # Apple Silicon — MPS backend, float32 required, small batch
    ACCELERATOR  = "mps"
    DEVICES      = 1
    PRECISION    = "32"       # MPS does not support float16 reliably
    BATCH_SIZE   = 1          # MPS has limited shared memory; keep small
    GRAD_ACCUM   = 4          # Effective batch = 1×4 = 4
    NUM_WORKERS  = 0          # DataLoader workers must be 0 on MPS
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    print("\n[MPS] Apple Silicon — slow training, use for debugging only")

elif TRAINING_TARGET == "dgx_single":
    # Single GPU on DGX station (A100 40/80GB)
    ACCELERATOR  = "gpu"
    DEVICES      = 1
    PRECISION    = "bf16-mixed"   # Best on A100; change to "16-mixed" on V100
    BATCH_SIZE   = 4
    GRAD_ACCUM   = 1          # Effective batch = 4
    NUM_WORKERS  = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Use first GPU
    print(f"\n[DGX single] CUDA GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(not found)'}")

elif TRAINING_TARGET == "dgx_multi":
    # Multi-GPU on DGX station — uses all GPUs visible to the process
    # Set CUDA_VISIBLE_DEVICES before launching, e.g.:
    #   CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python tutorial_train_single_gpu.py
    ACCELERATOR  = "gpu"
    DEVICES      = torch.cuda.device_count() if torch.cuda.is_available() else 1
    PRECISION    = "bf16-mixed"
    BATCH_SIZE   = 4
    GRAD_ACCUM   = 1          # Effective batch = 4 × DEVICES
    NUM_WORKERS  = 4
    print(f"\n[DGX multi] {DEVICES} CUDA GPUs")

elif TRAINING_TARGET == "workstation":
    # Single-GPU workstation (unknown GPU — auto-detect bf16 support)
    ACCELERATOR  = "gpu"
    DEVICES      = 1
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        PRECISION = "bf16-mixed"   # Ampere+ (A100, RTX 30xx, RTX 40xx)
    else:
        PRECISION = "16-mixed"     # Older GPU (V100, RTX 20xx, etc.)
    # Phase 2 trains image CN (~363M extra params) → needs ~1.4GB more VRAM for optimizer
    # states. Reduce batch to 1 and compensate with higher gradient accumulation.
    if TRAIN_IMAGE_CN:
        BATCH_SIZE = 1             # Phase 2: tight VRAM budget
        GRAD_ACCUM = 4             # Effective batch = 1×4 = 4
    else:
        BATCH_SIZE = 2             # Phase 1/1b: fits comfortably
        GRAD_ACCUM = 2             # Effective batch = 2×2 = 4
    NUM_WORKERS  = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(not found)"
    print(f"\n[Workstation] CUDA GPU: {gpu_name}, precision={PRECISION}, batch={BATCH_SIZE}")

else:
    raise ValueError(f"Unknown TRAINING_TARGET: {TRAINING_TARGET!r}. "
                     "Choose 'mps', 'dgx_single', 'dgx_multi', or 'workstation'")

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(CKPT_PATH, location='cpu'), strict=STRICT_LOAD)

model.learning_rate    = LR
model.sd_locked        = SD_LOCKED
model.only_mid_control = ONLY_MID_CTRL

# Fine-grained training control (read by configure_optimizers / p_losses in cldm.py)
model.unlock_last_n       = UNLOCK_LAST_N
model.train_mask_cn       = TRAIN_MASK_CN
model.train_image_cn      = TRAIN_IMAGE_CN

# Freeze mask CN parameters entirely when not training (saves ~1.4GB gradient memory)
if not TRAIN_MASK_CN:
    model.control_model.requires_grad_(False)
model.decoder_lr_scale    = DECODER_LR_SCALE
model.loss_weight_mask    = 1.0
model.loss_weight_image   = IMAGE_LOSS
model.loss_weight_distill = DISTILL_LOSS

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
dataset    = MyDataset(root=DATA_ROOT)
dataloader = DataLoader(dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True)

print(f"\nDataset: {len(dataset)} samples, batch_size={BATCH_SIZE}, "
      f"effective_bs={BATCH_SIZE * GRAD_ACCUM}, steps={MAX_STEPS}")

# ──────────────────────────────────────────────────────────────────────────────
# Callbacks & Logger
# ──────────────────────────────────────────────────────────────────────────────
logger_cb = ImageLogger(batch_frequency=LOGGER_FREQ)
ckpt_cb = pl.callbacks.ModelCheckpoint(
    every_n_train_steps=LOGGER_FREQ * 5,   # save every 2000 steps (5× image log freq)
    save_last=True,                # always keep last.ckpt (even if killed mid-epoch)
    save_top_k=1,                  # keep latest periodic ckpt + last.ckpt (~18 GB total)
)

# ──────────────────────────────────────────────────────────────────────────────
# Auto-resume: find latest last.ckpt in this preset's log dir
# ──────────────────────────────────────────────────────────────────────────────
if RESUME_PATH is None:
    candidates = sorted(glob.glob(f'{LOG_DIR}/*/checkpoints/last.ckpt'))
    # Legacy path: old runs wrote to lightning_logs/ — only check for scratch preset
    if not candidates and PRESET_NAME == "scratch":
        candidates = sorted(glob.glob('lightning_logs/*/checkpoints/last.ckpt'))
    if candidates:
        RESUME_PATH = candidates[-1]
        print(f"\nAuto-resume: found {RESUME_PATH}")

# ──────────────────────────────────────────────────────────────────────────────
# Sanity check: run 2 steps + 1 image log to verify setup (skip on resume)
# ──────────────────────────────────────────────────────────────────────────────
if RESUME_PATH is None:
    print("\n── Sanity check: 2 training steps + image generation ──")
    sanity_logger = ImageLogger(batch_frequency=1, log_first_step=True)
    sanity_csv = pl.loggers.CSVLogger(save_dir=LOG_DIR, name="sanity")
    sanity_trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        logger=sanity_csv,
        callbacks=[sanity_logger],
        max_steps=2,
        accumulate_grad_batches=GRAD_ACCUM,
        precision=PRECISION,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    sanity_trainer.fit(model, dataloader)
    print("── Sanity check passed ✓ ──\n")
else:
    print(f"\nSkipping sanity check (resuming from checkpoint)")

# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────
csv_logger = pl.loggers.CSVLogger(save_dir=LOG_DIR, name="")

trainer = pl.Trainer(
    accelerator=ACCELERATOR,
    devices=DEVICES,
    logger=csv_logger,
    callbacks=[logger_cb, ckpt_cb],
    max_steps=MAX_STEPS,
    accumulate_grad_batches=GRAD_ACCUM,
    precision=PRECISION,
    gradient_clip_val=1.0,
    log_every_n_steps=50,
    enable_checkpointing=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────
if RESUME_PATH:
    print(f"\nResuming from: {RESUME_PATH}")
    trainer.fit(model, dataloader, ckpt_path=RESUME_PATH)
else:
    print(f"\nStarting training from: {CKPT_PATH}")
    trainer.fit(model, dataloader)

print(f"\nTraining complete.  [preset={PRESET_NAME}]")
print(f"Checkpoints saved in: {LOG_DIR}/")
print(f"Load checkpoint with: model.load_state_dict(load_state_dict('{LOG_DIR}/.../last.ckpt'))")
print(f"Images saved in:      {LOG_DIR}/image_log/")
