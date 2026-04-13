"""
tutorial_train_single_gpu.py
============================
Training script for ADC with two env-var switches:

  TRAINING_TARGET  — hardware target (mps | workstation | dgx_single | dgx_multi)
  PRESET           — training config  (scratch | polyp_transfer | polyp_unlocked)

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

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

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
#   PRESET=polyp_unlocked    → polyp transfer + unlock SD UNet decoder
# ──────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "scratch": {
        "ckpt_path": "./stable-diffusion-v1-5/control_sd15.ckpt",
        "strict_load": True,
        "sd_locked": True,         # only train ControlNets
        "lr": 1e-5,
        "max_steps": 20000,
        "desc": "SD v1.5 base → liver ControlNets from scratch",
    },
    "polyp_transfer": {
        "ckpt_path": "./adc_weights/merged_pytorch_model.pth",
        "strict_load": False,      # ADC polyp ckpt has different key names
        "sd_locked": True,         # only train ControlNets
        "lr": 1e-5,
        "max_steps": 20000,
        "desc": "ADC polyp weights → liver (closer medical domain)",
    },
    "polyp_unlocked": {
        "ckpt_path": "./adc_weights/merged_pytorch_model.pth",
        "strict_load": False,
        "sd_locked": False,        # also fine-tune SD UNet decoder (deeper adaptation)
        "lr": 5e-6,                # lower LR when training more params (avoids catastrophic forgetting)
        "max_steps": 10000,
        "desc": "ADC polyp + unlocked UNet decoder (risk: forgetting on <2k images)",
    },
}

PRESET_NAME = os.environ.get("PRESET", "scratch")
assert PRESET_NAME in PRESETS, f"Unknown PRESET={PRESET_NAME!r}. Options: {list(PRESETS.keys())}"
preset = PRESETS[PRESET_NAME]

CKPT_PATH    = preset["ckpt_path"]
STRICT_LOAD  = preset["strict_load"]
SD_LOCKED    = preset["sd_locked"]
LR           = preset["lr"]
MAX_STEPS    = int(os.environ.get('MAX_STEPS', str(preset["max_steps"])))
RESUME_PATH  = os.environ.get("RESUME_PATH", None)     # explicit override, else auto-detected
                                                        # Auto-detected below if runs/{preset}/checkpoints/last.ckpt exists
LOG_DIR      = f"runs/{PRESET_NAME}"                    # separate output dir per preset
ONLY_MID_CTRL = False

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
print(f"  ckpt:      {CKPT_PATH}")
print(f"  sd_locked: {SD_LOCKED}  |  lr: {LR}  |  max_steps: {MAX_STEPS}")
print(f"  log_dir:   {LOG_DIR}")
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
    BATCH_SIZE   = 2               # Conservative: workstation GPUs may have ≤16GB VRAM
    GRAD_ACCUM   = 2               # Effective batch = 2×2 = 4
    NUM_WORKERS  = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(not found)"
    print(f"\n[Workstation] CUDA GPU: {gpu_name}, precision={PRECISION}")

else:
    raise ValueError(f"Unknown TRAINING_TARGET: {TRAINING_TARGET!r}. "
                     "Choose 'mps', 'dgx_single', 'dgx_multi', or 'workstation'")

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(CKPT_PATH, location='cpu'), strict=STRICT_LOAD)

model.learning_rate  = LR
model.sd_locked      = SD_LOCKED
model.only_mid_control = ONLY_MID_CTRL

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
import glob
if RESUME_PATH is None:
    candidates = sorted(glob.glob(f'{LOG_DIR}/*/checkpoints/last.ckpt'))
    # Legacy path: old runs wrote to lightning_logs/ — only match for scratch preset
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
