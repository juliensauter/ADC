"""
tutorial_train_single_gpu.py
============================
Training script for ADC. Works on Apple Silicon MPS (local) and a DGX CUDA cluster.

ONE-LINE SWITCH:
    For local MPS testing:  TRAINING_TARGET = "mps"
    For DGX single GPU:     TRAINING_TARGET = "dgx_single"
    For DGX multi-GPU:      TRAINING_TARGET = "dgx_multi"   # uses all visible GPUs

Usage:
    uv run python tutorial_train_single_gpu.py

Key differences vs original tutorial_train.py:
  - DeepSpeed removed → standard Lightning training
  - Single config variable TRAINING_TARGET controls everything
  - Gradient accumulation for simulating larger batch size on limited VRAM
  - Float32 enforced automatically for MPS
  - No zero_to_fp32.py or tool_merge_control.py needed after training —
    Lightning checkpoints are directly loadable with load_state_dict()

Resume training:
    Set RESUME_PATH to a .ckpt path, or None to start fresh.
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
# ★ ONE-LINE SWITCH — change this to adapt to your hardware
#   Can also be set via env var: TRAINING_TARGET=dgx_single uv run python ...
# ──────────────────────────────────────────────────────────────────────────────
TRAINING_TARGET = os.environ.get("TRAINING_TARGET", "mps")  # "mps" | "dgx_single" | "dgx_multi"

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
pl.seed_everything(42, workers=True)

# ──────────────────────────────────────────────────────────────────────────────
# Shared training config (same for all targets)
# ──────────────────────────────────────────────────────────────────────────────
CKPT_PATH    = './stable-diffusion-v1-5/control_sd15.ckpt'   # fresh start from SD v1.5
# CKPT_PATH  = './adc_weights/merged_pytorch_model.pth'        # start from ADC polyp weights (transfer)
# NOTE: When using ADC polyp weights for transfer learning, also set STRICT_LOAD = False below.
RESUME_PATH  = None          # Set to .ckpt path to resume, else None

# ──────────────────────────────────────────────────────────────────────────────
# Data config — set DATA_ROOT to your prepared liver data folder
# Run: uv run python prepare_liver_data.py --src /path/to/raw --out ./data
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT    = './data/train/prompt.json'   # train split (default for prepare_liver_data.py)
# DATA_ROOT  = './data/prompt.json'          # combined (⚠ includes test — only for quick demos)

LOGGER_FREQ  = 400
LR           = 1e-5
MAX_STEPS    = 3000          # 1000 for quick domain tests, 3000 for full training
SD_LOCKED    = True          # True = only train ControlNets (saves memory, avoids forgetting)
                             # False = also fine-tune SD UNet decoder (risk of forgetting on <2k images)
ONLY_MID_CTRL = False
STRICT_LOAD  = True          # False when loading ADC polyp weights (key mismatch ok)

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

else:
    raise ValueError(f"Unknown TRAINING_TARGET: {TRAINING_TARGET!r}. "
                     "Choose 'mps', 'dgx_single', or 'dgx_multi'")

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
    every_n_train_steps=LOGGER_FREQ,   # save at same frequency as image logging
    save_last=True,                # always keep last.ckpt (even if killed mid-epoch)
    save_top_k=-1,                 # keep all checkpoints (don't delete old ones)
)

# ──────────────────────────────────────────────────────────────────────────────
# Sanity check: run 2 steps + 1 image log to verify setup, then continue
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Sanity check: 2 training steps + image generation ──")
sanity_logger = ImageLogger(batch_frequency=1, log_first_step=True)
sanity_trainer = pl.Trainer(
    accelerator=ACCELERATOR,
    devices=DEVICES,
    callbacks=[sanity_logger],
    max_steps=2,
    accumulate_grad_batches=GRAD_ACCUM,
    precision=PRECISION,
    log_every_n_steps=1,
    enable_checkpointing=False,
)
sanity_trainer.fit(model, dataloader)
print("── Sanity check passed ✓ ──\n")

# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────
trainer = pl.Trainer(
    accelerator=ACCELERATOR,
    devices=DEVICES,
    callbacks=[logger_cb, ckpt_cb],
    max_steps=MAX_STEPS,
    accumulate_grad_batches=GRAD_ACCUM,
    precision=PRECISION,
    # Logging
    log_every_n_steps=50,
    # Checkpointing: saves every epoch (~400 steps)
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

print("\nTraining complete.")
print("Checkpoints saved in: lightning_logs/version_N/checkpoints/")
print("Load checkpoint with: model.load_state_dict(load_state_dict('path/to/epoch-step.ckpt'))")
