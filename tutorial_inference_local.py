"""
tutorial_inference_local.py
============================
MPS/CPU-compatible inference script for ADC.
Adapted from tutorial_inference.py — removes CUDA/DeepSpeed hardcoding.

Works on:
  - Apple Silicon (MPS)
  - CPU (slow but functional)
  - Single CUDA GPU

Usage:
    uv run python tutorial_inference_local.py

Notes:
  - Set CKPT_PATH to your merged checkpoint (merged_pytorch_model.pth) or
    to 'stable-diffusion-v1-5/control_sd15.ckpt' for the base ADC init.
  - Data folder must have data/prompt.json and corresponding images/masks.
  - Results saved to RESULT_DIR.
"""

import os
import sys

# Always run from the ADC project directory regardless of how the script is invoked
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import torch
import random
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from share import *
from tutorial_dataset_sample import MyDataset
from cldm.model import create_model, load_state_dict

# ──────────────────────────────────────────────────────────────────────────────
# Device selection — MPS / CUDA / CPU
# ──────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (slow)")
    return device

DEVICE = get_device()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
pl.seed_everything(0, workers=True)

BATCH_SIZE = 1
# Pretrained ADC polyp weights (downloaded from HuggingFace SylarQ/ADC)
# Override via env: CKPT_PATH=./lightning_logs/.../epoch=0-step=3000.ckpt uv run python ...
import glob as _glob
_trained = sorted(_glob.glob("lightning_logs/*/checkpoints/last.ckpt"))
_default_ckpt = _trained[-1] if _trained else "./adc_weights/merged_pytorch_model.pth"
CKPT_PATH = os.environ.get("CKPT_PATH", _default_ckpt)
# For a fresh SD v1.5 init (before finetuning):
# CKPT_PATH = "./stable-diffusion-v1-5/control_sd15.ckpt"

RESULT_DIR = os.environ.get("RESULT_DIR", "./generated_results/local_test/")
os.makedirs(RESULT_DIR, exist_ok=True)

DDIM_STEPS = 10  # 10 for quick demo (~2-4 min on M4 MPS); use 50 for quality
DDIM_ETA = 0.0                    # 0 = deterministic, 1 = stochastic
CFG_SCALE = 9.0                   # Classifier-Free Guidance scale

learning_rate = 1e-5
sd_locked = False
only_mid_control = False

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def get_model():
    model = create_model('./models/cldm_v15.yaml').cpu()

    # MPS does not handle float16 reliably — force float32
    if DEVICE.type in ("mps", "cpu"):
        model = model.float()

    model.load_state_dict(load_state_dict(CKPT_PATH, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.to(DEVICE)
    model.eval()
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Image saving
# ──────────────────────────────────────────────────────────────────────────────
def log_local(save_dir, images, batch_idx):
    samples_dir = os.path.join(save_dir, "images")
    mask_dir    = os.path.join(save_dir, "masks")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(mask_dir,    exist_ok=True)

    for k, v in images.items():
        for idx, image in enumerate(v):
            fname = f"b-{batch_idx:06d}_idx-{idx}.png"

            if k == f"samples_cfg_scale_{CFG_SCALE:.2f}_mask":
                # Mask-conditioned output: [-1,1] → [0,255] RGB
                img = (image + 1.0) / 2.0
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(samples_dir, fname))

            elif k == f"samples_cfg_scale_{CFG_SCALE:.2f}_image":
                # Dual-control (mask+image) output: [-1,1] → [0,255] RGB
                dual_dir = os.path.join(save_dir, "images_dual")
                os.makedirs(dual_dir, exist_ok=True)
                img = (image + 1.0) / 2.0
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(dual_dir, fname))

            elif k == "control_mask":
                # Input mask: [0,1] single channel → binary PNG
                msk = image.permute(1, 2, 0).squeeze(-1).numpy()
                msk = (msk * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(msk).convert('L').save(os.path.join(mask_dir, fname))

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"ADC Local Inference")
    print(f"Device:     {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"DDIM steps: {DDIM_STEPS}, eta={DDIM_ETA}, CFG={CFG_SCALE}")
    print(f"Results:    {RESULT_DIR}")
    print(f"{'='*60}\n")

    model = get_model()

    dataset    = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset size: {len(dataset)} samples")

    with torch.no_grad():
        with model.ema_scope():
            for idx, batch in enumerate(dataloader):
                print(f"\n[{idx+1}/{len(dataloader)}] Generating...")

                images = model.log_images(
                    batch,
                    N=BATCH_SIZE,
                    ddim_steps=DDIM_STEPS,
                    ddim_eta=DDIM_ETA,
                    unconditional_guidance_scale=CFG_SCALE,
                )

                # Move all tensors to CPU for saving
                for k in images:
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

                log_local(RESULT_DIR, images, idx)
                print(f"  → Saved to {RESULT_DIR}")

    print(f"\nDone. All results in: {RESULT_DIR}")
