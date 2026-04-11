"""
evaluate_adc.py
================
Evaluates ADC-generated images against real images.

Metrics computed:
  - FID  (Fréchet Inception Distance)  — distribution-level quality, lower = better
  - SSIM (Structural Similarity Index) — per-image structural match, higher = better
  - LPIPS (Perceptual similarity)      — per-image perceptual match, lower = better

Usage:
    # Run inference first to populate generated_results/
    uv run python tutorial_inference_local.py

    # Then evaluate:
    uv run python evaluate_adc.py \
        --real  data/images \
        --generated generated_results/local_test/images \
        --out   eval_results.json

Dependencies (install via uv):
    uv pip install torchmetrics[image] lpips

Notes:
  - FID operates on distributions — needs ≥50 images per set for meaningful results.
    With very few images (like a single test image), FID will be noisy.
  - SSIM/LPIPS are per-image; only meaningful if generated images correspond 1:1 to
    real images (i.e. same mask used to generate from real and generated).
  - For pure generation quality with no real reference, FID is the primary metric.
"""

import os
import sys
import json
import argparse
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def load_images_as_tensor(img_dir: Path, size: int = 299) -> torch.Tensor:
    """Load all images from directory, resize, return float32 tensor [N, C, H, W] in [0, 1]."""
    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    assert paths, f"No images found in {img_dir}"
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # → [0, 1]
    ])
    imgs = [tf(Image.open(p).convert("RGB")).unsqueeze(0) for p in paths]
    print(f"  Loaded {len(imgs)} images from {img_dir}")
    return torch.cat(imgs, dim=0), paths


def compute_fid(real_dir: Path, gen_dir: Path, device: torch.device, size: int = 299) -> float:
    """FID via torchmetrics. Requires ≥2 images per set."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("torchmetrics not installed — skipping FID. Run: uv pip install torchmetrics[image]")
        return float("nan")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    real_imgs, _ = load_images_as_tensor(real_dir, size)
    gen_imgs,  _ = load_images_as_tensor(gen_dir,  size)

    BATCH = 4
    for i in range(0, len(real_imgs), BATCH):
        fid.update(real_imgs[i:i+BATCH].to(device), real=True)
    for i in range(0, len(gen_imgs), BATCH):
        fid.update(gen_imgs[i:i+BATCH].to(device), real=False)

    return fid.compute().item()


def compute_ssim_lpips(real_dir: Path, gen_dir: Path, device: torch.device):
    """
    Compute SSIM and LPIPS per-image (pairs matched by sorted order).
    Only meaningful when generated images correspond 1:1 to real images.
    """
    real_paths = sorted([p for p in real_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    gen_paths  = sorted([p for p in gen_dir.iterdir()  if p.suffix.lower() in IMG_EXTS])
    n = min(len(real_paths), len(gen_paths))
    if n == 0:
        return float("nan"), float("nan")

    # SSIM
    ssim_scores = []
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        for rp, gp in zip(real_paths[:n], gen_paths[:n]):
            r = tf(Image.open(rp).convert("RGB")).unsqueeze(0).to(device)
            g = tf(Image.open(gp).convert("RGB")).unsqueeze(0).to(device)
            ssim_scores.append(ssim_fn(g, r).item())
    except ImportError:
        print("torchmetrics not installed — skipping SSIM.")

    # LPIPS
    lpips_scores = []
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        tf2 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])
        with torch.no_grad():
            for rp, gp in zip(real_paths[:n], gen_paths[:n]):
                r = tf2(Image.open(rp).convert("RGB")).unsqueeze(0).to(device)
                g = tf2(Image.open(gp).convert("RGB")).unsqueeze(0).to(device)
                lpips_scores.append(lpips_fn(g, r).item())
    except ImportError:
        print("lpips not installed — skipping LPIPS. Run: uv pip install lpips")

    mean_ssim  = float(np.mean(ssim_scores))  if ssim_scores  else float("nan")
    mean_lpips = float(np.mean(lpips_scores)) if lpips_scores else float("nan")
    return mean_ssim, mean_lpips


def main():
    parser = argparse.ArgumentParser(description="Evaluate ADC-generated images.")
    parser.add_argument("--real",      required=True, help="Directory of real reference images")
    parser.add_argument("--generated", required=True, help="Directory of ADC-generated images")
    parser.add_argument("--out",       default="eval_results.json", help="Output JSON path")
    parser.add_argument("--device",    default="auto", help="Device: auto|mps|cpu|cuda")
    args = parser.parse_args()

    real_dir = Path(args.real)
    gen_dir  = Path(args.generated)
    assert real_dir.exists(), f"Real images directory not found: {real_dir}"
    assert gen_dir.exists(),  f"Generated images directory not found: {gen_dir}"

    if args.device == "auto":
        if torch.cuda.is_available():       device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                               device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\nDevice: {device}")
    print(f"Real:      {real_dir}  ({sum(1 for p in real_dir.iterdir() if p.suffix.lower() in IMG_EXTS)} images)")
    print(f"Generated: {gen_dir}  ({sum(1 for p in gen_dir.iterdir()  if p.suffix.lower() in IMG_EXTS)} images)")

    print("\n[1/3] Computing FID ...")
    fid = compute_fid(real_dir, gen_dir, device)
    print(f"  FID = {fid:.4f}  (lower is better; meaningful with ≥50 images per set)")

    print("\n[2/3] Computing SSIM + LPIPS ...")
    ssim, lpips_score = compute_ssim_lpips(real_dir, gen_dir, device)
    print(f"  SSIM  = {ssim:.4f}  (higher is better; 1.0 = identical)")
    print(f"  LPIPS = {lpips_score:.4f}  (lower is better; 0.0 = identical)")

    results = {
        "real_dir":      str(real_dir.resolve()),
        "gen_dir":       str(gen_dir.resolve()),
        "n_real":        sum(1 for p in real_dir.iterdir() if p.suffix.lower() in IMG_EXTS),
        "n_generated":   sum(1 for p in gen_dir.iterdir()  if p.suffix.lower() in IMG_EXTS),
        "FID":           fid,
        "SSIM":          ssim,
        "LPIPS":         lpips_score,
        "notes": {
            "FID":   "distribution-level; meaningful only with ≥50 images per set",
            "SSIM":  "per-image structural similarity; assumes 1:1 correspondence between real and generated",
            "LPIPS": "per-image perceptual similarity; assumes 1:1 correspondence between real and generated",
        }
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
