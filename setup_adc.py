#!/usr/bin/env python3
"""
setup_adc.py
=============
One-command setup for the ADC project.

Downloads all required weights, creates the training init checkpoint,
and installs Python dependencies. Run this after cloning the repo.

Usage:
    # Full setup (weights + dependencies + control checkpoint):
    uv run python setup_adc.py

    # Weights only (skip dependency install):
    uv run python setup_adc.py --weights-only

    # Dependencies only (skip weights):
    uv run python setup_adc.py --deps-only

    # Skip control checkpoint creation (saves time if you only need inference):
    uv run python setup_adc.py --no-control-ckpt

    # Dry run (show what would happen):
    uv run python setup_adc.py --dry-run

What this script does:
    1. Installs Python dependencies via uv (torch, pytorch-lightning, etc.)
    2. Downloads SD v1.5 base weights (~7.7 GB) → stable-diffusion-v1-5/
    3. Downloads ADC pretrained weights (~9.6 GB) → adc_weights/
    4. Creates control_sd15.ckpt (~9 GB) from SD v1.5 → stable-diffusion-v1-5/
       (This is the training init checkpoint — only needed if you plan to train.)

Total download: ~17 GB. Total disk after setup: ~35 GB.
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# Always run from the ADC project directory
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
os.chdir(SCRIPT_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
SD15_DIR      = SCRIPT_DIR / "stable-diffusion-v1-5"
SD15_CKPT     = SD15_DIR / "v1-5-pruned.ckpt"
CONTROL_CKPT  = SD15_DIR / "control_sd15.ckpt"
ADC_DIR       = SCRIPT_DIR / "adc_weights"
ADC_CKPT      = ADC_DIR / "merged_pytorch_model.pth"

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies — these are the packages ADC needs beyond the Python stdlib
# ──────────────────────────────────────────────────────────────────────────────
CORE_DEPS = [
    "torch",
    "torchvision",
    "pytorch-lightning",
    "einops",
    "omegaconf",
    "albumentations",
    "transformers",
    "safetensors",
    "open-clip-torch",
    "kornia",
    "scipy",
    "opencv-python",
    "scikit-learn",
    "huggingface-hub",
]

EVAL_DEPS = [
    "torchmetrics[image]",
    "lpips",
]


def run_cmd(cmd: list[str], description: str, dry_run: bool = False) -> bool:
    """Run a shell command, return True on success."""
    print(f"\n{'─'*60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*60}")
    if dry_run:
        print("  [DRY RUN] Skipped.")
        return True
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ✗ Failed (exit code {result.returncode})")
        return False
    print(f"  ✓ Done")
    return True


def install_dependencies(dry_run: bool = False, eval_deps: bool = True):
    """Install Python packages via uv."""
    print("\n" + "="*60)
    print("  STEP 1: Installing Python dependencies")
    print("="*60)

    # Check if uv is available
    uv_path = shutil.which("uv")
    if uv_path is None:
        print("  ⚠ 'uv' not found. Falling back to pip.")
        pip_cmd = [sys.executable, "-m", "pip", "install"]
    else:
        pip_cmd = ["uv", "pip", "install"]

    deps = CORE_DEPS + (EVAL_DEPS if eval_deps else [])
    run_cmd(pip_cmd + deps, "Installing core + evaluation packages", dry_run)


def download_sd15(dry_run: bool = False):
    """Download SD v1.5 v1-5-pruned.ckpt from HuggingFace."""
    print("\n" + "="*60)
    print("  STEP 2: Downloading Stable Diffusion v1.5 (~7.7 GB)")
    print("="*60)

    if SD15_CKPT.exists():
        size_gb = SD15_CKPT.stat().st_size / (1024**3)
        print(f"  ✓ Already exists: {SD15_CKPT} ({size_gb:.1f} GB)")
        return True

    SD15_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"  [DRY RUN] Would download to: {SD15_CKPT}")
        return True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ✗ huggingface_hub not installed. Run dependencies step first.")
        return False

    print(f"  Downloading to: {SD15_DIR}/")
    hf_hub_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        filename="v1-5-pruned.ckpt",
        local_dir=str(SD15_DIR),
    )
    print(f"  ✓ SD v1.5 downloaded: {SD15_CKPT}")
    return True


def download_adc_weights(dry_run: bool = False):
    """Download ADC pretrained weights from HuggingFace."""
    print("\n" + "="*60)
    print("  STEP 3: Downloading ADC pretrained weights (~9.6 GB)")
    print("="*60)

    if ADC_CKPT.exists():
        size_gb = ADC_CKPT.stat().st_size / (1024**3)
        print(f"  ✓ Already exists: {ADC_CKPT} ({size_gb:.1f} GB)")
        return True

    ADC_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"  [DRY RUN] Would download to: {ADC_CKPT}")
        return True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ✗ huggingface_hub not installed. Run dependencies step first.")
        return False

    print(f"  Downloading to: {ADC_DIR}/")
    hf_hub_download(
        repo_id="SylarQ/ADC",
        filename="merged_pytorch_model.pth",
        local_dir=str(ADC_DIR),
    )
    print(f"  ✓ ADC weights downloaded: {ADC_CKPT}")
    return True


def create_control_checkpoint(dry_run: bool = False):
    """Create control_sd15.ckpt from SD v1.5 weights (needed for training from scratch)."""
    print("\n" + "="*60)
    print("  STEP 4: Creating control_sd15.ckpt (training init)")
    print("="*60)

    if CONTROL_CKPT.exists():
        size_gb = CONTROL_CKPT.stat().st_size / (1024**3)
        print(f"  ✓ Already exists: {CONTROL_CKPT} ({size_gb:.1f} GB)")
        return True

    if not SD15_CKPT.exists():
        print(f"  ✗ SD v1.5 checkpoint not found: {SD15_CKPT}")
        print("    Run the weight download step first.")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would create: {CONTROL_CKPT}")
        return True

    # Use create_control_ckpt.py (self-contained, already tested)
    result = subprocess.run([sys.executable, str(SCRIPT_DIR / "create_control_ckpt.py")])
    if result.returncode != 0:
        print(f"  ✗ Failed to create control checkpoint")
        return False

    print(f"  ✓ Control checkpoint created: {CONTROL_CKPT}")
    return True


def print_summary():
    """Print final status of all components."""
    print("\n" + "="*60)
    print("  SETUP SUMMARY")
    print("="*60)

    items = [
        ("SD v1.5 weights",       SD15_CKPT),
        ("ADC pretrained weights", ADC_CKPT),
        ("Control checkpoint",    CONTROL_CKPT),
    ]

    all_ok = True
    for name, path in items:
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            print(f"  ✓ {name:30s} {path.name} ({size_gb:.1f} GB)")
        else:
            print(f"  ✗ {name:30s} MISSING")
            all_ok = False

    # Check key Python packages
    print()
    for pkg_name, import_name in [("torch", "torch"), ("pytorch-lightning", "pytorch_lightning"),
                                    ("einops", "einops"), ("transformers", "transformers")]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✓ {pkg_name:30s} v{ver}")
        except ImportError:
            print(f"  ✗ {pkg_name:30s} NOT INSTALLED")
            all_ok = False

    print()
    if all_ok:
        print("  All components ready! You can now run:")
        print("    Inference:  uv run python tutorial_inference_local.py")
        print("    Training:   uv run python tutorial_train_single_gpu.py")
    else:
        print("  Some components are missing. Re-run setup_adc.py to fix.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="One-command setup for ADC. Downloads weights, installs deps, creates checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full setup:           uv run python setup_adc.py
  Weights only:         uv run python setup_adc.py --weights-only
  Deps only:            uv run python setup_adc.py --deps-only
  Skip control ckpt:    uv run python setup_adc.py --no-control-ckpt
  Preview what happens: uv run python setup_adc.py --dry-run
        """,
    )
    parser.add_argument("--weights-only",    action="store_true", help="Only download weights, skip deps")
    parser.add_argument("--deps-only",       action="store_true", help="Only install dependencies, skip weights")
    parser.add_argument("--no-control-ckpt", action="store_true", help="Skip creating control_sd15.ckpt")
    parser.add_argument("--no-eval-deps",    action="store_true", help="Skip evaluation packages (torchmetrics, lpips)")
    parser.add_argument("--dry-run",         action="store_true", help="Show what would happen without doing it")
    args = parser.parse_args()

    print("="*60)
    print("  ADC Project Setup")
    print("  https://github.com/juliensauter/ADC")
    print("="*60)

    success = True

    # Step 1: Dependencies
    if not args.weights_only:
        install_dependencies(dry_run=args.dry_run, eval_deps=not args.no_eval_deps)

    # Step 2: SD v1.5
    if not args.deps_only:
        if not download_sd15(dry_run=args.dry_run):
            success = False

    # Step 3: ADC weights
    if not args.deps_only:
        if not download_adc_weights(dry_run=args.dry_run):
            success = False

    # Step 4: Control checkpoint
    if not args.deps_only and not args.no_control_ckpt:
        if not create_control_checkpoint(dry_run=args.dry_run):
            success = False

    # Summary
    if not args.dry_run:
        print_summary()
    else:
        print("\n[DRY RUN] No changes were made.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
