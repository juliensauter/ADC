"""
run_all.py — Run all ADC training presets sequentially in one process.

Features:
  - Auto-detects completed presets (skips if done)
  - Handles preset dependencies (topological ordering)
  - Each preset runs in its own subprocess (clean GPU memory between runs)
  - Runs post-training analysis at the end → generates runs/training_report.md

Usage:
    # Run all presets (autodetect completion, skip done):
    TRAINING_TARGET=workstation uv run python run_all.py

    # Run only specific presets:
    TRAINING_TARGET=workstation uv run python run_all.py scratch polyp_transfer

    # Via SLURM (single job for everything):
    PRESET=all sbatch slurm/train.sh
"""

import os
import re
import sys
import glob
import subprocess
import time
import csv
from pathlib import Path
from datetime import datetime

# Always run from ADC project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# Preset dependency graph — defines execution order
# Key: preset name, Value: source preset (None = base preset, no dependency)
# ──────────────────────────────────────────────────────────────────────────────
PRESET_DEPS = {
    "scratch":                   None,
    "polyp_transfer":            None,
    "scratch_unlocked":          "scratch",
    "polyp_unlocked":            "polyp_transfer",
    "polyp_stage2":              "polyp_transfer",
    "scratch_stage2":            "scratch_unlocked",
    "polyp_stage2_from_unlocked": "polyp_unlocked",
}

# Max steps per preset (must match tutorial_train_single_gpu.py PRESETS dict)
PRESET_MAX_STEPS = {
    "scratch": 20000,
    "polyp_transfer": 20000,
    "scratch_unlocked": 10000,
    "polyp_unlocked": 10000,
    "polyp_stage2": 10000,
    "scratch_stage2": 10000,
    "polyp_stage2_from_unlocked": 10000,
}

# Default execution order (topologically sorted — respects dependencies)
DEFAULT_ORDER = [
    "scratch",
    "polyp_transfer",
    "scratch_unlocked",
    "polyp_unlocked",
    "polyp_stage2",
    "scratch_stage2",
    "polyp_stage2_from_unlocked",
]


def find_last_checkpoint(preset_name: str) -> str | None:
    """Find the latest last.ckpt for a preset, or None if not found."""
    candidates = sorted(glob.glob(f"runs/{preset_name}/*/checkpoints/last.ckpt"))
    if not candidates and preset_name == "scratch":
        candidates = sorted(glob.glob("lightning_logs/*/checkpoints/last.ckpt"))
    return candidates[-1] if candidates else None


def get_max_step_from_filenames(preset_name: str) -> int:
    """Parse step counts from checkpoint filenames (fast, no file loading).

    Lightning saves checkpoints like epoch=49-step=20000.ckpt.
    Returns the highest step found, or 0 if none found.
    """
    max_step = 0
    # Search in runs/ and legacy lightning_logs/
    patterns = [f"runs/{preset_name}/*/checkpoints/epoch=*-step=*.ckpt"]
    if preset_name == "scratch":
        patterns.append("lightning_logs/*/checkpoints/epoch=*-step=*.ckpt")

    for pattern in patterns:
        for f in glob.glob(pattern):
            m = re.search(r"step=(\d+)", os.path.basename(f))
            if m:
                max_step = max(max_step, int(m.group(1)))
    return max_step


def get_completed_step(preset_name: str) -> int | None:
    """Read the last training step from CSVLogger metrics.

    Returns the global_step from the last row of the metrics CSV,
    or None if no metrics found.
    """
    metrics_dirs = sorted(glob.glob(f"runs/{preset_name}/version_*/metrics.csv"))
    if not metrics_dirs:
        return None
    # Read the last metrics file (latest version)
    metrics_path = metrics_dirs[-1]
    try:
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row and "step" in last_row:
                return int(last_row["step"])
    except (OSError, ValueError, KeyError):
        pass
    return None


def is_preset_complete(preset_name: str) -> bool:
    """Check if a preset has completed training.

    Uses three methods (cheapest first):
      1. CSVLogger metrics (runs/{preset}/version_*/metrics.csv)
      2. Checkpoint filenames (epoch=N-step=M.ckpt — fast filename parse)
      3. Loading checkpoint metadata (expensive, last resort)
    """
    max_steps = PRESET_MAX_STEPS.get(preset_name, 0)

    # Method 1: CSV metrics
    step = get_completed_step(preset_name)
    if step is not None and step >= max_steps:
        return True

    # Method 2: Parse step from checkpoint filenames (fast)
    file_step = get_max_step_from_filenames(preset_name)
    if file_step >= max_steps:
        return True

    # Method 3: Load checkpoint and read global_step (expensive, ~12GB)
    ckpt = find_last_checkpoint(preset_name)
    if ckpt:
        try:
            import torch
            ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
            ckpt_step = ckpt_data.get("global_step", 0)
            del ckpt_data  # free memory immediately
            if ckpt_step >= max_steps:
                return True
        except Exception:
            pass

    return False


def source_checkpoint_available(preset_name: str) -> bool:
    """Check if a chain preset's source checkpoint exists."""
    dep = PRESET_DEPS.get(preset_name)
    if dep is None:
        return True  # base preset, no dependency
    return find_last_checkpoint(dep) is not None


def run_preset(preset_name: str, training_target: str) -> bool:
    """Run a single preset via subprocess.

    Returns True on success, False on failure.
    """
    env = os.environ.copy()
    env["PRESET"] = preset_name
    env["TRAINING_TARGET"] = training_target

    cmd = [sys.executable, "tutorial_train_single_gpu.py"]

    print(f"\n{'─'*60}")
    print(f"  Running preset: {preset_name}")
    print(f"  Command: PRESET={preset_name} {' '.join(cmd)}")
    print(f"{'─'*60}\n")

    start_time = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n  ✓ Preset '{preset_name}' completed in {elapsed/60:.1f} min")
        return True
    else:
        print(f"\n  ✗ Preset '{preset_name}' FAILED (exit code {result.returncode}) "
              f"after {elapsed/60:.1f} min")
        return False


def run_analysis():
    """Run post-training analysis to generate runs/training_report.md."""
    analysis_script = Path("analyze_runs.py")
    if not analysis_script.exists():
        print("\n  [!] analyze_runs.py not found — skipping analysis")
        return

    print(f"\n{'='*60}")
    print("  Running post-training analysis...")
    print(f"{'='*60}\n")

    result = subprocess.run([sys.executable, str(analysis_script)])
    if result.returncode == 0:
        print("  ✓ Analysis complete → runs/training_report.md")
    else:
        print(f"  ✗ Analysis failed (exit code {result.returncode})")


def check_base_weights() -> bool:
    """Verify that required base model weights exist before training.

    Returns True if all weights are available, False otherwise.
    """
    required_files = {
        "SD v1.5 base": "stable-diffusion-v1-5/control_sd15.ckpt",
        "ADC polyp weights": "adc_weights/merged_pytorch_model.pth",
    }
    all_ok = True
    for desc, path in required_files.items():
        if os.path.isfile(path):
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"    ✓ {desc}: {path} ({size_gb:.1f} GB)")
        else:
            print(f"    ✗ {desc}: {path} — NOT FOUND")
            all_ok = False
    if not all_ok:
        print("\n  [!] Missing base weights. Run: uv run python download_weights.py")
    return all_ok


def main():
    training_target = os.environ.get("TRAINING_TARGET", "workstation")

    # Determine which presets to run
    if len(sys.argv) > 1:
        presets = sys.argv[1:]
        # Validate
        for p in presets:
            if p not in PRESET_DEPS:
                print(f"Unknown preset: {p!r}. Options: {list(PRESET_DEPS.keys())}")
                sys.exit(1)
    else:
        presets = DEFAULT_ORDER

    print(f"\n{'='*60}")
    print(f"  ADC run_all — training {len(presets)} presets")
    print(f"  Target: {training_target}")
    print(f"  Presets: {' → '.join(presets)}")
    print(f"{'='*60}")

    # Pre-flight: check base weights
    print("\n  Base weights:")
    if not check_base_weights():
        sys.exit(1)

    # Status check
    print("\n  Preset status:")
    for p in presets:
        done = is_preset_complete(p)
        dep = PRESET_DEPS.get(p)
        dep_ok = source_checkpoint_available(p)
        status = "DONE (skip)" if done else ("READY" if dep_ok else f"BLOCKED (waiting for {dep})")
        print(f"    {p:30s}  {status}")

    # Run presets in order
    results = {}
    total_start = time.time()

    for preset_name in presets:
        # Skip completed presets
        if is_preset_complete(preset_name):
            print(f"\n  Skipping '{preset_name}' — already complete")
            results[preset_name] = "skipped"
            continue

        # Check dependency
        if not source_checkpoint_available(preset_name):
            dep = PRESET_DEPS[preset_name]
            # Check if the dependency was run earlier in this session and might now be available
            if results.get(dep) == "success":
                pass  # dependency just completed, should be available now
            else:
                print(f"\n  Skipping '{preset_name}' — source preset '{dep}' not available")
                results[preset_name] = "blocked"
                continue

        # Run it
        success = run_preset(preset_name, training_target)
        results[preset_name] = "success" if success else "failed"

        if not success:
            print(f"\n  [!] Preset '{preset_name}' failed. Continuing with remaining presets...")

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Training complete — {total_elapsed/60:.1f} min total")
    print(f"{'='*60}")
    print("\n  Results:")
    for p, status in results.items():
        icon = {"success": "✓", "skipped": "─", "failed": "✗", "blocked": "⊘"}.get(status, "?")
        print(f"    {icon} {p:30s}  {status}")

    # Run analysis
    run_analysis()

    # Final status
    failures = [p for p, s in results.items() if s == "failed"]
    if failures:
        print(f"\n  [!] {len(failures)} preset(s) failed: {failures}")
        sys.exit(1)
    else:
        print(f"\n  All presets completed successfully.")


if __name__ == "__main__":
    main()
