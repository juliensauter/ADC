"""Run fold-aware ADC training sequentially for one preset.

This script is intentionally narrow: one preset, `NUM_FOLDS` folds, one fold
per subprocess. It reuses `tutorial_train_single_gpu.py` as the actual
training entrypoint and sets `FOLD_INDEX` per run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment_config import resolve_num_folds


# Always run from the ADC project directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def parse_fold_list(fold_spec: str | None, num_folds: int) -> list[int]:
    if fold_spec is None or fold_spec.strip() == "":
        return list(range(num_folds))
    folds = [int(item.strip()) for item in fold_spec.split(",") if item.strip()]
    for fold_index in folds:
        if fold_index < 0 or fold_index >= num_folds:
            raise ValueError(f"Fold index {fold_index} must be within [0, {num_folds - 1}]")
    return folds


def run_fold(preset: str, training_target: str, num_folds: int, fold_index: int) -> int:
    env = os.environ.copy()
    env["PRESET"] = preset
    env["TRAINING_TARGET"] = training_target
    env["NUM_FOLDS"] = str(num_folds)
    env["FOLD_INDEX"] = str(fold_index)

    command = [sys.executable, "tutorial_train_single_gpu.py"]

    print(f"\n{'─' * 60}")
    print(f"  Running preset={preset} fold={fold_index}/{num_folds - 1}")
    print(f"  Command: PRESET={preset} FOLD_INDEX={fold_index} {' '.join(command)}")
    print(f"{'─' * 60}\n")

    start_time = time.time()
    result = subprocess.run(command, env=env)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n  ✓ fold {fold_index} finished in {elapsed / 60:.1f} min")
    else:
        print(f"\n  ✗ fold {fold_index} FAILED (exit code {result.returncode}) after {elapsed / 60:.1f} min")

    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run k-fold ADC training for one preset.")
    parser.add_argument("--preset", default=os.environ.get("PRESET", "scratch"), help="Training preset to run")
    parser.add_argument(
        "--training-target",
        default=os.environ.get("TRAINING_TARGET", "workstation"),
        help="Hardware target passed through to the trainer",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=resolve_num_folds(),
        help="Number of folds to run [default: 5]",
    )
    parser.add_argument(
        "--folds",
        default=None,
        help="Optional comma-separated fold list (e.g. 0,2,4). Defaults to all folds.",
    )
    args = parser.parse_args()

    fold_indices = parse_fold_list(args.folds, args.num_folds)

    for fold_index in fold_indices:
        train_root = Path(f"data/fold_{fold_index}/train/prompt.json")
        val_root = Path(f"data/fold_{fold_index}/val/prompt.json")
        if not train_root.exists() or not val_root.exists():
            raise FileNotFoundError(
                f"Missing fold data for fold {fold_index}: {train_root} / {val_root}. "
                "Run prepare_liver_data.py with --k-fold first."
            )

    print(f"\n{'=' * 60}")
    print(f"  ADC k-fold training")
    print(f"  preset:        {args.preset}")
    print(f"  training_target: {args.training_target}")
    print(f"  num_folds:     {args.num_folds}")
    print(f"  folds:         {fold_indices}")
    print(f"{'=' * 60}")

    for fold_index in fold_indices:
        exit_code = run_fold(args.preset, args.training_target, args.num_folds, fold_index)
        if exit_code != 0:
            sys.exit(exit_code)

    print(f"\nAll requested folds completed successfully for preset '{args.preset}'.")


if __name__ == "__main__":
    main()
