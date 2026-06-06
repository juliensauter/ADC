"""ADC eval CLI — `python -m eval.cli ...`.

Argparse entrypoint for single-checkpoint and A/B-branch evaluation runs.
See `working/notes/metrics_implementation_plan.md` §A.6 for the full
interface spec; for now this scaffolds `--help` so the module is importable
and the test `python -m eval.cli --help` succeeds.
"""

from __future__ import annotations
import argparse
import sys
import json
import os
import time
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from tutorial_dataset_sample import MyDataset
from eval.runners import load_model, generate_batched_images, score_generated_stream


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the ADC validation pipeline."""
    parser = argparse.ArgumentParser(
        prog="eval.cli",
        description="ADC evaluation pipeline (KID-DINOv2, LPIPS, DreamSim, Boundary IoU).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the ADC PyTorch Lightning checkpoint (e.g. runs/.../last.ckpt).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="paper_faithful_v2_polyp",
        help="Name of the training preset used (default: paper_faithful_v2_polyp).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation and PyTorch environments (default: 42).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Loader batch size (default: 2).",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=1,
        help="Number of batches to evaluate (default: 1).",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="DDIM sampling steps (default: 50).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to save the generated JSON metrics results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: auto | cuda | mps | cpu (default: auto).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/test/prompt.json",
        help="Path to the JSONL data prompt metadata file (default: ./data/test/prompt.json).",
    )
    return parser


def get_git_sha() -> str:
    """Extract current git commit SHA if git is available."""
    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        return sha
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Device Setup
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    print(f"\n{'=' * 60}")
    print(f"ADC Evaluation CLI")
    print(f"Device:      {device}")
    print(f"Checkpoint:  {args.ckpt}")
    print(f"Preset:      {args.preset}")
    print(f"Seed:        {args.seed}")
    print(f"Steps/Eta:   {args.ddim_steps} / 0.0")
    print(f"Batch config: {args.n_batches} batches of size {args.batch_size}")
    print(f"{'=' * 60}\n")

    # 2. Dataset loader initialization
    if not os.path.exists(args.data_root):
        print(f"Error: dataset path not found: {args.data_root}")
        return 1

    dataset = MyDataset(root=args.data_root)
    total_requested = args.batch_size * args.n_batches
    if total_requested > len(dataset):
        print(
            f"Warning: requested {total_requested} samples, but dataset only has "
            f"{len(dataset)}. Clamping evaluation size."
        )
        args.n_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    dataloader = DataLoader(
        dataset, num_workers=0, batch_size=args.batch_size, shuffle=False
    )
    # Wrap with a slicing generator to clamp stream to requested n_batches
    def slice_loader(loader, n):
        for idx, item in enumerate(loader):
            if idx >= n:
                break
            yield item

    sliced_loader = slice_loader(dataloader, args.n_batches)

    # 3. Model Loading
    print("Loading model...")
    model = load_model(args.ckpt, device)

    # 4. Stream & Evaluate
    print("Running evaluation loop...")
    start_time = time.time()
    stream = generate_batched_images(
        model=model,
        dataloader=sliced_loader,
        device=device,
        ddim_steps=args.ddim_steps,
        ddim_eta=0.0,
        cfg_scale=9.0,
    )

    metrics = score_generated_stream(
        gen_stream=stream,
        device=device,
        kid_subset_size=50,  # Match standard plan defaults
        kid_subsets=50,
    )
    elapsed = time.time() - start_time

    # 5. Build structured report
    env_flags = {k: os.environ.get(k, "0") for k in os.environ if k.startswith("ADC_")}

    # Get device identifier details
    gpu_name = "unknown"
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        gpu_name = "Apple Silicon"

    output_data = {
        "config": {
            "ckpt_path": str(Path(args.ckpt).resolve()),
            "preset": args.preset,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "n_batches": args.n_batches,
            "ddim_steps": args.ddim_steps,
            "git_sha": get_git_sha(),
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu_name": gpu_name,
            "env_flags": env_flags,
            "data_root": str(Path(args.data_root).resolve()),
        },
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Write results
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print clean console table report
    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete in {elapsed:.2f} seconds.")
    print(f"Results saved to: {args.out}")
    print(f"{'=' * 60}")
    print(f"| Metric | Score |")
    print(f"|---|---|")
    print(f"| KID (mean) | {metrics['kid_mean']:.5f} |")
    print(f"| KID (std) | {metrics['kid_std']:.5f} |")
    print(f"| LPIPS | {metrics['lpips_mean']:.5f} |")
    print(f"| DreamSim | {metrics['dreamsim_mean']:.5f} |")
    print(f"| Boundary IoU | {metrics['boundary_iou_mean']:.5f} |")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
