"""
tutorial_test_eval.py
=====================
Generate the full test split for multiple seeds and measure image quality.

Metrics:
  - KID  (DINOv2 embeddings)
  - FDD  (Fréchet distance on DINOv2 embeddings)
  - LPIPS (AlexNet backbone)
  - mIoU (configurable segmentation model)

Usage:
    uv run python tutorial_test_eval.py

The default seed sweep comes from experiment_config.py and uses 10 seeds.
Override with --seeds or TEST_SEEDS="0,1,2,...".
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Always run from the ADC project directory regardless of how the script is invoked.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from cldm.model import create_model, load_state_dict
from experiment_config import (
    DEFAULT_DINOV2_MODEL_NAME,
    DEFAULT_EVAL_CFG_SCALE,
    DEFAULT_LPIPS_BACKBONE,
    DEFAULT_MIOU_THRESHOLD,
    DEFAULT_SEGMENTATION_MODEL,
    DEFAULT_TEST_DDIM_STEPS,
    DEFAULT_TEST_SEEDS,
    resolve_seed_list,
    set_global_seed,
)
from adc_metrics import QualityMetrics, load_segmentation_model, save_mask_tensor, save_rgb_tensor
from tutorial_dataset_sample import MyDataset


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


def resolve_checkpoint_path() -> str:
    trained = sorted(
        _glob.glob("runs/*/*/checkpoints/last.ckpt")
        + _glob.glob("lightning_logs/*/checkpoints/last.ckpt")
    )
    return trained[-1] if trained else "./adc_weights/merged_pytorch_model.pth"


def get_model(device: torch.device, ckpt_path: str):
    model = create_model("./models/cldm_v15.yaml").cpu()

    if device.type in ("mps", "cpu"):
        model = model.float()

    model.load_state_dict(load_state_dict(ckpt_path, location="cpu"), strict=False)
    model.learning_rate = 1e-5
    model.sd_locked = False
    model.only_mid_control = False
    model.to(device)
    model.eval()
    return model


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def parse_seed_list(seed_argument: str | None) -> tuple[int, ...]:
    if seed_argument is None or seed_argument.strip() == "":
        return resolve_seed_list(default=DEFAULT_TEST_SEEDS)
    return tuple(int(item.strip()) for item in seed_argument.split(",") if item.strip())


def evaluate_seed(
    seed: int,
    model,
    dataloader: DataLoader,
    quality_metrics: QualityMetrics,
    result_root: Path,
    device: torch.device,
    ddim_steps: int,
    cfg_scale: float,
):
    set_global_seed(seed)

    seed_root = result_root / f"seed_{seed:03d}"
    generated_dir = seed_root / "generated"
    target_dir = seed_root / "target"
    mask_dir = seed_root / "mask"
    generated_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    generated_batches = []
    target_batches = []
    mask_batches = []

    output_key = f"samples_cfg_scale_{cfg_scale:.2f}_mask"

    with torch.no_grad(), model.ema_scope():
        for batch_index, batch in enumerate(dataloader):
            batch = move_batch_to_device(batch, device)
            images = model.log_images(
                batch,
                N=batch["jpg"].shape[0],
                ddim_steps=ddim_steps,
                unconditional_guidance_scale=cfg_scale,
            )

            generated = torch.clamp(images[output_key].detach().cpu(), -1.0, 1.0)
            target = batch["jpg"].detach().cpu()
            mask = batch["hint"].detach().cpu()

            generated_batches.append(generated)
            target_batches.append(target)
            mask_batches.append(mask)

            save_rgb_tensor(generated[0], generated_dir / f"{batch_index:06d}.png")
            save_rgb_tensor((target[0] + 1.0) / 2.0, target_dir / f"{batch_index:06d}.png")
            save_mask_tensor(mask[0], mask_dir / f"{batch_index:06d}.png")

    kid_score = quality_metrics.compute_kid(target_batches, generated_batches)
    fdd_score = quality_metrics.compute_fdd(target_batches, generated_batches)
    lpips_score = quality_metrics.compute_lpips(target_batches, generated_batches)
    miou_score = quality_metrics.compute_miou(generated_batches, mask_batches)

    return {
        "seed": seed,
        "kid": kid_score,
        "fdd": fdd_score,
        "lpips": lpips_score,
        "miou": miou_score,
        "n_images": len(dataloader.dataset),
        "result_dir": str(seed_root.resolve()),
    }


def summarize_results(results: list[dict]) -> dict:
    metric_names = ["kid", "fdd", "lpips", "miou"]
    summary = {"per_seed": results, "aggregate": {}}
    for metric_name in metric_names:
        values = np.array([entry[metric_name] for entry in results], dtype=np.float64)
        summary["aggregate"][metric_name] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if np.sum(~np.isnan(values)) > 1 else 0.0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate all test images and evaluate ADC quality metrics.")
    parser.add_argument("--data-root", default="./data/test/prompt.json", help="Test split prompt.json path")
    parser.add_argument("--ckpt-path", default=resolve_checkpoint_path(), help="Checkpoint to evaluate")
    parser.add_argument("--result-root", default="./generated_results/test_eval/", help="Output root directory")
    parser.add_argument("--ddim-steps", type=int, default=DEFAULT_TEST_DDIM_STEPS, help="DDIM steps for generation")
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_EVAL_CFG_SCALE, help="CFG scale for generation")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list; defaults to config")
    parser.add_argument("--metric-device", default="cpu", help="Device for metric computation")
    parser.add_argument("--dinov2-model", default=DEFAULT_DINOV2_MODEL_NAME, help="DINOv2 backbone for KID/FDD")
    parser.add_argument("--lpips-backbone", default=DEFAULT_LPIPS_BACKBONE, help="LPIPS backbone")
    parser.add_argument("--segmentation-model", default=DEFAULT_SEGMENTATION_MODEL, help="Segmentation model factory")
    parser.add_argument("--segmentation-checkpoint", default=None, help="Optional segmentation checkpoint")
    parser.add_argument("--miou-threshold", type=float, default=DEFAULT_MIOU_THRESHOLD, help="Binary mask threshold")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    device = get_device()
    seed_values = parse_seed_list(args.seeds)
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("ADC Test Evaluation")
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"DDIM steps: {args.ddim_steps}, CFG={args.cfg_scale}")
    print(f"Seeds:      {list(seed_values)}")
    print(f"Results:    {result_root}")
    print(f"{'='*60}\n")

    model = get_model(device, args.ckpt_path)
    dataset = MyDataset(root=args.data_root)
    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    segmentation_model, segmentation_name = load_segmentation_model(
        args.segmentation_model,
        args.segmentation_checkpoint,
        device=args.metric_device,
        strict=False,
    )
    print(f"Segmentation model: {segmentation_name}")

    quality_metrics = QualityMetrics(
        device=args.metric_device,
        dinov2_model_name=args.dinov2_model,
        lpips_backbone=args.lpips_backbone,
        segmentation_model=segmentation_model,
        segmentation_threshold=args.miou_threshold,
    )

    results = []
    for seed in seed_values:
        print(f"\n[seed={seed}] Generating test images and computing metrics ...")
        seed_result = evaluate_seed(
            seed=seed,
            model=model,
            dataloader=dataloader,
            quality_metrics=quality_metrics,
            result_root=result_root,
            device=device,
            ddim_steps=args.ddim_steps,
            cfg_scale=args.cfg_scale,
        )
        results.append(seed_result)
        print(
            f"  kid={seed_result['kid']:.6f}  fdd={seed_result['fdd']:.6f}  "
            f"lpips={seed_result['lpips']:.6f}  miou={seed_result['miou']:.6f}"
        )

    summary = summarize_results(results)
    summary_path = result_root / "summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print("Aggregate metrics:")
    for metric_name, stats in summary["aggregate"].items():
        print(f"  {metric_name}: mean={stats['mean']:.6f} std={stats['std']:.6f}")


if __name__ == "__main__":
    main()
