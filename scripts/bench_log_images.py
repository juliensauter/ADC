"""
Quick benchmark for log_images() inference latency.

Usage:
    # Baseline (no env vars)
    uv run python scripts/bench_log_images.py

    # With all inference fixes active
    ADC_LOG_IMAGES_N=2 ADC_LOG_IMAGES_SKIP_DUAL_CN=1 \
        uv run python scripts/bench_log_images.py

Reads CKPT_PATH from env (or falls back to a sensible default that you
should override). Loads the ControlLDM model, runs `log_images()` once
to warm up CUDA + autocast caches, then times 3 calls and reports
mean/median/stdev wall time.

Output is one line of JSON for easy grep/jq, plus a human-readable
summary.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cldm.model import create_model, load_state_dict  # noqa: E402
from tutorial_dataset_sample import MyDataset  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT_PATH = os.environ.get(
    "CKPT_PATH",
    "./lightning_logs/version_0/checkpoints/last.ckpt",
)
CONFIG_PATH = os.environ.get("CONFIG_PATH", "./models/cldm_v15_dual_decoder.yaml")
BATCH_SIZE = int(os.environ.get("BENCH_BATCH_SIZE", "4"))
DDIM_STEPS = int(os.environ.get("BENCH_DDIM_STEPS", "50"))
CFG_SCALE = float(os.environ.get("BENCH_CFG_SCALE", "9.0"))
N_RUNS = int(os.environ.get("BENCH_N_RUNS", "3"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "1"))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[bench] device={device}, ckpt={CKPT_PATH}")
    print(f"[bench] batch_size={BATCH_SIZE}, ddim_steps={DDIM_STEPS}, cfg={CFG_SCALE}")
    print(f"[bench] env ADC_LOG_IMAGES_SEED={os.environ.get('ADC_LOG_IMAGES_SEED', '<unset>')}")
    print(f"[bench] env ADC_LOG_IMAGES_N={os.environ.get('ADC_LOG_IMAGES_N', '<unset>')}")
    print(f"[bench] env ADC_LOG_IMAGES_SKIP_DUAL_CN={os.environ.get('ADC_LOG_IMAGES_SKIP_DUAL_CN', '<unset>')}")

    model = create_model(CONFIG_PATH).to(device)
    model.load_state_dict(load_state_dict(CKPT_PATH, location=device), strict=False)
    model.eval()

    dataset = MyDataset()
    loader = DataLoader(dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=False)
    batch = next(iter(loader))

    # Warmup
    with torch.no_grad(), model.ema_scope():
        for _ in range(WARMUP):
            _ = model.log_images(
                batch, N=BATCH_SIZE, ddim_steps=DDIM_STEPS,
                unconditional_guidance_scale=CFG_SCALE,
            )
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times: list[float] = []
    with torch.no_grad(), model.ema_scope():
        for i in range(N_RUNS):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.log_images(
                batch, N=BATCH_SIZE, ddim_steps=DDIM_STEPS,
                unconditional_guidance_scale=CFG_SCALE,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"[bench] run {i+1}/{N_RUNS}: {dt:.2f}s")

    summary = {
        "device": device,
        "ckpt": CKPT_PATH,
        "batch_size": BATCH_SIZE,
        "ddim_steps": DDIM_STEPS,
        "n_runs": N_RUNS,
        "mean_s": round(statistics.mean(times), 3),
        "median_s": round(statistics.median(times), 3),
        "stdev_s": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "env": {
            "ADC_LOG_IMAGES_SEED": os.environ.get("ADC_LOG_IMAGES_SEED"),
            "ADC_LOG_IMAGES_N": os.environ.get("ADC_LOG_IMAGES_N"),
            "ADC_LOG_IMAGES_SKIP_DUAL_CN": os.environ.get("ADC_LOG_IMAGES_SKIP_DUAL_CN"),
        },
    }
    print("\n[bench] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
