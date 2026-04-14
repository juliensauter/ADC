"""
vram_calculator.py — Exact VRAM budget estimation for each ADC training preset.

Loads the actual model architecture and counts parameters precisely,
then estimates VRAM usage for different training configurations.

Usage:
    uv run python vram_calculator.py                    # all presets
    uv run python vram_calculator.py paper_faithful_polyp  # single preset
    uv run python vram_calculator.py --batch 2          # override batch size
"""

import os
import sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import torch
from cldm.model import create_model


def count_params(module: torch.nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def count_unique_params(module: torch.nn.Module) -> int:
    """Count unique parameters (avoids double-counting shared params)."""
    seen = set()
    total = 0
    for p in module.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def fmt_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_gb(n_bytes: int) -> str:
    """Format bytes as GB."""
    return f"{n_bytes / (1024**3):.2f} GB"


def estimate_activation_memory(batch_size: int, resolution: int = 384,
                               latent_channels: int = 4, use_checkpointing: bool = True) -> int:
    """Rough estimate of activation memory during training.

    Based on empirical observations for SD v1.5 UNet at 384×384:
    - Latent resolution = 384/8 = 48×48
    - Without checkpointing: ~4-6 GB at batch 1
    - With checkpointing: ~2-3 GB at batch 1
    - Scales roughly linearly with batch size
    """
    latent_res = resolution // 8  # VAE downsamples 8×
    # Base activation estimate (single forward pass through UNet + 2× ControlNets)
    # Empirically calibrated for 48×48 latent
    base_bytes_per_sample = 1.5 * (1024**3) if use_checkpointing else 4.0 * (1024**3)
    # Scale with latent area (quadratic in resolution)
    scale = (latent_res / 48) ** 2
    return int(base_bytes_per_sample * batch_size * scale)


def estimate_deep_copy_overhead(model) -> int:
    """Estimate memory for copy.deepcopy(h) + copy.deepcopy(hs) in UNet forward.

    These are activation tensors, not parameters. Rough estimate based on
    the number of feature channels flowing through the encoder.
    """
    # Roughly 0.5-1 GB for batch 1 at 384×384
    return int(0.75 * (1024**3))


def analyze_preset(preset_name: str, preset_config: dict, model, batch_size_override: int | None = None,
                   training_target: str = "workstation"):
    """Analyze VRAM requirements for a specific preset."""
    diffusion_model = model.model.diffusion_model

    # Component parameter counts
    components = {
        "Mask ControlNet (control_model)": count_params(model.control_model),
        "Image ControlNet (image_control_model)": count_params(model.image_control_model),
        "UNet Encoder (input_blocks + middle)": (
            sum(count_params(b) for b in diffusion_model.input_blocks) +
            count_params(diffusion_model.middle_block)
        ),
        "Mask Decoder (output_blocks)": sum(count_params(b) for b in diffusion_model.output_blocks),
        "Image Decoder (image_output_blocks)": sum(count_params(b) for b in diffusion_model.image_output_blocks),
        "Mask Out (out)": count_params(diffusion_model.out),
        "Image Out (image_out)": count_params(diffusion_model.image_out),
        "VAE (first_stage_model)": count_params(model.first_stage_model),
        "CLIP (cond_stage_model)": count_params(model.cond_stage_model),
    }

    # Determine which components are trainable
    sd_locked = preset_config.get("sd_locked", True)
    train_mask_cn = preset_config.get("train_mask_cn", True)
    train_image_cn = preset_config.get("train_image_cn", True)
    unlock_last_n = preset_config.get("unlock_last_n", 0)

    trainable_params = 0
    frozen_params = 0
    trainable_breakdown = {}

    # Mask ControlNet
    cn_mask_params = components["Mask ControlNet (control_model)"]
    if train_mask_cn:
        trainable_params += cn_mask_params
        trainable_breakdown["Mask ControlNet"] = cn_mask_params
    else:
        frozen_params += cn_mask_params

    # Image ControlNet
    cn_image_params = components["Image ControlNet (image_control_model)"]
    if train_image_cn:
        trainable_params += cn_image_params
        trainable_breakdown["Image ControlNet"] = cn_image_params
    else:
        frozen_params += cn_image_params

    # Encoder is always frozen
    frozen_params += components["UNet Encoder (input_blocks + middle)"]

    # Decoder blocks
    n_total = len(diffusion_model.output_blocks)
    if not sd_locked and unlock_last_n > 0:
        for i in range(n_total - unlock_last_n, n_total):
            mask_block_params = count_params(diffusion_model.output_blocks[i])
            trainable_params += mask_block_params
            trainable_breakdown[f"output_blocks[{i}]"] = mask_block_params
            if train_image_cn:
                img_block_params = count_params(diffusion_model.image_output_blocks[i])
                trainable_params += img_block_params
                trainable_breakdown[f"image_output_blocks[{i}]"] = img_block_params
        # Frozen blocks
        for i in range(n_total - unlock_last_n):
            frozen_params += count_params(diffusion_model.output_blocks[i])
            frozen_params += count_params(diffusion_model.image_output_blocks[i])
        # out layers
        out_params = count_params(diffusion_model.out)
        trainable_params += out_params
        trainable_breakdown["out"] = out_params
        if train_image_cn:
            img_out_params = count_params(diffusion_model.image_out)
            trainable_params += img_out_params
            trainable_breakdown["image_out"] = img_out_params
        else:
            frozen_params += count_params(diffusion_model.image_out)
    else:
        frozen_params += components["Mask Decoder (output_blocks)"]
        frozen_params += components["Image Decoder (image_output_blocks)"]
        frozen_params += components["Mask Out (out)"]
        frozen_params += components["Image Out (image_out)"]

    # VAE and CLIP are always frozen (inference only)
    frozen_params += components["VAE (first_stage_model)"]
    frozen_params += components["CLIP (cond_stage_model)"]

    total_params = trainable_params + frozen_params

    # VRAM estimation (bf16-mixed precision training)
    # Model weights: all params in bf16 (2 bytes each)
    model_weights_bytes = total_params * 2

    # Master weights: trainable params in fp32 (4 bytes) for AdamW
    master_weights_bytes = trainable_params * 4

    # AdamW optimizer states: momentum (fp32) + variance (fp32) = 8 bytes per trainable param
    optimizer_bytes = trainable_params * 8

    # Gradients: trainable params in bf16 (2 bytes)
    gradient_bytes = trainable_params * 2

    # Determine batch size
    if batch_size_override is not None:
        batch_size = batch_size_override
    elif training_target == "workstation":
        batch_size = 1 if train_image_cn else 2
    elif training_target in ("dgx_single", "dgx_multi"):
        batch_size = 4
    else:
        batch_size = 1

    # Activation memory
    activation_bytes = estimate_activation_memory(batch_size)

    # Deep copy overhead
    deep_copy_bytes = estimate_deep_copy_overhead(model)

    # Total
    total_vram = (model_weights_bytes + master_weights_bytes + optimizer_bytes +
                  gradient_bytes + activation_bytes + deep_copy_bytes)

    # Print report
    print(f"\n{'='*70}")
    print(f"  PRESET: {preset_name}")
    print(f"  {preset_config.get('desc', '')}")
    print(f"{'='*70}")

    print(f"\n  Model Architecture:")
    for name, count in components.items():
        print(f"    {name:45s} {fmt_params(count):>10s}")
    print(f"    {'─'*55}")
    print(f"    {'TOTAL':45s} {fmt_params(sum(components.values())):>10s}")

    print(f"\n  Training Configuration:")
    print(f"    sd_locked={sd_locked}  train_mask_cn={train_mask_cn}  train_image_cn={train_image_cn}")
    print(f"    unlock_last_n={unlock_last_n}  batch_size={batch_size}")

    print(f"\n  Trainable Parameters ({fmt_params(trainable_params)}):")
    for name, count in trainable_breakdown.items():
        print(f"    {name:45s} {fmt_params(count):>10s}")
    print(f"  Frozen Parameters: {fmt_params(frozen_params)}")

    print(f"\n  VRAM Breakdown (bf16-mixed, batch={batch_size}):")
    print(f"    Model weights (bf16):          {fmt_gb(model_weights_bytes):>10s}")
    print(f"    Master weights (fp32):         {fmt_gb(master_weights_bytes):>10s}")
    print(f"    AdamW states (fp32):           {fmt_gb(optimizer_bytes):>10s}")
    print(f"    Gradients (bf16):              {fmt_gb(gradient_bytes):>10s}")
    print(f"    Activations (est., ckpt):      {fmt_gb(activation_bytes):>10s}")
    print(f"    Deep-copy overhead (est.):     {fmt_gb(deep_copy_bytes):>10s}")
    print(f"    {'─'*45}")
    total_gb = total_vram / (1024**3)
    fits = "✅" if total_gb <= 24 else "❌"
    print(f"    TOTAL:                         {fmt_gb(total_vram):>10s}  {fits} (24 GB RTX 4090)")

    if total_gb <= 80:
        h100_fits = "✅"
    else:
        h100_fits = "❌"
    print(f"                                              {h100_fits} (80 GB H100)")

    return {
        "preset": preset_name,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "total_vram_gb": total_gb,
        "fits_4090": total_gb <= 24,
        "fits_h100": total_gb <= 80,
        "batch_size": batch_size,
    }


def main():
    parser = argparse.ArgumentParser(description="ADC VRAM Calculator")
    parser.add_argument("preset", nargs="?", default=None,
                        help="Preset name (default: all presets)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size for estimation")
    parser.add_argument("--target", default="workstation",
                        choices=["mps", "workstation", "dgx_single", "dgx_multi"],
                        help="Training target (default: workstation)")
    args = parser.parse_args()

    print("Loading model architecture (no weights)...")
    model = create_model('./models/cldm_v15.yaml').cpu()

    # Import presets from training script (safe — guarded by if __name__ == "__main__")
    from tutorial_train_single_gpu import PRESETS

    # Quick total model stats
    total = count_unique_params(model)
    print(f"Total model parameters: {fmt_params(total)} ({total:,})")

    presets_to_analyze = ([args.preset] if args.preset else list(PRESETS.keys()))

    results = []
    for name in presets_to_analyze:
        if name not in PRESETS:
            print(f"\n⚠️  Unknown preset: {name}")
            continue
        r = analyze_preset(name, PRESETS[name], model,
                           batch_size_override=args.batch,
                           training_target=args.target)
        results.append(r)

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Preset':<35s} {'Trainable':>10s} {'VRAM':>8s} {'4090':>5s} {'H100':>5s} {'BS':>3s}")
        print(f"  {'─'*67}")
        for r in results:
            print(f"  {r['preset']:<35s} {fmt_params(r['trainable']):>10s} "
                  f"{r['total_vram_gb']:>7.1f}G "
                  f"{'✅' if r['fits_4090'] else '❌':>5s} "
                  f"{'✅' if r['fits_h100'] else '❌':>5s} "
                  f"{r['batch_size']:>3d}")


if __name__ == "__main__":
    main()
