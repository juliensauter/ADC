import torch
from torch import Tensor
from typing import Iterable
from eval.metrics import make_lpips, make_dreamsim, make_kid, boundary_iou


def score_generated_stream(
    gen_stream: Iterable[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    kid_subset_size: int = 50,
    kid_subsets: int = 50,
    segmenter: torch.nn.Module | None = None,
) -> dict:
    """Computes evaluation metrics over a stream of generated batches.

    Args:
        gen_stream: Iterator yielding (gen_imgs, ref_imgs, masks)
        device: Device to perform computations on
        kid_subset_size: Subset size for KID
        kid_subsets: Number of subsets for KID bootstrapping
        segmenter: Optional pretrained segmentation model (Phase C)

    Returns:
        Dict containing aggregated metric mean and standard deviation values.
    """
    lpips_fn = make_lpips().to(device)
    dreamsim_fn = make_dreamsim(device=device)
    kid_fn = make_kid(subset_size=kid_subset_size, subsets=kid_subsets).to(device)

    biou_scores = []
    total_samples = 0

    for gen_imgs, ref_imgs, masks in gen_stream:
        batch_size = gen_imgs.size(0)
        total_samples += batch_size

        # Move to target device for metric updates
        gen_imgs = gen_imgs.to(device)
        ref_imgs = ref_imgs.to(device)
        masks = masks.to(device)

        # 1. Paired perceptual metric updates
        lpips_fn.update(gen_imgs, ref_imgs)
        dreamsim_fn.update(gen_imgs, ref_imgs)

        # 2. Distributional KID update
        kid_fn.update(ref_imgs, real=True)
        kid_fn.update(gen_imgs, real=False)

        # 3. Dormant segmenter-based boundary IoU check (Phase C hook)
        if segmenter is not None:
            with torch.no_grad():
                pred_masks = segmenter(gen_imgs)
                biou = boundary_iou(pred_masks, masks)
                biou_scores.extend(biou.cpu().tolist())

    # Compute metric values
    lpips_val = float(lpips_fn.compute().item())
    dreamsim_val = float(dreamsim_fn.compute().item())

    # Safeguard KID estimation against too few samples
    if total_samples >= kid_subset_size:
        kid_mean, kid_std = kid_fn.compute()
        kid_mean_val = float(kid_mean.item())
        kid_std_val = float(kid_std.item())
    else:
        print(
            f"Warning: Total generated samples ({total_samples}) is less than "
            f"kid_subset_size ({kid_subset_size}). KID skipped."
        )
        kid_mean_val = float("nan")
        kid_std_val = float("nan")

    biou_val = (
        float(torch.tensor(biou_scores).mean().item())
        if biou_scores
        else float("nan")
    )

    return {
        "kid_mean": kid_mean_val,
        "kid_std": kid_std_val,
        "lpips_mean": lpips_val,
        "dreamsim_mean": dreamsim_val,
        "boundary_iou_mean": biou_val,
    }

