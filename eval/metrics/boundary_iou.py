import torch
import torch.nn.functional as F
from torch import Tensor


def boundary_iou(
    pred_mask: Tensor,
    gt_mask: Tensor,
    iterations: int = 5,
) -> Tensor:
    """Compute Boundary IoU at the specified pixel dilation iteration (default 5px).

    Args:
        pred_mask: Binary mask tensor of shape [..., H, W] containing values 0 or 1.
        gt_mask: Binary mask tensor of shape [..., H, W] containing values 0 or 1.
        iterations: Number of pixels to erode (boundary width). Default is 5.

    Returns:
        Tensor of shape [...] containing boundary IoU scores in [0, 1].
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch: {pred_mask.shape} vs {gt_mask.shape}"
        )

    # Ensure shape is at least 4D [..., C, H, W] for max_pool2d
    orig_shape = pred_mask.shape
    if len(orig_shape) < 3:
        # e.g., [H, W] -> [1, 1, H, W]
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    elif len(orig_shape) == 3:
        # e.g., [C, H, W] -> [1, C, H, W]
        pred_mask = pred_mask.unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0)

    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    # Erosion structured kernel size = 2 * iterations + 1
    kernel_size = 2 * iterations + 1
    padding = iterations

    # PyTorch morphological binary erosion: 1 - dilation(1 - mask)
    # We use max_pool2d for fast multi-pixel dilation.
    pred_eroded = 1.0 - F.max_pool2d(
        1.0 - pred_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    gt_eroded = 1.0 - F.max_pool2d(
        1.0 - gt_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    # Boundary is the mask XOR eroded_mask
    pred_bound = pred_mask - pred_eroded
    gt_bound = gt_mask - gt_eroded

    # Compute intersection and union along spatial dimensions
    intersection = (pred_bound * gt_bound).sum(dim=(-2, -1))
    union = (pred_bound + gt_bound).clamp(0, 1).sum(dim=(-2, -1))

    # Match standard IoU behaviors:
    # If both boundaries are empty (no mask contour present), IoU is 1.0.
    iou = torch.where(
        union == 0.0,
        torch.ones_like(union),
        intersection / union,
    )

    # Restore original batch dimensions if we added any
    if len(orig_shape) < 3:
        iou = iou.squeeze(0).squeeze(0)
    elif len(orig_shape) == 3:
        iou = iou.squeeze(0)

    return iou


__all__ = ["boundary_iou"]

