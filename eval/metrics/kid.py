"""KID with DINOv2 ViT-L/14 features.

Implements ``DINOv2Features``, an ``nn.Module`` wrapping
``facebook/dinov2-large`` (ViT-L/14) and exposing CLS-token embeddings of
dimension 1024. The wrapper is passed as the ``feature=`` argument to
``torchmetrics.image.kid.KernelInceptionDistance`` so that KID is computed
on DINOv2 features rather than the default InceptionV3-2048.

Rationale (see ``working/notes/metrics_kid_lpips.md`` §4 + plan §0):
- Stein et al. (NeurIPS 2023) established DINOv2 ViT-L/14 as the de-facto
  standard backbone for diffusion-model KID since 2023.
- DINOv2 is Apache-2.0; DINOv3 (Aug 2025) is too new and commercial-licensed.
- Migrating to DINOv3 later is a one-line swap (same nn.Module interface).

Input convention: ``forward(x)`` expects ``x: [B, 3, H, W]`` in [0, 1] range
(``normalize=True`` on the KID side handles the [0, 1] → wrapper path). The
wrapper internally resizes to 224x224 (bilinear) and applies ImageNet
normalization, matching DINOv2's pre-training convention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.image.kid import KernelInceptionDistance

# ImageNet stats (DINOv2 pre-training convention).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Default DINOv2 model identifier — change to "facebook/dinov2-base" to
# trade quality for speed/VRAM if needed.
_DEFAULT_MODEL_ID = "facebook/dinov2-large"
_DEFAULT_INPUT_SIZE = 224
_FEATURE_DIM = 1024  # CLS token width for ViT-L/14


class DINOv2Features(nn.Module):
    """Frozen DINOv2 backbone returning CLS-token embeddings.

    Suitable as the ``feature=`` argument to torchmetrics' KID. Output shape
    is ``[B, 1024]``.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        input_size: int = _DEFAULT_INPUT_SIZE,
    ) -> None:
        super().__init__()
        # Lazy import so this module can be imported without transformers
        # installed (e.g. for static checks). Real use requires transformers.
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_id)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.input_size = input_size
        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        )

    @torch.inference_mode()
    def forward(self, x: Tensor) -> Tensor:
        """Map ``[B, 3, H, W]`` in [0, 1] to ``[B, 1024]`` CLS embeddings."""
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(
                f"Expected x of shape [B, 3, H, W], got {tuple(x.shape)}"
            )
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self._mean) / self._std
        out = self.backbone(pixel_values=x)
        # transformers ViT-style output: last_hidden_state[:, 0] is the CLS token.
        return out.last_hidden_state[:, 0]


def make_kid(
    subset_size: int = 50,
    subsets: int = 50,
    model_id: str = _DEFAULT_MODEL_ID,
    input_size: int = _DEFAULT_INPUT_SIZE,
) -> KernelInceptionDistance:
    """Construct a ``KernelInceptionDistance`` configured with DINOv2 features.

    Defaults follow Bińkowski et al. (2018) recommendations for sample-efficient
    estimation. Call ``.update(images, real=True/False)`` then ``.compute()``
    to get ``(mean, std)``.

    Note: ``normalize=True`` tells torchmetrics that input images are already
    in [0, 1] range; the wrapper handles ImageNet-normalization internally.
    """
    feature = DINOv2Features(model_id=model_id, input_size=input_size)
    return KernelInceptionDistance(
        feature=feature,
        subset_size=subset_size,
        subsets=subsets,
        normalize=True,
    )


__all__ = ["DINOv2Features", "make_kid", "_FEATURE_DIM"]
