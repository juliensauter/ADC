"""LPIPS-AlexNet adapter (torchmetrics native).

Thin wrapper around
``torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity`` enforcing a
single input convention across our eval pipeline.

Rationale (see ``working/notes/metrics_kid_lpips.md`` §2 + §3):
- LPIPS is a *paired* perceptual metric: it compares matched (gen_i, real_i)
  image pairs, NOT two distributions (that is KID's job). Lower = more similar.
- We keep the original AlexNet backbone. LPIPS is a learned linear calibration
  on top of fixed backbone features (BAPPS human judgements); the backbone is
  NOT swappable to DINO without invalidating the metric. DreamSim is our
  modern DINO-based perceptual complement (see ``dreamsim.py``).
- AlexNet specifically is the de-facto comparability standard reported across
  diffusion papers since 2018.

Input convention: ``[N, 3, H, W]`` in [0, 1] range (``normalize=True`` handles
the [0, 1] -> internal [-1, 1] mapping), matching ``kid.py``.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

_DEFAULT_NET = "alex"
_DEFAULT_INPUT_SIZE = 256


class NormalizedResizedLPIPS(LearnedPerceptualImagePatchSimilarity):
    """Subclass of LearnedPerceptualImagePatchSimilarity that enforces resizing.

    Forces input tensors to 256x256 before computing LPIPS, matching the
    resolution the weights were trained on and ensuring literature comparability.
    """

    def __init__(
        self,
        net_type: str = _DEFAULT_NET,
        reduction: str = "mean",
        normalize: bool = True,
        input_size: int = _DEFAULT_INPUT_SIZE,
        **kwargs,
    ) -> None:
        super().__init__(net_type=net_type, reduction=reduction, normalize=normalize, **kwargs)
        self.input_size = input_size

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Update state with resized matched image pairs."""
        if img1.shape[-2:] != (self.input_size, self.input_size):
            img1 = F.interpolate(
                img1,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        if img2.shape[-2:] != (self.input_size, self.input_size):
            img2 = F.interpolate(
                img2,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        super().update(img1, img2)


def make_lpips(
    net_type: str = _DEFAULT_NET,
    reduction: str = "mean",
) -> LearnedPerceptualImagePatchSimilarity:
    """Construct an LPIPS metric configured for our [0, 1] paired-input setup.

    Call ``.update(img1, img2)`` with matched pairs (both ``[N, 3, H, W]`` in
    [0, 1]) then ``.compute()`` for the mean LPIPS distance.
    """
    return NormalizedResizedLPIPS(
        net_type=net_type,
        reduction=reduction,
        normalize=True,
    )


__all__ = ["make_lpips", "NormalizedResizedLPIPS"]

