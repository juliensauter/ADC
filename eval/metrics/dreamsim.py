import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from dreamsim import dreamsim

_DEFAULT_TYPE = "ensemble"
_DEFAULT_CACHE_DIR = "models/dreamsim"
_DEFAULT_SIZE = 224


class DreamSimMetric(Metric):
    """Torchmetrics adapter for the DreamSim perceptual similarity metric.

    DreamSim measures mid-level semantic and layout similarity based on DINO/CLIP
    features. Inputs must be in the [0, 1] range. Images are resized to 224x224
    internally before distance calculation.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        pretrained: bool = True,
        dreamsim_type: str = _DEFAULT_TYPE,
        device: str | None = None,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device_str = device
        
        # Load the model
        self.model, _ = dreamsim(
            pretrained=pretrained,
            device=device,
            cache_dir=cache_dir,
            dreamsim_type=dreamsim_type,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        # State buffers
        self.add_state("sum_scores", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pairs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Accumulate distance scores between paired batch images."""
        if img1.shape != img2.shape:
            raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

        img1 = img1.to(self.device_str)
        img2 = img2.to(self.device_str)

        # Resize internally to 224x224 as required by the ViT backbone
        if img1.shape[-2:] != (_DEFAULT_SIZE, _DEFAULT_SIZE):
            img1 = F.interpolate(
                img1,
                size=(_DEFAULT_SIZE, _DEFAULT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            img2 = F.interpolate(
                img2,
                size=(_DEFAULT_SIZE, _DEFAULT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        with torch.no_grad():
            # returns a [B] distance tensor
            dist = self.model(img1, img2)

        self.sum_scores += dist.sum()
        self.total_pairs += img1.size(0)

    def compute(self) -> Tensor:
        """Compute mean distance."""
        if self.total_pairs == 0:
            return torch.tensor(0.0)
        return self.sum_scores / self.total_pairs


def make_dreamsim(
    pretrained: bool = True,
    dreamsim_type: str = _DEFAULT_TYPE,
    device: str | None = None,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> DreamSimMetric:
    """Construct a DreamSim metric wrapper."""
    return DreamSimMetric(
        pretrained=pretrained,
        dreamsim_type=dreamsim_type,
        device=device,
        cache_dir=cache_dir,
    )


__all__ = ["make_dreamsim", "DreamSimMetric"]

