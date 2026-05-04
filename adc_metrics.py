"""Reusable quality-metric helpers for ADC training and evaluation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, Sequence

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch import nn
from torchvision import transforms
from transformers import Dinov2Model

from experiment_config import (
    DEFAULT_DINOV2_MODEL_NAME,
    DEFAULT_LPIPS_BACKBONE,
    DEFAULT_MIOU_THRESHOLD,
    DEFAULT_SEGMENTATION_MODEL,
)

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _ensure_batch_tensor(images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    if isinstance(images, torch.Tensor):
        return images.detach().cpu()
    if not images:
        raise ValueError("Expected at least one image batch.")
    return torch.cat([image.detach().cpu() for image in images], dim=0)


def _to_unit_range(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    if images.max() > 1.5:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0
    return images.clamp(0.0, 1.0)


def save_rgb_tensor(image: torch.Tensor, path: str | Path) -> None:
    image = image.detach().cpu().float()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("save_rgb_tensor expects a single image or a batch of size 1.")
        image = image[0]
    image = _to_unit_range(image.unsqueeze(0))[0]
    array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_mask_tensor(mask: torch.Tensor, path: str | Path) -> None:
    mask = mask.detach().cpu().float()
    if mask.ndim == 4:
        if mask.shape[0] != 1:
            raise ValueError("save_mask_tensor expects a single mask or a batch of size 1.")
        mask = mask[0]
    if mask.ndim == 3:
        if mask.shape[0] in (1, 3):
            mask = mask[0]
        else:
            mask = mask.squeeze(0)
    if mask.max() > 1.5:
        mask = mask / 255.0
    elif mask.min() < 0.0:
        mask = (mask + 1.0) / 2.0
    array = (mask.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).convert("L").save(path)


def load_object_from_spec(spec: str):
    module_name, _, attribute_name = spec.partition(":")
    if not module_name or not attribute_name:
        raise ValueError(f"Expected 'module:attribute' spec, got {spec!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def load_segmentation_model(
    model_spec: str | None = None,
    checkpoint_path: str | None = None,
    device: torch.device | str = "cpu",
    strict: bool = False,
) -> tuple[nn.Module, str]:
    if model_spec is None:
        model_spec = DEFAULT_SEGMENTATION_MODEL

    if ":" in model_spec:
        model_factory = load_object_from_spec(model_spec)
        model = model_factory() if callable(model_factory) else model_factory
        model_name = model_spec
    else:
        model_factory = load_object_from_spec(model_spec)
        model = model_factory() if callable(model_factory) else model_factory
        model_name = model_spec

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=strict)

    model = model.to(device)
    model.eval()
    return model, model_name


class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, model_name: str = DEFAULT_DINOV2_MODEL_NAME, device: torch.device | str = "cpu"):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.backbone.eval()
        self.device = torch.device(device)
        self.backbone.to(self.device)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _to_unit_range(images)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        outputs = self.backbone(pixel_values=images)
        features = outputs.last_hidden_state[:, 0]
        return features.float()


def extract_dinov2_features(
    images: torch.Tensor | Sequence[torch.Tensor],
    extractor: DinoV2FeatureExtractor,
    batch_size: int = 8,
) -> torch.Tensor:
    images = _ensure_batch_tensor(images)
    feature_batches = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(extractor.device)
        feature_batches.append(extractor(batch).detach().cpu())
    return torch.cat(feature_batches, dim=0)


def _polynomial_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    degree: int = 3,
    gamma: float | None = None,
    coef: float = 1.0,
) -> torch.Tensor:
    if gamma is None:
        gamma = 1.0 / left.shape[1]
    return (gamma * left @ right.T + coef) ** degree


def compute_kid_from_features(
    real_features: torch.Tensor | np.ndarray,
    fake_features: torch.Tensor | np.ndarray,
    degree: int = 3,
    gamma: float | None = None,
    coef: float = 1.0,
) -> float:
    real = torch.as_tensor(real_features, dtype=torch.float64)
    fake = torch.as_tensor(fake_features, dtype=torch.float64)
    if real.shape[0] < 2 or fake.shape[0] < 2:
        return float("nan")

    k_rr = _polynomial_kernel(real, real, degree=degree, gamma=gamma, coef=coef)
    k_ff = _polynomial_kernel(fake, fake, degree=degree, gamma=gamma, coef=coef)
    k_rf = _polynomial_kernel(real, fake, degree=degree, gamma=gamma, coef=coef)

    n_real = real.shape[0]
    n_fake = fake.shape[0]
    sum_rr = (k_rr.sum() - k_rr.diag().sum()) / (n_real * (n_real - 1))
    sum_ff = (k_ff.sum() - k_ff.diag().sum()) / (n_fake * (n_fake - 1))
    sum_rf = k_rf.mean()
    kid = sum_rr + sum_ff - 2.0 * sum_rf
    return float(kid.item())


def compute_fdd_from_features(
    real_features: torch.Tensor | np.ndarray,
    fake_features: torch.Tensor | np.ndarray,
) -> float:
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
    if real.shape[0] < 2 or fake.shape[0] < 2:
        return float("nan")

    mean_real = real.mean(axis=0)
    mean_fake = fake.mean(axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_fake = np.cov(fake, rowvar=False)

    if cov_real.ndim == 0:
        cov_real = np.array([[cov_real]])
    if cov_fake.ndim == 0:
        cov_fake = np.array([[cov_fake]])

    eps = 1e-6
    cov_real = cov_real + np.eye(cov_real.shape[0]) * eps
    cov_fake = cov_fake + np.eye(cov_fake.shape[0]) * eps

    cov_sqrt = linalg.sqrtm(cov_real @ cov_fake)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    diff = mean_real - mean_fake
    fdd = diff.dot(diff) + np.trace(cov_real + cov_fake - 2.0 * cov_sqrt)
    return float(np.real(fdd))


class QualityMetrics:
    def __init__(
        self,
        device: torch.device | str = "cpu",
        dinov2_model_name: str = DEFAULT_DINOV2_MODEL_NAME,
        lpips_backbone: str = DEFAULT_LPIPS_BACKBONE,
        segmentation_model: nn.Module | None = None,
        segmentation_threshold: float = DEFAULT_MIOU_THRESHOLD,
    ):
        self.device = torch.device(device)
        self.extractor = DinoV2FeatureExtractor(model_name=dinov2_model_name, device=self.device)
        self.lpips = lpips.LPIPS(net=lpips_backbone).to(self.device)
        self.lpips.eval()
        self.segmentation_model = segmentation_model.to(self.device).eval() if segmentation_model is not None else None
        self.segmentation_threshold = segmentation_threshold

    def _stack_batches(self, batches: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        return _ensure_batch_tensor(batches)

    def compute_kid(self, real_batches: torch.Tensor | Sequence[torch.Tensor], fake_batches: torch.Tensor | Sequence[torch.Tensor]) -> float:
        real = self._stack_batches(real_batches)
        fake = self._stack_batches(fake_batches)
        real_features = extract_dinov2_features(real, self.extractor)
        fake_features = extract_dinov2_features(fake, self.extractor)
        return compute_kid_from_features(real_features, fake_features)

    def compute_fdd(self, real_batches: torch.Tensor | Sequence[torch.Tensor], fake_batches: torch.Tensor | Sequence[torch.Tensor]) -> float:
        real = self._stack_batches(real_batches)
        fake = self._stack_batches(fake_batches)
        real_features = extract_dinov2_features(real, self.extractor)
        fake_features = extract_dinov2_features(fake, self.extractor)
        return compute_fdd_from_features(real_features, fake_features)

    def compute_lpips(self, real_batches: torch.Tensor | Sequence[torch.Tensor], fake_batches: torch.Tensor | Sequence[torch.Tensor], batch_size: int = 4) -> float:
        real = self._stack_batches(real_batches)
        fake = self._stack_batches(fake_batches)
        if real.shape[0] < 1:
            return float("nan")

        scores = []
        for start in range(0, real.shape[0], batch_size):
            real_batch = real[start : start + batch_size].to(self.device)
            fake_batch = fake[start : start + batch_size].to(self.device)
            real_batch = _to_unit_range(real_batch)
            fake_batch = _to_unit_range(fake_batch)
            real_batch = F.interpolate(real_batch, size=(256, 256), mode="bilinear", align_corners=False)
            fake_batch = F.interpolate(fake_batch, size=(256, 256), mode="bilinear", align_corners=False)
            real_batch = real_batch * 2.0 - 1.0
            fake_batch = fake_batch * 2.0 - 1.0
            with torch.no_grad():
                score = self.lpips(fake_batch, real_batch)
            scores.append(score.detach().float().mean().cpu())

        return float(torch.stack(scores).mean().item())

    def predict_segmentation_masks(self, image_batches: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if self.segmentation_model is None:
            raise RuntimeError("No segmentation model is configured.")

        images = _to_unit_range(self._stack_batches(image_batches)).to(self.device)
        with torch.no_grad():
            logits = self.segmentation_model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.shape[1] == 1:
                masks = (torch.sigmoid(logits) >= self.segmentation_threshold).long().squeeze(1)
            else:
                masks = logits.argmax(dim=1).long()
        return masks.detach().cpu()

    def compute_miou(
        self,
        image_batches: torch.Tensor | Sequence[torch.Tensor],
        target_masks: torch.Tensor | Sequence[torch.Tensor],
    ) -> float:
        pred_masks = self.predict_segmentation_masks(image_batches)
        target_masks = self._stack_batches(target_masks)
        if target_masks.ndim == 4:
            if target_masks.shape[1] in (1, 3):
                target_masks = target_masks[:, 0]
            else:
                target_masks = target_masks.squeeze(1)
        elif target_masks.ndim == 3 and target_masks.shape[0] in (1, 3):
            target_masks = target_masks[0]
        if target_masks.max() > 1.5:
            target_masks = (target_masks / 255.0 >= self.segmentation_threshold).long()
        else:
            target_masks = target_masks.long()

        num_classes = int(max(pred_masks.max().item(), target_masks.max().item()) + 1)
        ious = []
        for class_index in range(num_classes):
            pred_class = pred_masks == class_index
            target_class = target_masks == class_index
            union = pred_class | target_class
            if union.sum() == 0:
                continue
            intersection = pred_class & target_class
            ious.append((intersection.sum().float() / union.sum().float()).item())

        if not ious:
            return float("nan")
        return float(np.mean(ious))

    @staticmethod
    def composite_score(kid: float, lpips_score: float) -> float:
        return float(kid + lpips_score)
