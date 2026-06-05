"""Metric wrappers (KID, LPIPS, DreamSim, Boundary IoU).

Each metric is implemented as a small adapter so the rest of the pipeline can
treat them uniformly. See `metrics_implementation_plan.md` §A.2–A.5 for
specs.
"""

from .boundary_iou import boundary_iou
from .dreamsim import DreamSimMetric, make_dreamsim
from .kid import DINOv2Features, make_kid
from .lpips import NormalizedResizedLPIPS, make_lpips

__all__ = [
    "boundary_iou",
    "make_dreamsim",
    "DreamSimMetric",
    "DINOv2Features",
    "make_kid",
    "make_lpips",
    "NormalizedResizedLPIPS",
]

