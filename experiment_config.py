"""Shared experiment defaults for ADC training and evaluation."""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np
import pytorch_lightning as pl
import torch

DEFAULT_SEED: Final[int] = 42
DEFAULT_NUM_FOLDS: Final[int] = 5
DEFAULT_FOLD_INDEX: Final[int] = 0
DEFAULT_VALIDATION_INTERVAL_STEPS: Final[int] = 2000
DEFAULT_EARLY_STOP_PATIENCE: Final[int] = 3
DEFAULT_VALIDATION_DDIM_STEPS: Final[int] = 50
DEFAULT_EVAL_CFG_SCALE: Final[float] = 9.0
DEFAULT_TEST_DDIM_STEPS: Final[int] = 50
DEFAULT_TEST_SEEDS: Final[tuple[int, ...]] = tuple(range(10))
DEFAULT_DINOV2_MODEL_NAME: Final[str] = "facebook/dinov2-base"
DEFAULT_LPIPS_BACKBONE: Final[str] = "alex"
DEFAULT_MIOU_THRESHOLD: Final[float] = 0.5
DEFAULT_SEGMENTATION_MODEL: Final[str] = "segmentation_integration:MinimalSegModel"

PRESET_MAX_STEPS: Final[dict[str, int]] = {
    "scratch": 20000,
    "polyp_transfer": 20000,
    "scratch_unlocked": 10000,
    "polyp_unlocked": 10000,
    "polyp_stage2": 10000,
    "scratch_stage2": 10000,
    "polyp_stage2_from_unlocked": 10000,
    "paper_faithful_polyp": 24000,
    "paper_faithful_scratch": 24000,
    "paper_faithful_v2_polyp": 30000,
}


def resolve_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def resolve_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def resolve_seed(env_name: str = "SEED", default: int = DEFAULT_SEED) -> int:
    return resolve_int_env(env_name, default)


def resolve_num_folds(env_name: str = "NUM_FOLDS", default: int = DEFAULT_NUM_FOLDS) -> int:
    return max(1, resolve_int_env(env_name, default))


def resolve_fold_index(
    env_name: str = "FOLD_INDEX",
    default: int = DEFAULT_FOLD_INDEX,
    num_folds: int | None = None,
) -> int:
    fold_index = resolve_int_env(env_name, default)
    if fold_index < 0:
        raise ValueError(f"{env_name} must be >= 0, got {fold_index}")
    if num_folds is not None and fold_index >= num_folds:
        raise ValueError(f"{env_name} must be < {num_folds}, got {fold_index}")
    return fold_index


def resolve_seed_list(env_name: str = "TEST_SEEDS", default: tuple[int, ...] = DEFAULT_TEST_SEEDS) -> tuple[int, ...]:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default
    values = [item.strip() for item in raw_value.split(",")]
    seeds = tuple(int(item) for item in values if item)
    return seeds or default


def resolve_max_steps(preset_name: str, env_name: str = "MAX_STEPS") -> int:
    return resolve_int_env(env_name, PRESET_MAX_STEPS[preset_name])


def resolve_validation_interval_steps(env_name: str = "VALIDATION_INTERVAL_STEPS") -> int:
    return resolve_int_env(env_name, DEFAULT_VALIDATION_INTERVAL_STEPS)


def resolve_early_stop_patience(env_name: str = "EARLY_STOP_PATIENCE") -> int:
    return resolve_int_env(env_name, DEFAULT_EARLY_STOP_PATIENCE)


def set_global_seed(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
