"""Training callbacks for periodic validation-quality evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from adc_metrics import QualityMetrics, save_mask_tensor, save_rgb_tensor


class ValidationQualityCallback(Callback):
    def __init__(
        self,
        val_dataset,
        save_root: str,
        interval_steps: int = 2000,
        patience: int = 3,
        ddim_steps: int = 50,
        cfg_scale: float = 9.0,
        metric_device: str = "cpu",
        metric_batch_size: int = 4,
        num_workers: int = 0,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.interval_steps = interval_steps
        self.patience = patience
        self.ddim_steps = ddim_steps
        self.cfg_scale = cfg_scale
        self.metric_device = metric_device
        self.metric_batch_size = metric_batch_size
        self.num_workers = num_workers
        self.metrics = QualityMetrics(device=metric_device)
        self.best_score: float | None = None
        self.best_metrics: dict[str, float] | None = None
        self.bad_events = 0
        self.save_root = Path(save_root)
        self.best_checkpoint_path = self.save_root / "checkpoints" / "best.ckpt"

    def _artifact_root(self, trainer) -> Path:
        log_dir = getattr(trainer, "log_dir", None)
        if log_dir:
            return Path(log_dir)
        return self.save_root

    def _should_evaluate(self, trainer) -> bool:
        return trainer.global_step > 0 and trainer.global_step % self.interval_steps == 0

    def _evaluate_batch(self, pl_module, batch):
        batch = {
            key: value.to(pl_module.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            images = pl_module.log_images(
                batch,
                N=batch["jpg"].shape[0],
                ddim_steps=self.ddim_steps,
                unconditional_guidance_scale=self.cfg_scale,
            )

        output_key = f"samples_cfg_scale_{self.cfg_scale:.2f}_mask"
        generated = images[output_key].detach().cpu()
        target = batch["jpg"].detach().cpu()
        mask = batch["hint"].detach().cpu()
        return generated, target, mask

    def on_validation_end(self, trainer, pl_module):
        if not self._should_evaluate(trainer):
            return
        if not trainer.is_global_zero:
            stop_tensor = torch.zeros(1, device=pl_module.device, dtype=torch.int32)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(stop_tensor, src=0)
            trainer.should_stop = bool(stop_tensor.item())
            return

        artifact_root = self._artifact_root(trainer)
        self.best_checkpoint_path = artifact_root / "checkpoints" / "best.ckpt"

        loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

        generated_batches = []
        target_batches = []
        mask_batches = []

        output_dir = artifact_root / "validation_metrics" / f"step_{trainer.global_step:06d}"
        sample_dir = output_dir / "generated"
        target_dir = output_dir / "target"
        mask_dir = output_dir / "mask"
        sample_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad(), pl_module.ema_scope():
            for batch_index, batch in enumerate(loader):
                generated, target, mask = self._evaluate_batch(pl_module, batch)
                generated_batches.append(generated)
                target_batches.append(target)
                mask_batches.append(mask)

                save_rgb_tensor(generated[0], sample_dir / f"{batch_index:06d}.png")
                save_rgb_tensor((target[0] + 1.0) / 2.0, target_dir / f"{batch_index:06d}.png")
                save_mask_tensor(mask[0], mask_dir / f"{batch_index:06d}.png")

        if was_training:
            pl_module.train()

        kid_score = self.metrics.compute_kid(target_batches, generated_batches)
        lpips_score = self.metrics.compute_lpips(target_batches, generated_batches, batch_size=self.metric_batch_size)
        composite_score = self.metrics.composite_score(kid_score, lpips_score)

        metrics = {
            "val/kid": kid_score,
            "val/lpips": lpips_score,
            "val/composite": composite_score,
        }
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        improved = self.best_score is None or composite_score < self.best_score
        if improved:
            self.best_score = composite_score
            self.best_metrics = metrics
            self.bad_events = 0
            self.best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(self.best_checkpoint_path))
        else:
            self.bad_events += 1

        if trainer.is_global_zero:
            best_score_text = f"{self.best_score:.6f}" if self.best_score is not None else "nan"
            print(
                f"[validation] step={trainer.global_step} kid={kid_score:.6f} lpips={lpips_score:.6f} "
                f"score={composite_score:.6f} best={best_score_text} "
                f"bad_events={self.bad_events}/{self.patience}"
            )

        stop_now = self.bad_events >= self.patience
        stop_tensor = torch.tensor([1 if stop_now else 0], device=pl_module.device, dtype=torch.int32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(stop_tensor, src=0)
        trainer.should_stop = bool(stop_tensor.item())
