import torch
import pytorch_lightning as pl
from torch import Tensor
from typing import Generator
from cldm.model import create_model, load_state_dict


def load_model(checkpoint_path: str, device: torch.device) -> pl.LightningModule:
    """Load the ADC checkpoint using cldm model structures.

    Forces float32 on CPU/MPS for compatibility.
    """
    model = create_model("./models/cldm_v15.yaml").cpu()

    if device.type in ("mps", "cpu"):
        model = model.float()

    state_dict = load_state_dict(checkpoint_path, location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # Standard inference config parameters matching local tutorial script
    model.learning_rate = 1e-5
    model.sd_locked = False
    model.only_mid_control = False

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def generate_batched_images(
    model: pl.LightningModule,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
    cfg_scale: float = 9.0,
) -> Generator[tuple[Tensor, Tensor, Tensor], None, None]:
    """Generates images batch-by-batch from mask and reference loaders.

    Yields:
        Tuple (gen_imgs, ref_imgs, masks) as float32 tensors on CPU, in [0, 1] range.
        - gen_imgs: [B, 3, H, W] mask-conditioned generated images
        - ref_imgs: [B, 3, H, W] reference real images
        - masks: [B, 3, H, W] mask conditioning hint
    """
    with model.ema_scope():
        for batch in dataloader:
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, Tensor):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v

            images = model.log_images(
                batch_on_device,
                N=batch["jpg"].size(0),
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                unconditional_guidance_scale=cfg_scale,
            )

            # Match standard mask samples generated output key
            key = f"samples_cfg_scale_{cfg_scale:.2f}_mask"
            if key not in images:
                keys = [k for k in images.keys() if "samples" in k and "mask" in k]
                if keys:
                    key = keys[0]
                else:
                    raise KeyError(
                        f"Could not find mask-conditioned output key in log_images: {list(images.keys())}"
                    )

            gen_raw = images[key]
            ref_raw = batch_on_device["jpg"]
            masks_raw = batch_on_device["hint"]

            # Rearrange shapes from [B, H, W, C] to [B, C, H, W] for metric compat
            ref_raw = ref_raw.permute(0, 3, 1, 2)
            masks_raw = masks_raw.permute(0, 3, 1, 2)

            # Normalize values from [-1, 1] to [0, 1] for metric ingestion
            gen_imgs = torch.clamp((gen_raw + 1.0) / 2.0, 0.0, 1.0).cpu()
            ref_imgs = torch.clamp((ref_raw + 1.0) / 2.0, 0.0, 1.0).cpu()
            masks = masks_raw.cpu()

            yield gen_imgs, ref_imgs, masks

