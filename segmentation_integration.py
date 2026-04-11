"""
segmentation_integration.py
============================
End-to-end ADC + Segmentation model training.

ARCHITECTURE OVERVIEW
=====================
ADC generates synthetic images conditioned on segmentation masks.
This module connects ADC's generator output to a downstream segmentation model,
enabling a joint training loop where the segmentation loss backpropagates through
the generated images — improving synthesis quality for the segmentation task.

Two integration strategies are implemented:

  Strategy A: Differentiable single-step generation (fast, approximate)
    → Use a single-step DDIM (eta=1, T=1) or DDPM x0-prediction from noisy latent.
    → Gradient flows directly from seg_loss → decoder → UNet → ControlNet.
    → Approximation: not the full 50-step DDIM chain (not differentiable through time).
    → Suitable for: fast iteration, gradient signal shaping, distillation objectives.

  Strategy B: Latent space supervision (recommended for practical training)
    → Generate images with ADC (no_grad, full DDIM).
    → Freeze ADC weights. Train only the segmentation model on generated images.
    → Periodically regenerate the dataset with updated ADC weights.
    → Suitable for: stable training, standard segmentation training pipelines.

  Strategy C: Score distillation sampling (SDS) loss (research direction)
    → From DreamFusion / InstructPix2Pix: treat ADC as a score function.
    → Backprop a "guidance gradient" into the segmentation model's attention maps.
    → Suitable for: research into text-driven segmentation improvement.

RECOMMENDED STARTING POINT
============================
Start with Strategy B: it's stable, well-understood, and doesn't require
differentiating through the diffusion chain. Once you have a working baseline
segmentation model, revisit Strategy A for end-to-end gradient experiments.

SEGMENTATION MODEL INTERFACE
==============================
Your segmentation model should have this interface:
    class SegmentationModel(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # images: [B, 3, H, W] float32 in [0, 1] (or [-1, 1], adapt below)
            # returns logits: [B, num_classes, H, W]  OR  [B, 1, H, W] for binary
            ...

USAGE (Strategy B — recommended)
==================================
    1. Train ADC on liver data (run tutorial_train_single_gpu.py first)
    2. Generate a synthetic dataset (run tutorial_inference_local.py on all masks)
    3. Use this module to train the segmentation model:

        trainer = StrategyBTrainer(
            seg_model=your_segmentation_model,
            real_data_root="./data/images",
            gen_data_root="./generated_results/liver/images",
            mask_data_root="./data/masks",
        )
        trainer.train(epochs=50)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")


# ──────────────────────────────────────────────────────────────────────────────
# Strategy A: Differentiable single-step generation
# ──────────────────────────────────────────────────────────────────────────────

class SingleStepADCSegmentation(nn.Module):
    """
    Wraps ADC + segmentation model for end-to-end training.

    Forward pass:
      mask → ADC single-step decode → image → seg_model → seg_logits

    Loss:
      L = λ_diff * diffusion_loss(predicted_noise, noise) + λ_seg * seg_loss(logits, gt_mask)

    NOTE: Single-step generation is a rough approximation of the full DDIM chain.
    The gradient through the generation is valid but noisy compared to full sampling.
    """

    def __init__(
        self,
        adc_model,            # ControlLDM instance (from cldm.cldm)
        seg_model,            # your segmentation model (see interface above)
        lambda_diff: float = 1.0,        # weight for diffusion loss
        lambda_seg:  float = 0.5,        # weight for segmentation loss
        seg_input_range: str = "01",     # "01" or "-11" depending on seg_model input expected
        t_fixed: int = 200,              # timestep for single-step decode (0 = cleaner, 999 = noisier)
    ):
        super().__init__()
        self.adc       = adc_model
        self.seg       = seg_model
        self.lambda_diff = lambda_diff
        self.lambda_seg  = lambda_seg
        self.seg_input_range = seg_input_range
        self.t_fixed   = t_fixed

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to pixel space using VAE: [B, 4, H/8, W/8] → [B, 3, H, W] in [-1, 1]

        NOTE: no torch.no_grad() here — Strategy A requires gradients to flow
        back through the VAE decoder → UNet → ControlNet for the seg loss.
        """
        # Scale factor from ADC config: 1/0.18215
        z_scaled = 1.0 / self.adc.scale_factor * z
        x = self.adc.first_stage_model.decode(z_scaled)
        return x  # [-1, 1]

    def forward(self, batch: dict):
        """
        batch must contain:
          - 'hint':  [B, 3, H, W] binary mask in [0, 1]
          - 'jpg':   [B, 3, H, W] real image in [-1, 1]  (for diffusion loss)
          - 'txt':   list of prompt strings
          - 'gt_seg_mask': [B, H, W] integer class labels OR [B, 1, H, W] binary  (for seg loss)
        """
        # 1. Encode real image to latent
        x_start, cond = self.adc.get_input(batch, self.adc.first_stage_key)

        # 2. Sample timestep and add noise
        t = torch.full((x_start.shape[0],), self.t_fixed, device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.adc.q_sample(x_start=x_start, t=t, noise=noise)

        cond_image = {
            "c_crossattn":    cond["c_crossattn"],
            "c_concat":       cond["c_concat_mask"],
            "c_concat_image": cond["c_concat_image"],
        }

        # 3. Single-step noise prediction (differentiable)
        model_output = self.adc.apply_model(x_noisy, t, cond_image)

        # 4. Diffusion loss — main training signal for ADC
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]
        diff_loss = F.mse_loss(model_output, noise)

        # 5. Decode predicted x0 from the single-step prediction
        # x0_pred = (x_noisy - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)
        alphas_t = self.adc.alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_pred = (x_noisy - (1 - alphas_t).sqrt() * model_output) / alphas_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # 6. Decode to pixel space (differentiable through VAE decoder)
        gen_images = self.decode_latent(x0_pred)  # [-1, 1]

        # 7. Adapt to seg_model's input range
        if self.seg_input_range == "01":
            seg_input = (gen_images + 1.0) / 2.0    # [-1,1] → [0,1]
        else:
            seg_input = gen_images                    # keep [-1,1]

        # 8. Segmentation forward pass (gradient-connected to generation)
        seg_logits = self.seg(seg_input)

        # 9. Segmentation loss — ADAPT THIS to your seg_model's output format
        gt_mask = batch["gt_seg_mask"].to(x_start.device)
        if seg_logits.shape[1] == 1:
            # Binary segmentation: logits are [B, 1, H, W]
            seg_loss = F.binary_cross_entropy_with_logits(
                seg_logits.squeeze(1),
                gt_mask.float()
            )
        else:
            # Multi-class: logits are [B, num_classes, H, W]
            seg_loss = F.cross_entropy(seg_logits, gt_mask.long())

        # 10. Total loss
        total_loss = self.lambda_diff * diff_loss + self.lambda_seg * seg_loss

        return {
            "loss":      total_loss,
            "diff_loss": diff_loss.detach(),
            "seg_loss":  seg_loss.detach(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Strategy B: Latent-space supervision (recommended starting point)
# ──────────────────────────────────────────────────────────────────────────────

class StrategyBTrainer:
    """
    Train segmentation model on a mix of real + ADC-generated images.
    No gradient through ADC — stable and practical.

    Workflow:
      1. Run tutorial_inference_local.py to generate synthetic liver images.
      2. Instantiate this trainer with your seg_model.
      3. Call train() to fine-tune the segmentation model.
      4. Periodically regenerate with updated ADC (outer loop).
    """

    def __init__(
        self,
        seg_model: nn.Module,
        real_data_root: str,          # path to real laparoscopic images
        gen_data_root: str,           # path to ADC-generated images
        mask_data_root: str,          # path to masks (GT for seg loss)
        mix_ratio: float = 0.5,       # fraction of generated images in each batch
        lr: float = 1e-4,
        device: str = "auto",
    ):
        if device == "auto":
            if torch.cuda.is_available():      self.device = torch.device("cuda")
            elif torch.backends.mps.is_available(): self.device = torch.device("mps")
            else:                              self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.seg      = seg_model.to(self.device)
        self.real_dir = Path(real_data_root)
        self.gen_dir  = Path(gen_data_root)
        self.mask_dir = Path(mask_data_root)
        self.mix_ratio  = mix_ratio
        self.optimizer  = torch.optim.AdamW(self.seg.parameters(), lr=lr)

    def load_image_pair(self, img_path: Path, mask_path: Path):
        """Load image + mask, return tensors on device."""
        from torchvision import transforms
        from PIL import Image

        tf_img  = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])
        tf_mask = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])

        img  = tf_img(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
        mask = tf_mask(Image.open(mask_path).convert("L")).unsqueeze(0).to(self.device)
        return img, mask

    def train(self, epochs: int = 50):
        """
        Simple training loop — PLACEHOLDER.
        Replace with your actual DataLoader + training logic once data is ready.
        """
        print(f"\n[StrategyBTrainer] Training segmentation model for {epochs} epochs.")
        print(f"  Real images:  {self.real_dir}")
        print(f"  Generated:    {self.gen_dir}")
        print(f"  Masks:        {self.mask_dir}")
        print(f"  Device:       {self.device}")
        print(f"  Mix ratio:    {self.mix_ratio} (fraction of generated images per batch)")
        print("\n  → Build your DataLoader here and call self.seg(images) + compute loss.")
        print("  → See SingleStepADCSegmentation for the loss computation pattern.")

        # TODO: Build DataLoader that mixes real + generated images at mix_ratio
        # TODO: Training loop over epochs
        # TODO: Validation on held-out real data
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Example: Minimal U-Net segmentation model stub
# Replace this with your actual segmentation model
# ──────────────────────────────────────────────────────────────────────────────

class MinimalSegModel(nn.Module):
    """
    Placeholder tiny encoder-decoder for testing the integration.
    Replace with your actual segmentation model (anything that takes [B,3,H,W] → [B,1,H,W]).
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(), # 1/4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),   # 1/2
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),    # original
            nn.Conv2d(32, 1, 1),  # binary logits
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────────────────────────────────────
# Quick integration test (no data required)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Integration architecture components:")
    print("  - SingleStepADCSegmentation: end-to-end differentiable (Strategy A)")
    print("  - StrategyBTrainer: stable 2-stage training (Strategy B — recommended)")
    print("  - MinimalSegModel: placeholder segmentation model stub")
    print("\nTo use:")
    print("  1. Replace MinimalSegModel with your actual segmentation model")
    print("  2. Choose Strategy A or B based on your training setup")
    print("  3. For Strategy A: wrap ADC with SingleStepADCSegmentation")
    print("     For Strategy B: run inference first, then use StrategyBTrainer")
    print("\nSee docstrings for full usage instructions.")

    # Test MinimalSegModel
    model = MinimalSegModel()
    x = torch.randn(1, 3, 384, 384)
    out = model(x)
    print(f"\nMinimalSegModel test: input {x.shape} → output {out.shape}")
    assert out.shape == (1, 1, 384, 384), f"Unexpected output shape: {out.shape}"
    print("Shape check: PASSED")
