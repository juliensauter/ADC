"""
create_control_ckpt.py
=======================
Creates control_sd15.ckpt from SD v1.5 weights.
This is a self-contained wrapper around tool_add_control.py logic.

Equivalent to:
    python tool_add_control.py \
        stable-diffusion-v1-5/v1-5-pruned.ckpt \
        stable-diffusion-v1-5/control_sd15.ckpt

Run once before training to produce the ADC base checkpoint.

Memory-optimised: peak ~12 GB RAM instead of ~25 GB.
"""
import os
import sys
import gc

# Always run from the ADC project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

INPUT_PATH  = "./stable-diffusion-v1-5/v1-5-pruned.ckpt"
OUTPUT_PATH = "./stable-diffusion-v1-5/control_sd15.ckpt"

assert os.path.exists(INPUT_PATH), f"SD v1.5 checkpoint not found: {INPUT_PATH}"
if os.path.exists(OUTPUT_PATH):
    print(f"control_sd15.ckpt already exists at {OUTPUT_PATH} — skipping.")
    sys.exit(0)

import torch
from share import *
from cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ""
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ""
    return True, name[len(parent_name):]


print(f"Loading SD v1.5 weights from {INPUT_PATH} ...")

# Step 1: Load pretrained weights
# weights_only=False needed: SD v1.5 .ckpt contains PL callback state (PyTorch ≥2.6 changed default)
pretrained_weights = torch.load(INPUT_PATH, map_location="cpu", weights_only=False)
if "state_dict" in pretrained_weights:
    pretrained_weights = pretrained_weights["state_dict"]

# Step 2: Create model and get its state dict keys (to know the target structure)
print("Creating ADC model structure...")
model = create_model(config_path="./models/cldm_v15.yaml")
scratch_keys = list(model.state_dict().keys())

# Step 3: Build target dict directly — no intermediate scratch_dict copy
print("Mapping weights...")
target_dict = {}
for k in scratch_keys:
    is_control, name = get_node_name(k, "control_")
    is_image_control, image_name = get_node_name(k, "image_control_")
    if is_control:
        copy_k = "model.diffusion_" + name
    elif is_image_control:
        # image_control_model.* → model.diffusion_model.* (same encoder architecture)
        copy_k = "model.diffusion_" + image_name
    else:
        copy_k = k

    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        # Keep the randomly initialised value from model for this key
        target_dict[k] = model.state_dict()[k].clone()

    # image_ decoder layers ← copy from corresponding SD output layers
    if k.startswith("model.diffusion_model.image_"):
        output_layer = k.replace("image_", "", 1)
        if output_layer in pretrained_weights:
            target_dict[k] = pretrained_weights[output_layer].clone()

# Step 4: Free pretrained weights before loading into model
del pretrained_weights
gc.collect()

# Step 5: Load and save
print("Loading mapped weights into model...")
model.load_state_dict(target_dict, strict=True)
del target_dict
gc.collect()

print(f"Saving to {OUTPUT_PATH} ...")
torch.save(model.state_dict(), OUTPUT_PATH)
del model
gc.collect()

print(f"Saved: {OUTPUT_PATH}")
print("Done — control_sd15.ckpt ready for training.")
