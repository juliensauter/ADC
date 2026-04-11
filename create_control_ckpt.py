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
"""
import os
import sys

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
model = create_model(config_path="./models/cldm_v15.yaml")

# weights_only=False needed: SD v1.5 .ckpt contains PL callback state (PyTorch ≥2.6 changed default)
pretrained_weights = torch.load(INPUT_PATH, map_location="cpu", weights_only=False)
if "state_dict" in pretrained_weights:
    pretrained_weights = pretrained_weights["state_dict"]

scratch_dict = model.state_dict()
target_dict  = {}

for k in scratch_dict.keys():
    is_control, name = get_node_name(k, "control_")
    copy_k = "model.diffusion_" + name if is_control else k

    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()

    # image_ decoder layers ← copy from corresponding SD output layers
    if k.startswith("model.diffusion_model.image_"):
        output_layer = k.replace("image_", "", 1)
        if output_layer in pretrained_weights:
            target_dict[k] = pretrained_weights[output_layer].clone()

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")
print("Done — control_sd15.ckpt ready for training.")
