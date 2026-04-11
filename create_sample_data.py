"""Create synthetic liver mask + placeholder image for quick inference demo.

NOTE: This writes to data/demo_prompt.json (not data/prompt.json) to avoid
overwriting real training data produced by prepare_liver_data.py.
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("data/masks", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

H, W = 384, 384

# Irregular organ-like white blob on black background (liver-shaped)
mask = Image.new("L", (W, H), 0)
draw = ImageDraw.Draw(mask)
draw.ellipse([(80, 100), (310, 260)], fill=255)    # Main lobe
draw.ellipse([(50, 120), (180, 220)], fill=255)    # Left extension
draw.polygon([(130, 220), (195, 220), (162, 265)], fill=0)  # Hilum notch
mask.save("data/masks/sample_001.png")
print("Saved: data/masks/sample_001.png")

# Dark reddish-brown placeholder (liver tissue background)
img_arr = np.zeros((H, W, 3), dtype=np.uint8)
img_arr[:, :, 0] = 60   # red
img_arr[:, :, 1] = 25   # green
img_arr[:, :, 2] = 20   # blue
Image.fromarray(img_arr).save("data/images/sample_001.png")
print("Saved: data/images/sample_001.png")

# JSONL prompt file — writes to demo_prompt.json to avoid overwriting training data
entry = {
    "source": "data/masks/sample_001.png",
    "target": "data/images/sample_001.png",
    "prompt_target": "a laparoscopic image of the liver",
}
with open("data/demo_prompt.json", "w") as f:
    f.write(json.dumps(entry) + "\n")
print("Saved: data/demo_prompt.json")
print("Sample data ready. Use data/demo_prompt.json for inference demos.")
