"""Create synthetic polyp mask + placeholder image for inference demo."""
import os
import json
import numpy as np
from PIL import Image, ImageDraw

os.makedirs("data/masks", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

H, W = 384, 384

# Irregular polyp-like white blob on black background
mask = Image.new("L", (W, H), 0)
draw = ImageDraw.Draw(mask)
draw.ellipse([(130, 150), (270, 250)], fill=255)
draw.ellipse([(200, 130), (280, 200)], fill=255)
draw.ellipse([(120, 170), (180, 240)], fill=255)
mask.save("data/masks/sample_001.png")
print("Saved: data/masks/sample_001.png")

# Dark greenish placeholder (colonoscopy tissue background)
img_arr = np.zeros((H, W, 3), dtype=np.uint8)
img_arr[:, :, 1] = 40
img_arr[:, :, 2] = 20
Image.fromarray(img_arr).save("data/images/sample_001.png")
print("Saved: data/images/sample_001.png")

# JSONL prompt file
entry = {
    "source": "data/masks/sample_001.png",
    "target": "data/images/sample_001.png",
    "prompt_target": "a colonoscopy image showing a polyp",
}
with open("data/prompt.json", "w") as f:
    f.write(json.dumps(entry) + "\n")
print("Saved: data/prompt.json")
print("Sample data ready.")
