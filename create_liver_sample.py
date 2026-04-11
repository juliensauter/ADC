"""Create a synthetic liver-shaped mask for ADC inference demo."""
import os, json, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("data/masks", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

SIZE = 384

# Liver-shaped mask: large right lobe + smaller left lobe + hilum notch
mask = Image.new("L", (SIZE, SIZE), 0)
draw = ImageDraw.Draw(mask)

# Right (main) lobe
draw.ellipse([50, 60, 310, 250], fill=255)
# Left lobe extension
draw.ellipse([30, 80, 180, 200], fill=255)
# Hepatic hilum notch (inferior border indent)
draw.polygon([(130, 210), (195, 210), (162, 255)], fill=0)

# Smooth edges
arr = np.array(mask.filter(ImageFilter.GaussianBlur(radius=7)))
mask_final = Image.fromarray((arr > 127).astype(np.uint8) * 255, mode="L")
mask_final.save("data/masks/liver_001.png")

# Black placeholder target (only mask is used at inference time)
Image.new("RGB", (SIZE, SIZE), (0, 0, 0)).save("data/images/liver_001.png")

entry = {
    "source": "data/masks/liver_001.png",
    "target": "data/images/liver_001.png",
    "prompt_target": "laparoscopic view of liver surface, hepatic tissue texture, surgical lighting, intraoperative photograph"
}
with open("data/prompt.json", "w") as f:
    json.dump(entry, f)

print("Liver mask + prompt.json written OK")
print(f"  Mask:   data/masks/liver_001.png")
print(f"  Prompt: {entry['prompt_target']}")
subprocess.run(["open", "data/masks/liver_001.png"])
