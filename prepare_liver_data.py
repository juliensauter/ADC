"""
prepare_liver_data.py
=====================
Converts raw liver surgery data (images + masks) into the ADC training format.

Expected raw data structure:
    <raw_data_root>/
        images/          ← RGB laparoscopic frames (.png, .jpg, .bmp, etc.)
        masks/           ← Binary segmentation masks (matching filenames, any format)
                            White = liver region, Black = background

Output structure (under ADC data/):
    data/
        images/          ← resized RGB PNG frames
        masks/           ← binarized, resized single-channel PNG masks
        prompt.json      ← JSONL used by MyDataset

Usage:
    uv run python prepare_liver_data.py \
        --src /path/to/raw/liver_data \
        --out ./data \
        --prompt "a laparoscopic image of the liver" \
        --size 384 \
        --val-split 0.1

NOTES:
  - Pairs are matched by filename (stem must be identical).
  - Images/masks that have no counterpart are skipped with a warning.
  - Use --dry-run to preview what would be processed without writing files.
"""

import os
import sys
import json
import shutil
import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np

# Always run from ADC root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

IMG_EXTS  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def binarize_mask(mask_path: Path, threshold: int = 127) -> Image.Image:
    """Load mask and convert to binary (0/255) single-channel image."""
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)
    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def resize_image(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


def collect_pairs(src: Path):
    """Find all (image, mask) pairs by matching stems."""
    img_dir  = src / "images"
    mask_dir = src / "masks"

    assert img_dir.exists(),  f"images/ directory not found under {src}"
    assert mask_dir.exists(), f"masks/ directory not found under {src}"

    img_stems  = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    mask_stems = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in MASK_EXTS}

    paired, orphan_imgs, orphan_masks = [], [], []
    for stem, img_p in sorted(img_stems.items()):
        if stem in mask_stems:
            paired.append((img_p, mask_stems[stem]))
        else:
            orphan_imgs.append(stem)
    for stem in mask_stems:
        if stem not in img_stems:
            orphan_masks.append(stem)

    return paired, orphan_imgs, orphan_masks


def process_pair(img_path, mask_path, out_img_dir, out_mask_dir, size, idx):
    """Resize image + binarize mask, save to output dirs. Returns (img_out, mask_out)."""
    stem = f"{idx:06d}"
    img_out  = out_img_dir  / f"{stem}.png"
    mask_out = out_mask_dir / f"{stem}.png"

    img  = Image.open(img_path).convert("RGB")
    mask = binarize_mask(mask_path)

    img  = resize_image(img,  size)
    mask = resize_image(mask, size)

    # Convert binary mask to RGB (ADC dataset loader reads 'L' then converts to 'RGB' itself)
    img.save(img_out)
    mask.save(mask_out)

    return img_out, mask_out


def main():
    parser = argparse.ArgumentParser(description="Prepare liver surgery data for ADC training.")
    parser.add_argument("--src",        required=True, help="Root of raw data (must contain images/ and masks/)")
    parser.add_argument("--out",        default="./data", help="Output directory [default: ./data]")
    parser.add_argument("--prompt",     default="a laparoscopic image of the liver",
                        help="Text conditioning prompt [default: 'a laparoscopic image of the liver']")
    parser.add_argument("--size",       type=int, default=384, help="Output image size [default: 384]")
    parser.add_argument("--val-split",  type=float, default=0.1, help="Fraction for validation set [default: 0.1]")
    parser.add_argument("--dry-run",    action="store_true", help="Preview without writing files")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    assert src.exists(), f"Source directory not found: {src}"

    # Collect pairs
    pairs, orphan_imgs, orphan_masks = collect_pairs(src)
    print(f"\nFound {len(pairs)} matched pairs.")
    if orphan_imgs:
        print(f"  ⚠ {len(orphan_imgs)} images without masks (skipped): {orphan_imgs[:5]}{'…' if len(orphan_imgs)>5 else ''}")
    if orphan_masks:
        print(f"  ⚠ {len(orphan_masks)} masks without images (skipped): {orphan_masks[:5]}{'…' if len(orphan_masks)>5 else ''}")
    if not pairs:
        print("No pairs found — check that filenames match between images/ and masks/.")
        sys.exit(1)

    # Train / val split
    random.seed(args.seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_split)) if args.val_split > 0 else 0
    val_pairs   = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    print(f"  Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Create output dirs
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)

    # Process pairs and build JSONL
    def write_split(pairs_list, split):
        entries = []
        for idx, (img_p, mask_p) in enumerate(pairs_list):
            img_out, mask_out = process_pair(
                img_p, mask_p,
                out / split / "images",
                out / split / "masks",
                args.size, idx
            )
            entries.append({
                "source": str(img_out.relative_to(out.parent)),  # relative to ADC root
                "target": str(mask_out.relative_to(out.parent)),
                "prompt_target": args.prompt,
            })
            if (idx + 1) % 50 == 0:
                print(f"  [{split}] Processed {idx+1}/{len(pairs_list)}")

        jsonl_path = out / split / "prompt.json"
        with open(jsonl_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"  [{split}] Wrote {len(entries)} entries → {jsonl_path}")

    print("\nProcessing training set...")
    write_split(train_pairs, "train")
    print("Processing validation set...")
    write_split(val_pairs, "val")

    # Also write a combined prompt.json at the top level (for when val_split=0 or testing)
    combined_path = out / "prompt.json"
    with open(combined_path, "w") as f:
        for idx, (img_p, mask_p) in enumerate(train_pairs):
            f.write(json.dumps({
                "source": f"data/train/masks/{idx:06d}.png",
                "target": f"data/train/images/{idx:06d}.png",
                "prompt_target": args.prompt,
            }) + "\n")
        for idx, (img_p, mask_p) in enumerate(val_pairs):
            f.write(json.dumps({
                "source": f"data/val/masks/{idx:06d}.png",
                "target": f"data/val/images/{idx:06d}.png",
                "prompt_target": args.prompt,
            }) + "\n")

    print(f"\nDone! Data written to: {out.resolve()}")
    print(f"For training, set MyDataset root to 'data/train/prompt.json'")


if __name__ == "__main__":
    main()
