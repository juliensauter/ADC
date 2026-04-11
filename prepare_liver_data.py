"""
prepare_liver_data.py
=====================
Converts raw liver surgery data (images + masks) into the ADC training format.

Expected raw data structure (flat):
    <raw_data_root>/
        images/          ← RGB laparoscopic frames (.png, .jpg, .bmp, etc.)
        masks/           ← Binary segmentation masks (matching filenames, any format)
                            White = liver region, Black = background

Expected raw data structure (DSAD pre-split with --dsad flag):
    <raw_data_root>/
        images/{train,val,test}/  ← e.g. 23_image00.png
        masks/{train,val,test}/   ← e.g. 23_mask00_liver.png

Output structure (under ADC data/):
    data/
        train/images/    ← resized RGB PNG frames
        train/masks/     ← binarized, resized single-channel PNG masks
        train/prompt.json
        val/images/
        val/masks/
        val/prompt.json
        prompt.json      ← combined JSONL used by MyDataset

Usage:
    # Flat data (stems match between images/ and masks/):
    uv run python prepare_liver_data.py \
        --src /path/to/raw/liver_data \
        --out ./data \
        --prompt "a laparoscopic image of the liver" \
        --size 384 \
        --val-split 0.1

    # DSAD liver data (pre-split, different naming convention):
    uv run python prepare_liver_data.py \
        --src /path/to/anatomy_aware_diffusion/dataUtils/dataset \
        --out ./data \
        --dsad \
        --prompt "a laparoscopic image of the liver"

NOTES:
  - Default mode: pairs matched by filename stem (must be identical).
  - DSAD mode (--dsad): pairs matched by video_id + frame_id across
    naming convention (23_image00.png ↔ 23_mask00_liver.png).
  - Images/masks that have no counterpart are skipped with a warning.
  - Use --dry-run to preview what would be processed without writing files.
"""

import os
import sys
import json
import re
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


def _dsad_key(filename: str) -> str | None:
    """Extract a pairing key from DSAD naming conventions.

    Handles two naming patterns in the DSAD liver dataset:
      - DSAD originals:  '{vid}_image{frame}.png' / '{vid}_mask{frame}_liver.png'
                         → key: '{vid}_{frame}'
      - Augmented liver: 'liver_{vid}_{frame}[_aug_...].png' (identical stems)
                         → key: stem (no special extraction needed)

    Falls back to filename stem for any other pattern.
    """
    stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
    # Pattern 1: DSAD original images — {vid}_image{frame}
    m = re.match(r'^(\d+)_image(\d+)', stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # Pattern 2: DSAD original masks — {vid}_mask{frame}_liver
    m = re.match(r'^(\d+)_mask(\d+)_liver', stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # Pattern 3: liver_ prefix or any other — use full stem (images and masks share the same name)
    return stem


def collect_pairs(src: Path, dsad: bool = False):
    """Find all (image, mask) pairs.

    Default: match by identical stem.
    DSAD mode: match by extracted video_id + frame_id.
    """
    img_dir  = src / "images"
    mask_dir = src / "masks"

    assert img_dir.exists(),  f"images/ directory not found under {src}"
    assert mask_dir.exists(), f"masks/ directory not found under {src}"

    img_files  = {p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    mask_files = {p for p in mask_dir.iterdir() if p.suffix.lower() in MASK_EXTS}

    if dsad:
        img_keys  = {_dsad_key(p.name): p for p in img_files}
        mask_keys = {_dsad_key(p.name): p for p in mask_files}
    else:
        img_keys  = {p.stem: p for p in img_files}
        mask_keys = {p.stem: p for p in mask_files}

    paired, orphan_imgs, orphan_masks = [], [], []
    for key, img_p in sorted(img_keys.items()):
        if key in mask_keys:
            paired.append((img_p, mask_keys[key]))
        else:
            orphan_imgs.append(key)
    for key in mask_keys:
        if key not in img_keys:
            orphan_masks.append(key)

    return paired, orphan_imgs, orphan_masks


def collect_pairs_presplit(src: Path, dsad: bool = False):
    """Collect pairs from a pre-split directory (train/val/test subdirs under images/ and masks/)."""
    all_splits = {}
    img_root  = src / "images"
    mask_root = src / "masks"

    for split_dir in sorted(img_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        mask_split = mask_root / split_name
        if not mask_split.exists():
            print(f"  ⚠ Skipping split '{split_name}': no matching masks/{split_name}/ directory")
            continue
        pairs, o_img, o_mask = [], [], []

        img_files = {p for p in split_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        mask_files = {p for p in mask_split.iterdir() if p.suffix.lower() in MASK_EXTS}

        if dsad:
            img_keys  = {_dsad_key(p.name): p for p in img_files}
            mask_keys = {_dsad_key(p.name): p for p in mask_files}
        else:
            img_keys  = {p.stem: p for p in img_files}
            mask_keys = {p.stem: p for p in mask_files}

        for key, img_p in sorted(img_keys.items()):
            if key in mask_keys:
                pairs.append((img_p, mask_keys[key]))
            else:
                o_img.append(key)
        for key in mask_keys:
            if key not in img_keys:
                o_mask.append(key)

        all_splits[split_name] = (pairs, o_img, o_mask)

    return all_splits


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


def write_split(pairs_list, split, out, args):
    """Process pairs and write JSONL for one split.

    IMPORTANT: In ADC's MyDataset:
      - 'source' = mask (loaded as grayscale, binarized → hint/condition)
      - 'target' = image (loaded as RGB → jpg/ground truth)
    """
    entries = []
    for idx, (img_p, mask_p) in enumerate(pairs_list):
        img_out, mask_out = process_pair(
            img_p, mask_p,
            out / split / "images",
            out / split / "masks",
            args.size, idx
        )
        entries.append({
            "source": str(mask_out.relative_to(out.parent)),   # mask → condition input
            "target": str(img_out.relative_to(out.parent)),    # image → generation target
            "prompt_target": args.prompt,
        })
        if (idx + 1) % 100 == 0:
            print(f"  [{split}] Processed {idx+1}/{len(pairs_list)}")

    jsonl_path = out / split / "prompt.json"
    with open(jsonl_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"  [{split}] Wrote {len(entries)} entries → {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare liver surgery data for ADC training.")
    parser.add_argument("--src",        required=True, help="Root of raw data (must contain images/ and masks/)")
    parser.add_argument("--out",        default="./data", help="Output directory [default: ./data]")
    parser.add_argument("--prompt",     default="a laparoscopic image of the liver",
                        help="Text conditioning prompt [default: 'a laparoscopic image of the liver']")
    parser.add_argument("--size",       type=int, default=384, help="Output image size [default: 384]")
    parser.add_argument("--val-split",  type=float, default=0.1, help="Fraction for validation set [default: 0.1]")
    parser.add_argument("--dsad",       action="store_true",
                        help="DSAD naming mode: pairs 23_image00.png ↔ 23_mask00_liver.png")
    parser.add_argument("--pre-split",  action="store_true",
                        help="Data is already split into train/val(/test) subdirs")
    parser.add_argument("--dry-run",    action="store_true", help="Preview without writing files")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    # Auto-detect pre-split if images/train/ exists
    if not args.pre_split and (Path(args.src) / "images" / "train").is_dir():
        print("Auto-detected pre-split data structure (images/train/ found). Using --pre-split mode.")
        args.pre_split = True

    src = Path(args.src)
    out = Path(args.out)
    assert src.exists(), f"Source directory not found: {src}"

    if args.pre_split:
        # ── Pre-split mode (e.g. DSAD with train/val/test) ─────────────────
        split_data = collect_pairs_presplit(src, dsad=args.dsad)
        if not split_data:
            print("No valid splits found. Check directory structure.")
            sys.exit(1)

        total = sum(len(pairs) for pairs, _, _ in split_data.values())
        print(f"\nFound {total} total pairs across {len(split_data)} splits:")
        for split_name, (pairs, o_img, o_mask) in split_data.items():
            print(f"  {split_name}: {len(pairs)} pairs")
            if o_img:
                print(f"    ⚠ {len(o_img)} images without masks: {o_img[:3]}{'…' if len(o_img)>3 else ''}")
            if o_mask:
                print(f"    ⚠ {len(o_mask)} masks without images: {o_mask[:3]}{'…' if len(o_mask)>3 else ''}")
            if not pairs:
                hint = ' Try adding --dsad if this is DSAD data.' if not args.dsad else ''
                print(f"    ✗ No pairs in {split_name}.{hint}")

        if args.dry_run:
            print("\n[DRY RUN] No files written.")
            return

        for split_name, (pairs, _, _) in split_data.items():
            if not pairs:
                continue
            (out / split_name / "images").mkdir(parents=True, exist_ok=True)
            (out / split_name / "masks").mkdir(parents=True, exist_ok=True)
            write_split(pairs, split_name, out, args)

    else:
        # ── Flat mode (images/ + masks/ at root, we do our own split) ─────
        pairs, orphan_imgs, orphan_masks = collect_pairs(src, dsad=args.dsad)
        print(f"\nFound {len(pairs)} matched pairs.")
        if orphan_imgs:
            hint = ' Try adding --dsad if this is DSAD data.' if not args.dsad else ''
            print(f"  ⚠ {len(orphan_imgs)} images without masks (skipped): {orphan_imgs[:5]}{'…' if len(orphan_imgs)>5 else ''}{hint}")
        if orphan_masks:
            print(f"  ⚠ {len(orphan_masks)} masks without images (skipped): {orphan_masks[:5]}{'…' if len(orphan_masks)>5 else ''}")
        if not pairs:
            hint = ' Try adding --dsad if this is DSAD data.' if not args.dsad else ''
            print(f"No pairs found — check that filenames match between images/ and masks/.{hint}")
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

        for split in ("train", "val"):
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "masks").mkdir(parents=True, exist_ok=True)

        print("\nProcessing training set...")
        write_split(train_pairs, "train", out, args)
        print("Processing validation set...")
        write_split(val_pairs, "val", out, args)

    # ── Combined prompt.json at top level ─────────────────────────────────
    combined_path = out / "prompt.json"
    with open(combined_path, "w") as f:
        for split_name in sorted((out).iterdir()):
            if not split_name.is_dir():
                continue
            split_prompt = split_name / "prompt.json"
            if split_prompt.exists():
                with open(split_prompt) as sf:
                    f.write(sf.read())
    print(f"\nCombined prompt.json: {combined_path}")
    print(f"Done! Data written to: {out.resolve()}")
    print(f"For training, set MyDataset root to 'data/train/prompt.json'")


if __name__ == "__main__":
    main()
