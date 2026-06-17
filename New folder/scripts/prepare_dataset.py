"""
prepare_dataset.py
------------------
Prepares the chicken detection dataset:
  - Splits images+labels into train / val / test
  - Generates datasets/data.yaml

Usage:
  python scripts/prepare_dataset.py \
      --images_dir /path/to/raw_images \
      --labels_dir /path/to/raw_labels \
      --split 0.7 0.2 0.1

If you already placed files manually into datasets/images/{train,val,test}
and datasets/labels/{train,val,test}, just run:
  python scripts/prepare_dataset.py --generate_yaml_only
"""

import os
import sys
import shutil
import random
import argparse
import yaml
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Adjust class names to match your dataset ──────────────────────────────────
# Single class: detect ALL chicken pieces as one category
CLASS_NAMES = [
    "chicken_piece",
]
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "datasets"
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def generate_yaml():
    """Write datasets/data.yaml based on CLASS_NAMES."""
    yaml_path = DATASET_DIR / "data.yaml"
    config = {
        "path":  str(DATASET_DIR),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[✓] data.yaml written → {yaml_path}")
    return yaml_path


def split_dataset(images_dir: Path, labels_dir: Path, split: tuple):
    """Copy images + labels from raw dirs into the split folder structure."""
    train_r, val_r, test_r = split
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    random.seed(42)
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    counts = {}
    for split_name, split_images in splits.items():
        dst_img = DATASET_DIR / "images" / split_name
        dst_lbl = DATASET_DIR / "labels" / split_name
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing_labels = []
        for img_path in split_images:
            # Copy image
            shutil.copy2(img_path, dst_img / img_path.name)

            # Copy matching label
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl / lbl_path.name)
                copied += 1
            else:
                missing_labels.append(img_path.stem)

        counts[split_name] = len(split_images)
        if missing_labels:
            print(f"  ⚠  {len(missing_labels)} label file(s) missing for {split_name}: "
                  f"{missing_labels[:5]}{'...' if len(missing_labels) > 5 else ''}")

    print(f"\n[✓] Dataset split complete:")
    for s, c in counts.items():
        print(f"    {s:6s} → {c:5d} images")
    print(f"    Total  → {n} images\n")


def verify_dataset():
    """Quick sanity check on the split dataset."""
    print("[i] Verifying dataset …")
    issues = []
    for split in ("train", "val", "test"):
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS] if img_dir.exists() else []
        labels = [p for p in lbl_dir.iterdir() if p.suffix == ".txt"]             if lbl_dir.exists() else []

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        unmatched = img_stems - lbl_stems
        if unmatched:
            issues.append(f"  {split}: {len(unmatched)} image(s) have no label → {list(unmatched)[:3]}")

        print(f"    {split:6s} → {len(images):5d} images | {len(labels):5d} labels")

    if issues:
        print("\n⚠  Issues found:")
        for i in issues:
            print(i)
    else:
        print("\n[✓] All images have matching labels.\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare chicken detection dataset for YOLO")
    parser.add_argument("--images_dir",       type=str,   default=None,
                        help="Directory containing all raw images")
    parser.add_argument("--labels_dir",       type=str,   default=None,
                        help="Directory containing all YOLO .txt label files")
    parser.add_argument("--split",            type=float, nargs=3, default=[0.7, 0.2, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train / val / test split ratios (must sum to 1.0)")
    parser.add_argument("--generate_yaml_only", action="store_true",
                        help="Only (re)generate data.yaml, skip copying files")
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.generate_yaml_only:
        if not args.images_dir or not args.labels_dir:
            parser.error("--images_dir and --labels_dir are required unless --generate_yaml_only is set.")

        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir)

        if not images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

        split_dataset(images_dir, labels_dir, tuple(args.split))

    generate_yaml()
    verify_dataset()
    print("[✓] Dataset preparation complete. Run train.py next.")


if __name__ == "__main__":
    main()
