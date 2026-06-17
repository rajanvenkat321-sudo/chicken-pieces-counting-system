"""
train.py
--------
Trains a YOLOv8 model to detect chicken pieces (single class).
Optimized for SMALL datasets and maximum accuracy.

Usage:
  python scripts/train.py                          # recommended defaults
  python scripts/train.py --model yolov8l.pt       # larger model
  python scripts/train.py --epochs 500             # even more epochs
  python scripts/train.py --resume                 # resume last run
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics not found. Install with:\n"
        "  pip install ultralytics"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML    = PROJECT_ROOT / "datasets" / "data.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"


def check_data_yaml():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {DATA_YAML}.\n"
            "Run: python scripts/prepare_dataset.py --generate_yaml_only"
        )
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    print(f"[OK] Dataset config loaded:")
    print(f"     Classes ({cfg['nc']}): {cfg['names']}")
    # Count training images
    train_dir = PROJECT_ROOT / "datasets" / "images" / "train"
    val_dir   = PROJECT_ROOT / "datasets" / "images" / "val"
    n_train   = len(list(train_dir.glob("*.*"))) if train_dir.exists() else 0
    n_val     = len(list(val_dir.glob("*.*")))   if val_dir.exists()   else 0
    print(f"     Train images : {n_train}")
    print(f"     Val images   : {n_val}")
    if n_train < 10:
        print(f"[!] WARNING: Very few training images ({n_train}). "
              f"Consider collecting more data for better accuracy.")
    return cfg


def train(args):
    cfg = check_data_yaml()
    n_train = len(list((PROJECT_ROOT / "datasets" / "images" / "train").glob("*.*")))

    # --- Model selection ---
    if args.resume:
        last_run = sorted(RUNS_DIR.glob("**/last.pt"))
        if not last_run:
            raise FileNotFoundError("No previous run found to resume.")
        model_path = str(last_run[-1])
        print(f"[->] Resuming from: {model_path}")
    else:
        model_path = args.model
        print(f"[->] Starting fresh training with: {model_path}")

    model = YOLO(model_path)

    # --- Auto-tune batch size for small datasets ---
    # With <20 images, large batches give same batch every epoch → overfit
    # With 100 images, batch=16 is safe and faster
    batch = min(args.batch, max(4, n_train // 6))
    print(f"[->] Using batch size: {batch} (dataset has {n_train} train images)")

    # --- Training ---
    results = model.train(
        # Dataset
        data        = str(DATA_YAML),
        project     = str(RUNS_DIR),
        name        = "chicken_detect",

        # Core
        epochs      = args.epochs,
        batch       = batch,
        imgsz       = args.img,
        device      = args.device,

        # Single-class optimizations
        single_cls  = True,        # treat all classes as one

        # Optimizer  (AdamW works well for medium datasets)
        optimizer        = "AdamW",
        lr0              = 0.001,       # standard LR for 100-image dataset
        lrf              = 0.01,
        momentum         = 0.937,
        weight_decay     = 0.0005,
        warmup_epochs    = 5,           # longer warmup for stability
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,
        cos_lr           = True,        # cosine LR annealing -> better convergence

        # Augmentation (heavy aug compensates for tiny dataset)
        augment     = True,
        hsv_h       = 0.02,    # hue jitter
        hsv_s       = 0.75,    # saturation jitter
        hsv_v       = 0.5,     # brightness jitter
        degrees     = 15.0,    # rotation
        translate   = 0.1,     # shift
        scale       = 0.6,     # zoom in/out
        shear       = 5.0,     # shear transform
        perspective = 0.0005,  # slight perspective warp
        fliplr      = 0.5,     # horizontal flip
        flipud      = 0.1,     # vertical flip (light — food can be upside-down)
        mosaic      = 1.0,     # mosaic (combines 4 images) — critical for small data
        mixup       = 0.15,    # blend images
        copy_paste  = 0.1,     # copy-paste augmentation
        erasing     = 0.3,     # random erasing (forces robust features)
        close_mosaic= 20,      # disable mosaic in last 20 epochs for fine-tuning

        # Training quality
        patience    = 50,      # 50 epochs patience with 100 images
        cache       = "disk",  # disk cache safer for 100 images
        workers     = 0,       # 0 = main thread only (required on Windows)
        save        = True,
        save_period = 20,      # save checkpoint every 20 epochs
        exist_ok    = True,

        # Validation
        val         = True,
        plots       = True,

        # Resume
        resume      = args.resume,
    )

    # --- Summary ---
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    weights_dir = RUNS_DIR / "chicken_detect" / "weights"
    print(f"  Best weights : {weights_dir / 'best.pt'}")
    print(f"  Last weights : {weights_dir / 'last.pt'}")
    print(f"  Results dir  : {RUNS_DIR / 'chicken_detect'}")
    print("="*60)

    # Auto-validate best model
    print("\n[->] Validating best model ...")
    best_model  = YOLO(str(weights_dir / "best.pt"))
    val_results = best_model.val(data=str(DATA_YAML), workers=0)
    print(f"\n  mAP@0.5      : {val_results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {val_results.box.map:.4f}")
    print(f"  Precision    : {val_results.box.mp:.4f}")
    print(f"  Recall       : {val_results.box.mr:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 chicken piece detector (single class)")

    parser.add_argument("--model",   type=str, default="yolov8l.pt",
                        choices=["yolov8n.pt","yolov8s.pt","yolov8m.pt",
                                 "yolov8l.pt","yolov8x.pt"],
                        help="YOLOv8 model variant. yolov8l=large (recommended for 100+ images), "
                             "yolov8x=xlarge (max accuracy, slowest)")
    parser.add_argument("--epochs",  type=int, default=500,
                        help="Training epochs (default 500 for max accuracy)")
    parser.add_argument("--batch",   type=int, default=16,
                        help="Max batch size (auto-reduced for tiny datasets)")
    parser.add_argument("--img",     type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device",  type=str, default="",
                        help="Device: '' (auto), 'cpu', '0' (GPU)")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
