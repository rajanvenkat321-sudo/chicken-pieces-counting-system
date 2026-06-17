"""
auto_label.py
-------------
Uses YOLO-World (zero-shot open-vocab detector) to automatically generate
YOLO-format bounding box labels for all images in the dataset.

Detected class: 'chicken' -> saved as class index 0 (chicken_piece).

Usage:
  python scripts/auto_label.py
  python scripts/auto_label.py --conf 0.01
"""

import sys
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
IMAGES_DIR   = DATASETS_DIR / "images"
LABELS_DIR   = DATASETS_DIR / "labels"

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def auto_label(conf: float):
    print(f"[->] Loading YOLO-World model ...")
    model = YOLO("yolov8s-world.pt")
    model.set_classes(["chicken"])

    total_imgs   = 0
    total_boxes  = 0
    empty_imgs   = 0

    for split in SPLITS:
        img_dir = IMAGES_DIR / split
        lbl_dir = LABELS_DIR / split
        if not img_dir.exists():
            print(f"[SKIP] {img_dir} not found")
            continue

        imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        print(f"\n[->] Processing split '{split}': {len(imgs)} images ...")

        for img_path in sorted(imgs):
            results = model.predict(img_path, conf=conf, verbose=False)
            r = results[0]
            boxes = r.boxes

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            lbl_path.parent.mkdir(parents=True, exist_ok=True)

            lines = []
            if boxes is not None and len(boxes) > 0:
                iw, ih = r.orig_shape[1], r.orig_shape[0]
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2 / iw
                    cy = (y1 + y2) / 2 / ih
                    w  = (x2 - x1) / iw
                    h  = (y2 - y1) / ih
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            n = len(lines)
            total_imgs  += 1
            total_boxes += n
            if n == 0:
                empty_imgs += 1

            status = f"{n} box(es)" if n > 0 else "no detections"
            print(f"    {img_path.name} -> {status}")

    print("\n" + "=" * 55)
    print("  AUTO-LABELING COMPLETE")
    print("=" * 55)
    print(f"  Total images processed : {total_imgs}")
    print(f"  Total boxes generated  : {total_boxes}")
    print(f"  Images with no boxes   : {empty_imgs}")
    print(f"  Avg boxes/image        : {total_boxes/max(total_imgs,1):.2f}")
    print("=" * 55)

    if empty_imgs == total_imgs:
        print("\n[WARN] No boxes were detected in ANY image.")
        print("       Try lowering --conf further, e.g. --conf 0.005")
    elif total_boxes < 20:
        print(f"\n[WARN] Very few boxes ({total_boxes}) detected.")
        print("       Consider re-running with a lower --conf value.")
    else:
        print("\n[OK] Labels written. You can now run:")
        print("       python scripts/train.py --model yolov8m.pt --epochs 300")


def main():
    parser = argparse.ArgumentParser(description="Auto-label chicken images with YOLO-World")
    parser.add_argument("--conf", type=float, default=0.01,
                        help="Confidence threshold for YOLO-World (default: 0.01)")
    args = parser.parse_args()
    auto_label(args.conf)


if __name__ == "__main__":
    main()
