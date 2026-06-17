"""
predict.py
----------
Run inference with the trained chicken piece detector (single class).

Usage:
  # Test set folder
  python scripts/predict.py --source datasets/images/test/

  # Single image
  python scripts/predict.py --source path/to/image.jpg

  # Webcam
  python scripts/predict.py --source 0

  # Custom weights and threshold
  python scripts/predict.py \
      --weights runs/chicken_detect/weights/best.pt \
      --source  datasets/images/test/ \
      --conf    0.25 \
      --save
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
RUNS_DIR     = PROJECT_ROOT / "runs"


def find_best_weights():
    candidates = sorted(RUNS_DIR.glob("**/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No weights found in runs/. Train a model first:\n"
            "  python scripts/train.py"
        )
    return candidates[-1]


def predict(args):
    weights = Path(args.weights) if args.weights else find_best_weights()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"[->] Model   : {weights}")
    print(f"[->] Source  : {args.source}")
    print(f"[->] Conf    : {args.conf}  (lower = detects more, higher = fewer false positives)")
    print(f"[->] IoU     : {args.iou}")
    print()

    model = YOLO(str(weights))

    results = model.predict(
        source      = args.source,
        imgsz       = args.img,
        conf        = args.conf,
        iou         = args.iou,
        save        = args.save,
        save_txt    = args.save_txt,
        save_conf   = True,
        show        = args.show,
        line_width  = 3,
        project     = str(RUNS_DIR),
        name        = "predict",
        exist_ok    = True,
        stream      = True,
        single_cls  = True,
    )

    total = 0
    for i, result in enumerate(results):
        boxes = result.boxes
        n     = len(boxes) if boxes is not None else 0
        total += n

        img_name = Path(result.path).name if result.path else f"image_{i+1}"

        if n == 0:
            print(f"  [{i+1}] {img_name} -- no chicken pieces detected")
            print(f"        (try lowering --conf below {args.conf})")
        else:
            print(f"\n  [{i+1}] {img_name} -- {n} chicken piece(s) detected:")
            for j, box in enumerate(boxes):
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"       Piece {j+1}: conf={conf:.2f}  "
                      f"bbox=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]")

    print(f"\n[OK] Total chicken pieces detected: {total}")
    if args.save:
        print(f"[OK] Annotated images saved to: {RUNS_DIR / 'predict'}")


def main():
    parser = argparse.ArgumentParser(description="Detect chicken pieces with YOLOv8")
    parser.add_argument("--weights",  type=str,   default=None,
                        help="Path to weights (default: latest best.pt in runs/)")
    parser.add_argument("--source",   type=str,   required=True,
                        help="Image, folder, or webcam index (0)")
    parser.add_argument("--img",      type=int,   default=640)
    parser.add_argument("--conf",     type=float, default=0.25,
                        help="Confidence threshold 0-1 (default: 0.25)")
    parser.add_argument("--iou",      type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--save",     action="store_true", default=True,
                        help="Save annotated output images (default: True)")
    parser.add_argument("--save_txt", action="store_true",
                        help="Also save detections as YOLO .txt files")
    parser.add_argument("--show",     action="store_true",
                        help="Display results window (needs display)")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
