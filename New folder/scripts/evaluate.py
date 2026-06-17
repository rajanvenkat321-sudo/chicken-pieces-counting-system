"""
evaluate.py
-----------
Evaluates a trained chicken piece detector (single class).

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --weights runs/chicken_detect/weights/best.pt
  python scripts/evaluate.py --split test
  python scripts/evaluate.py --conf 0.25 --iou 0.5
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
DATA_YAML    = PROJECT_ROOT / "datasets" / "data.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"


def find_best_weights():
    candidates = sorted(RUNS_DIR.glob("**/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No best.pt found in runs/. Train a model first:\n"
            "  python scripts/train.py"
        )
    return candidates[-1]


def evaluate(args):
    weights = Path(args.weights) if args.weights else find_best_weights()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"[->] Loading model : {weights}")
    print(f"[->] Evaluating on '{args.split}' split ...\n")

    model = YOLO(str(weights))

    results = model.val(
        data      = str(DATA_YAML),
        split     = args.split,
        imgsz     = args.img,
        conf      = args.conf,
        iou       = args.iou,
        single_cls= True,
        plots     = True,
        save_json = args.save_json,
        project   = str(RUNS_DIR),
        name      = "eval",
        exist_ok  = True,
        workers   = 0,
    )

    box = results.box
    print("=" * 55)
    print("  DETECTION METRICS  (Class: chicken_piece)")
    print("=" * 55)
    print(f"  mAP@0.5        : {box.map50:.4f}")
    print(f"  mAP@0.5:0.95   : {box.map:.4f}")
    print(f"  Precision      : {box.mp:.4f}")
    print(f"  Recall         : {box.mr:.4f}")
    denom = box.mp + box.mr + 1e-9
    print(f"  F1 Score       : {2 * box.mp * box.mr / denom:.4f}")
    print("=" * 55)
    print(f"\n[OK] Plots saved to: {RUNS_DIR / 'eval'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 chicken piece detector")
    parser.add_argument("--weights",   type=str,   default=None,
                        help="Path to model weights (default: latest best.pt)")
    parser.add_argument("--split",     type=str,   default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--img",       type=int,   default=640)
    parser.add_argument("--conf",      type=float, default=0.25,
                        help="Confidence threshold (lower = more detections)")
    parser.add_argument("--iou",       type=float, default=0.5,
                        help="IoU threshold for NMS")
    parser.add_argument("--save_json", action="store_true",
                        help="Save COCO-format JSON results")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
