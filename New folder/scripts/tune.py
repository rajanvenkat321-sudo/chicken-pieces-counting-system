"""
tune.py
-------
Performs hyperparameter auto-tuning (genetic evolution) for the YOLOv8 model.
This searches for the best learning rates, loss parameters, and image augmentations.

Usage:
    python scripts/tune.py --epochs 30 --iterations 50
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML    = PROJECT_ROOT / "datasets" / "data.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"


def main():
    parser = argparse.ArgumentParser(description="Tune YOLOv8 hyperparameters using genetic search")
    parser.add_argument("--model", type=str, default="yolov8l.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                        help="YOLOv8 model base to tune")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train per tuning iteration (default: 30)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of genetic search iterations (default: 50)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: '' (auto), 'cpu', '0' (GPU)")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}. Run prepare_dataset.py first.")

    print("\n" + "=" * 60)
    # Highlight the computationally intensive nature of hyperparameter tuning
    print("  YOLOV8 HYPERPARAMETER AUTO-TUNING (GENETIC EVOLUTION)")
    print("=" * 60)
    print(f"  Base Model  : {args.model}")
    print(f"  Dataset     : {DATA_YAML}")
    print(f"  Epochs/Iter : {args.epochs}")
    print(f"  Iterations  : {args.iterations}")
    print(f"  Total Runs  : {args.iterations} runs of {args.epochs} epochs")
    print(f"  Device      : {args.device if args.device else 'Auto'}")
    print("=" * 60)
    print("[WARNING] This process is extremely computationally intensive!")
    print("          It is highly recommended to run on a GPU.")
    print("          To stop at any time, press Ctrl+C.")
    print("=" * 60 + "\n")

    print("[->] Initializing base model ...")
    model = YOLO(args.model)

    print("[->] Starting genetic hyperparameter tuning ...")
    # model.tune runs genetic algorithm over 'iterations' iterations.
    # In each iteration, it runs training for 'epochs' epochs.
    model.tune(
        data=str(DATA_YAML),
        epochs=args.epochs,
        iterations=args.iterations,
        optimizer="AdamW",
        single_cls=True,      # treat all labels as single class
        device=args.device,
        workers=0,            # required for Windows safety
        plots=True,
        save=False,           # don't save every run weight (saves disk space)
        val=True,
        project=str(RUNS_DIR),
        name="chicken_tune"
    )

    print("\n" + "=" * 60)
    print("  HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)
    print(f"  Optimized hyperparameters saved to: {RUNS_DIR / 'chicken_tune'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
