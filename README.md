# 🐔 Chicken Pieces Detection & Counting System — YOLOv8 Pipeline

An AI-powered computer vision system to detect and count chicken pieces as a single class (`chicken_piece`) using YOLOv8. The pipeline is optimized for training on custom datasets with heavy data augmentation.

## Project Structure
```
├── New folder/
│   ├── datasets/
│   │   ├── images/
│   │   │   ├── train/        ← Training images (10 images)
│   │   │   ├── val/          ← Validation images (3 images)
│   │   │   └── test/         ← Test images (2 images)
│   │   ├── labels/
│   │   │   ├── train/        ← YOLO .txt labels (all class IDs = 0)
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── data.yaml         ← Dataset config (nc=1, class: chicken_piece)
│   ├── scripts/
│   │   ├── convert_labels.py    ← Remap all class IDs to 0 (run once)
│   │   ├── prepare_dataset.py   ← Dataset preparation & YAML generation
│   │   ├── train.py             ← Model training (YOLOv8m, 300 epochs)
│   │   ├── evaluate.py          ← Evaluation & metrics
│   │   └── predict.py           ← Run inference on new images
│   ├── runs/                    ← Training outputs, weights, plots
│   └── requirements.txt
└── README.md
```

## Quick Start

### Step 1 — Install dependencies
Navigate to the project directory and install the required dependencies:
```bash
cd "New folder"
pip install -r runs/requirements.txt
```

### Step 2 — Convert labels to single class (run ONCE)
```bash
python scripts/convert_labels.py
```

### Step 3 — Regenerate data.yaml
```bash
python scripts/prepare_dataset.py --generate_yaml_only
```

### Step 4 — Train (300 epochs, YOLOv8m, heavy augmentation)
```bash
python scripts/train.py
```
For maximum accuracy (slower):
```bash
python scripts/train.py --model yolov8l.pt --epochs 500
```

### Step 5 — Evaluate
```bash
python scripts/evaluate.py --split val
python scripts/evaluate.py --split test
```

### Step 6 — Predict and Count on new images
```bash
python scripts/predict.py --source datasets/images/test/
python scripts/predict.py --source path/to/your/image.jpg
```

## YOLO Label Format
Each `.txt` file (same name as image) — one row per chicken piece:
```
0 <x_center> <y_center> <width> <height>
```
All values are **normalized** (0.0 – 1.0) relative to image size.
Class ID is always `0` = `chicken_piece`.

Example — two pieces in one image:
```
0 0.5 0.5 0.4 0.3
0 0.2 0.3 0.1 0.2
```

## Model Training Details

| Setting         | Value             |
|----------------|-------------------|
| Model          | YOLOv8m (medium)  |
| Class          | chicken_piece (1) |
| Epochs         | 300               |
| Batch size     | Auto (≤8)         |
| Image size     | 640×640           |
| LR schedule    | Cosine annealing  |
| Augmentation   | Heavy (mosaic, mixup, flips, erasing) |
| Cache          | RAM (fast for small datasets) |

## Tips for Better Accuracy
- **More data = better accuracy.** Add more labeled images (aim for 100+).
- Use [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/HumanSignal/labelImg) to label new images.
- If you get 0 detections, lower the confidence threshold: `--conf 0.15`
- For more training time: `--epochs 500 --model yolov8l.pt`

## Recommended Labeling Tools
- **Roboflow** (web-based, easiest): https://roboflow.com
- **LabelImg**: `pip install labelImg && labelImg`
- **CVAT**: https://cvat.ai
