# 🐔 Chicken Pieces Detection & Counting System

![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-blue?style=for-the-badge&logo=ultralytics)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch)

A complete, end-to-end AI-powered computer vision system to detect and count chicken pieces as a single class (`chicken_piece`) using state-of-the-art **YOLOv8**. The pipeline is optimized for training on custom datasets with heavy data augmentation.

---

## ✨ Features

- **🚀 YOLOv8 Training Pipeline**: Automated script (`train.py`) with optimal hyperparameters and heavy data augmentation for custom datasets.
- **🔄 Label Conversion**: Utility script (`convert_labels.py`) to easily remap all class IDs to a single class (0) for focused detection.
- **📷 Inference & Counting**: Robust inference script (`predict.py`) to test the model on new images and provide accurate counts.
- **📊 Evaluation**: Evaluation script (`evaluate.py`) to generate validation and test metrics.

## 📁 Repository Structure

```text
chicken-pieces-counting-system/
├── datasets/                    # Image and label datasets
│   ├── images/                  # train, val, test splits
│   ├── labels/                  # YOLO format .txt labels
│   └── data.yaml                # Dataset configuration
├── scripts/
│   ├── convert_labels.py        # Remap all class IDs to 0
│   ├── prepare_dataset.py       # Dataset preparation & YAML generation
│   ├── train.py                 # Model training (YOLOv8m)
│   ├── evaluate.py              # Evaluation & metrics
│   └── predict.py               # Run inference on new images
├── runs/                        # Training outputs, weights, plots
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## 🛠️ Installation

1. **Navigate to the repository:**
   ```bash
   cd "New folder"
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r runs/requirements.txt
   ```
   *Note: Ensure `ultralytics`, `torch`, `torchvision`, and `opencv-python` are installed properly.*

## 🚀 Usage

### 1. Data Preparation
Convert all labels to a single class (run ONCE):
```bash
python scripts/convert_labels.py
```
Regenerate dataset configuration (`data.yaml`):
```bash
python scripts/prepare_dataset.py --generate_yaml_only
```

### 2. Training the Model
Start training (300 epochs, YOLOv8m, heavy augmentation):
```bash
python scripts/train.py
```
*For maximum accuracy (slower):*
```bash
python scripts/train.py --model yolov8l.pt --epochs 500
```

### 3. Evaluation
Evaluate the model on validation and test sets:
```bash
python scripts/evaluate.py --split val
python scripts/evaluate.py --split test
```

### 4. Prediction & Counting
Run inference on new images:
```bash
python scripts/predict.py --source datasets/images/test/
python scripts/predict.py --source path/to/your/image.jpg
```

## 🧠 Model Training Details

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

## 💡 Tips for Better Accuracy
- **More data = better accuracy.** Add more labeled images (aim for 100+).
- Use **[Roboflow](https://roboflow.com)** (web-based, easiest), **[CVAT](https://cvat.ai)**, or **LabelImg** to label new images.
- If you get 0 detections, lower the confidence threshold: `--conf 0.15`

## 🏷️ YOLO Label Format
Each `.txt` file (same name as image) — one row per chicken piece:
```
0 <x_center> <y_center> <width> <height>
```
*All values are **normalized** (0.0 – 1.0) relative to image size. Class ID is always `0` = `chicken_piece`.*
