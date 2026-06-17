<div align="center">
  <h1>🍗 Chicken Pieces Detection & Counting System</h1>
  <p><strong>A complete, end-to-end AI-powered computer vision system to detect and count chicken pieces.</strong></p>

  ![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-blue?style=for-the-badge&logo=ultralytics)
  ![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge&logo=python)
  ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch)
  ![Tkinter](https://img.shields.io/badge/GUI-Tkinter-purple?style=for-the-badge)

</div>

---

## ✨ Key Features

- **🚀 Live Dashboard (`dashboard.py`)**: A premium Tkinter GUI for real-time inference. Select images, adjust confidence/IoU thresholds on the fly, and see dynamic bounding boxes and piece counts. Includes an **Auto-Tune** feature!
- **🤖 YOLOv8 Training Pipeline (`train.py`)**: Automated script with optimal hyperparameters and heavy data augmentation for robust detection.
- **🛠️ Dataset Utilities**:
  - `convert_labels.py`: Remap class IDs to a single class (0).
  - `prepare_dataset.py`: Prepare YOLO `.yaml` configuration dynamically.
  - `auto_label.py`: Auto-annotate new images using your trained model.
  - `add_backgrounds.py`: Data augmentation script to introduce varied backgrounds.
- **📈 Tuning & Evaluation**:
  - `tune.py`: Hyperparameter evolution for maximizing mAP.
  - `evaluate.py`: Generate rigorous validation metrics.

## 📁 Repository Structure

```text
chicken-pieces-counting-system/
├── datasets/                    # Image and label datasets
│   ├── images/                  # train, val, test splits
│   ├── labels/                  # YOLO format .txt labels
│   └── data.yaml                # Dataset configuration
├── scripts/
│   ├── dashboard.py             # Live GUI application
│   ├── train.py                 # Model training (YOLOv8)
│   ├── predict.py               # Run inference from CLI
│   ├── evaluate.py              # Evaluation & metrics
│   ├── tune.py                  # Hyperparameter tuning
│   ├── auto_label.py            # Auto-annotate new datasets
│   ├── add_backgrounds.py       # Background augmentation
│   ├── convert_labels.py        # Remap all class IDs to 0
│   └── prepare_dataset.py       # Dataset preparation & YAML generation
├── runs/                        # Training outputs, weights, plots
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rajanvenkat321-sudo/chicken-pieces-counting-system.git
   cd chicken-pieces-counting-system
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note**: Ensure you have `ultralytics`, `torch`, `torchvision`, `opencv-python`, and `Pillow` installed.

## 🚀 Usage Guide

### 🎨 1. Launch the Live Dashboard (Recommended)
Experience the model interactively through the built-in GUI:
```bash
python scripts/dashboard.py
```
*Features: Image selection, interactive confidence/IoU sliders, auto-tuning, and real-time detection rendering.*

### 🛠️ 2. Data Preparation
Convert all labels to a single class (Run ONCE):
```bash
python scripts/convert_labels.py
```
Regenerate dataset configuration (`data.yaml`):
```bash
python scripts/prepare_dataset.py --generate_yaml_only
```

### 🧠 3. Training the Model
Start training (YOLOv8m, heavy augmentation):
```bash
python scripts/train.py
```

### 📊 4. Evaluation & Prediction
Evaluate the model on the validation set:
```bash
python scripts/evaluate.py --split val
```
Run CLI prediction on new images:
```bash
python scripts/predict.py --source datasets/images/test/
```

## 🧠 Model Specifications

| Setting | Value |
|---------|-------|
| **Architecture** | YOLOv8m (Medium) |
| **Classes** | `chicken_piece` (ID: 0) |
| **Input Size** | 640x640 |
| **Optimization** | Cosine Annealing LR |
| **Augmentation** | Heavy (Mosaic, Mixup, Flips, Erasing) |

## 💡 Tips for Maximum Accuracy
- **Expand the Dataset**: More labeled images = better generalization.
- **Tune Hyperparameters**: Use `python scripts/tune.py` to evolve hyperparameters specifically for your dataset.
- **Auto-labeling**: Use `scripts/auto_label.py` with your best weights to quickly label new unannotated images.
- **Threshold Adjustment**: If false positives occur, increase the Confidence threshold. If pieces are missed, lower it. Use the Auto-Tune button in the dashboard!
