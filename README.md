# SAM2 for Severstal Steel Defect Detection

This repository contains the implementation of SAM2 (Segment Anything Model 2) for steel defect detection on the Severstal dataset, along with baseline models (UNet, YOLO) for comparison.

## Project Structure

```
new_src/
├── training/                 # Training scripts for all models
│   ├── train_sam2_*.py     # SAM2 training scripts
│   ├── train_unet_*.py     # UNet training scripts
│   ├── train_yolo_*.py     # YOLO training scripts
│   └── config.py           # Training configuration
├── evaluation/              # Evaluation scripts
│   ├── eval_sam2_*.py      # SAM2 evaluation
│   ├── eval_unet_*.py      # UNet evaluation
│   ├── eval_yolo_*.py      # YOLO evaluation
│   └── config.py           # Evaluation configuration
├── experiments/             # Experiment configurations
├── utils/                   # Utility functions
└── .gitignore              # Excludes heavy result folders
```

## Models

- **SAM2**: Segment Anything Model 2 with LoRA fine-tuning
- **UNet**: Baseline U-Net++ architecture
- **YOLO**: YOLOv8 detection + SAM2 refinement pipeline

## Features

- Parametric dataset loading (25, 50, 100, 200+ images)
- LoRA fine-tuning for SAM2
- Comprehensive evaluation metrics (IoU, Dice, Precision, Recall, F1)
- Semi-supervised learning with pseudo-labels
- Post-processing utilities
- Visualization tools

## Installation

```bash
pip install -r requirements_unet.txt
```

## Usage

### Training

```bash
# SAM2 training
python training/train_sam2_large_subsets.py --config experiments/config.yaml

# UNet training
python training/train_unet_plus_plus.py --config experiments/config.yaml

# YOLO training
python training/train_yolov8_full_dataset.py --config experiments/config.yaml
```

### Evaluation

```bash
# Evaluate SAM2 models
python evaluation/eval_sam2_large_subsets.py --config experiments/config.yaml

# Evaluate UNet models
python evaluation/eval_unet_full_dataset.py --config experiments/config.yaml

# Evaluate YOLO pipeline
python evaluation/eval_yolo+sam2_full_dataset.py --config experiments/config.yaml
```

## Configuration

All experiments use YAML configuration files with parameters for:
- Model architecture
- Dataset subsets
- Training hyperparameters
- Prompt types (points, boxes)
- Post-processing options

## Results

Results are saved in CSV format with comprehensive metrics:
- IoU, IoU@50, IoU@75, IoU@90, IoU@95
- Dice coefficient
- Precision, Recall, F1 score
- Training parameters and timestamps

## License

This project is part of a Master's thesis on steel defect detection using SAM2.
