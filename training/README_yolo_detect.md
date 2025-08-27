# YOLO Detection Pipeline for Severstal Dataset

This directory contains scripts for training YOLO detection models on the Severstal dataset and generating bounding box predictions for SAM2 refinement.

## Scripts Overview

### 1. `new_src/experiments/prepare_yolo_detect_dataset.py`
**Purpose**: Converts bitmap annotations to YOLO format bounding boxes
**Input**: Supervisely format JSON with bitmap annotations
**Output**: YOLO detection dataset structure with images/ and labels/ directories

### 2. `train_yolo_detect_full_dataset.py`
**Purpose**: Trains YOLO detection model on prepared dataset
**Input**: Pre-prepared YOLO dataset
**Output**: Trained model and training metrics

### 3. `new_src/evaluation/eval_yolo_detect.py`
**Purpose**: Runs inference on test split and generates bounding boxes for SAM2
**Input**: Trained YOLO model and test images
**Output**: Bounding box predictions in SAM2-compatible format

### 4. `run_yolo_detect_pipeline.py`
**Purpose**: Orchestrates the complete workflow
**Input**: Command line arguments to control which steps to run
**Output**: Complete pipeline execution

## Quick Start

### Step 1: Prepare Dataset
```bash
python new_src/experiments/prepare_yolo_detect_dataset.py
```

This will:
- Convert bitmap annotations to YOLO format bboxes from hardcoded Severstal paths:
  - Annotations: `datasets/Data/splits/` (train_split, val_split, test_split)
  - Images: `datasets/Data/splits/` (individual image files)
  - Output: `datasets/yolo_detect/`
- Generate `yolo_detect.yaml` configuration file

### Step 2: Train Model
```bash
python train_yolo_detect_full_dataset.py \
    --lr 1e-4 \
    --epochs 100 \
    --batch_size 8
```

This will:
- Train YOLOv8s on the prepared dataset
- Save best model to `runs_tfm/yolov8s_det_lr1e-04_seed42/weights/best.pt`
- Save training metrics to CSV

### Step 3: Run Inference
```bash
python new_src/evaluation/eval_yolo_detect.py \
    --model_path runs_tfm/yolov8s_det_lr1e-04_seed42/weights/best.pt \
    --test_images_path data/severstal_det/images/test \
    --gt_annotations_path datasets/Data/splits/test_split.json \
    --images_path datasets/Data/images
```

This will:
- Generate bounding box predictions on test images
- Save predictions in SAM2-compatible format
- Calculate detection metrics if ground truth is available

## Complete Pipeline

Run all steps at once:
```bash
python run_yolo_detect_pipeline.py \
    --prepare_dataset \
    --train \
    --inference \
    --lr 1e-4 \
    --epochs 100 \
    --model_path runs_tfm/yolov8s_det_lr1e-04_seed42/weights/best.pt
```

## Dataset Structure

After preparation, you'll have:
```
datasets/yolo_detect/
├── images/
│   ├── train_split/          # Symlinks to original images
│   ├── val_split/            # Symlinks to original images
│   └── test_split/           # Symlinks to original images
├── labels/
│   ├── train_split/          # YOLO format labels (.txt)
│   ├── val_split/            # YOLO format labels (.txt)
│   └── test_split/           # YOLO format labels (.txt)
└── yolo_detect.yaml          # YOLO dataset configuration
```

## Label Format

Each `.txt` file contains one line per detection:
```
0 0.123456 0.234567 0.345678 0.456789
```

Where:
- `0`: Class ID (defect)
- `0.123456`: Normalized x_center
- `0.234567`: Normalized y_center
- `0.345678`: Normalized width
- `0.456789`: Normalized height

## Output for SAM2

The inference script generates:
- `yolo_predictions.json`: Main predictions file
- `bboxes_txt/`: Individual text files per image
- `results/`: Training and inference metrics

## Training Parameters

- **Model**: YOLOv8s (single class)
- **Image Size**: 1024x1024
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3 or 1e-4 (TFM comparison)
- **Weight Decay**: 1e-4
- **Data Augmentation**: No rotation (degrees=0), scale≤0.1
- **Early Stopping**: Patience=7 (monitor mAP50)

## Requirements

- Python 3.8+
- ultralytics (YOLO)
- OpenCV
- PIL/Pillow
- numpy
- tqdm
- PyYAML

## Troubleshooting

### Dataset not found
```bash
Error: Dataset YAML file not found: datasets/yolo_detect/yolo_detect.yaml
```
**Solution**: Run dataset preparation first:
```bash
python new_src/experiments/prepare_yolo_detect_dataset.py
```

**Note**: The script uses hardcoded paths:
- Annotations: `datasets/Data/splits/`
- Images: `datasets/Data/splits/`
- Output: `datasets/yolo_detect/`

### Missing dependencies
```bash
Error: YOLO command not found
```
**Solution**: Install ultralytics:
```bash
pip install ultralytics
```

### Memory issues
**Solution**: Reduce batch size:
```bash
python train_yolo_detect_full_dataset.py --batch_size 4
```

## Next Steps

After generating bounding boxes:
1. Use `yolo_predictions.json` as input to SAM2
2. Pass bboxes to SAM2 for mask refinement
3. Evaluate final segmentation performance