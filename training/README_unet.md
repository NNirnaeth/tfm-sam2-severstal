# UNet++ Training Script for Severstal Dataset

This script trains a U-Net++ model on the full Severstal steel defect dataset.

## Features

- **Binary Segmentation**: Unifies all defect classes into a single "defect" class vs background
- **Aspect Ratio Preservation**: Resizes images to 1024x256 while maintaining aspect ratio with padding
- **Stratified Sampling**: Ensures balanced representation of images with/without defects
- **Bitmap Annotation Support**: Handles Supervisely format PNG bitmap compressed annotations
- **Same Metrics**: Uses the same evaluation metrics as SAM2 models (IoU, Dice, Precision, Recall, F1)
- **Early Stopping**: Monitors validation Dice and mIoU with configurable patience

## Usage

### Basic Training
```bash
python3 train_unet_base_full_dataset.py
```

### Custom Parameters
```bash
python3 train_unet_base_full_dataset.py \
    --lr 1e-4 \
    --batch_size 16 \
    --epochs 200 \
    --patience 10 \
    --save_dir models/unet_custom
```

### Available Arguments

- `--train_path`: Path to training data (default: `/home/ptp/sam2/datasets/Data/splits/train_split`)
- `--val_path`: Path to validation data (default: `/home/ptp/sam2/datasets/Data/splits/val_split`)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 100)
- `--img_size`: Image size [width, height] (default: [1024, 256])
- `--weight_decay`: Weight decay (default: 1e-4)
- `--patience`: Early stopping patience (default: 7)
- `--save_dir`: Directory to save models (default: `models/unet_severstal`)

## Model Architecture

- **U-Net++**: Enhanced U-Net with nested skip connections
- **Input**: 3-channel RGB images (1024x256)
- **Output**: 1-channel binary mask with sigmoid activation
- **Loss**: Combined Dice + Focal loss
- **Optimizer**: AdamW with cosine annealing scheduler

## Data Processing

- **Image Resizing**: Maintains aspect ratio with padding to 1024x256
- **Annotation Decoding**: Handles PNG bitmap compressed annotations from Supervisely format
- **Data Augmentation**: Horizontal/vertical flips, rotations, elastic transforms, noise, blur, color adjustments
- **Stratified Sampling**: Ensures balanced training set representation

## Training Features

- **Mixed Precision**: Uses AMP for faster training
- **Gradient Clipping**: Prevents gradient explosion
- **Checkpointing**: Saves best models by Dice and IoU, plus periodic checkpoints
- **Early Stopping**: Stops training when validation metrics don't improve
- **Learning Rate Scheduling**: Cosine annealing with warm restarts

## Output

- **Checkpoints**: Best models by Dice and IoU metrics
- **Periodic Checkpoints**: Every 10 epochs
- **Metrics Logging**: Real-time training and validation metrics
- **Model State**: Complete training state including optimizer state

## Requirements

Install dependencies from `requirements_unet.txt`:
```bash
pip install -r requirements_unet.txt
```

## Notes

- Uses seed=42 for reproducibility
- CUDA deterministic mode enabled
- Compatible with NumPy < 2.0
- Requires OpenCV < 4.12 for NumPy compatibility

