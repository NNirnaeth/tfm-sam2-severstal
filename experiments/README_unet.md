# UNet Training System - Modular Architecture

This directory contains a modular UNet training system for the Severstal dataset, **fully compliant with TFM requirements**.

## Files Structure

```
new_src/
├── experiments/
│   ├── model_unet.py              # UNet/UNet++ model definitions
│   ├── prepare_data_unet.py       # Data loading and preparation (TFM-compliant)
│   └── README_unet.md            # This file
└── training/
    └── train_unet_base_full_dataset.py  # Main training script (TFM-compliant)
```

## TFM Requirements Implemented

###  **Optimization**
- **AdamW** with `weight_decay=1e-4` (aligned with project standards)
- **Learning rates**: Compare `lr = 1e-3` vs `1e-4` (2 runs)
- **Mixed precision (AMP)**, gradient clipping (`max_norm=1.0`)

###  **Loss Function**
- **BCE + Dice** combination for stability with minority classes
- Eliminates IoU/Dice instability with imbalanced data

###  **Model Selection**
- **Primary**: Save "best-val" by mIoU or Dice (monitor principal)
- **Secondary**: Checkpoints every 10 epochs
- **Priority**: "Best until now" approach

###  **Threshold Optimization**
- **Learn optimal threshold** on validation set (max F1/Dice)
- **Report both**: Fixed=0.5 and optimal-val for transparency
- **Freeze threshold** for test evaluation

###  **Extra Metrics (SAM2 Comparison)**
- **IoU@{0.50, 0.75, 0.90}**
- **AUPRC** (defect class)
- **Recall@FP/image**
- **Single seed (42)** for reproducibility

###  **Stratification**
- **Stratified sampling** by defect presence in train/val
- **Balanced training** for minority class handling

###  **Preprocessing**
- **Horizontal aspect preservation**: `resize_longer + pad` to 1024×256
- **No stretching** of defects
- **PNG compressed in JSON** (Supervisely format) - validated decoding

###  **Augmentation (Steel-Specific)**
- **No vertical flip** (preserves steel rolling physics)
- **Limited rotation**: ±5-10° max (avoids defect distortion)
- **Mild transformations**: Avoid strong elastic that deforms defects
- **Steel-appropriate**: Brightness/contrast, Gaussian noise, mild blur

###  **Stability & Cost**
- **Mixed precision (AMP)**
- **Gradient clipping** (`max_norm=1.0`)
- **Fixed seed (42)** and deterministic computation
- **Performance metrics**: Time/epoch, ms/img inference

###  **Tracing & Logging**
- **W&B/MLflow** integration
- **CSV metrics** per epoch
- **Predicted masks** for val/test (TFM figures)
- **Comprehensive logging** for reproducibility

## Usage

### TFM Experiments (2 runs)

For your TFM, you need to run **2 experiments** to compare learning rates:

```bash
cd new_src/training

# Experiment 1: lr=1e-3 (default)
python train_unet_base_full_dataset.py \
    --unet_plus_plus \
    --encoder resnet34 \
    --lr 1e-3 \
    --seed 42 \
    --save_dir models/unet_lr1e-3_seed42 \
    --use_wandb

# Experiment 2: lr=1e-4
python train_unet_base_full_dataset.py \
    --unet_plus_plus \
    --encoder resnet34 \
    --lr 1e-4 \
    --seed 42 \
    --save_dir models/unet_lr1e-4_seed42 \
    --use_wandb
```

### Single Experiment

```bash
# UNet++ with ResNet34, custom configuration
python train_unet_base_full_dataset.py \
    --unet_plus_plus \
    --encoder resnet50 \
    --lr 5e-4 \
    --batch_size 16 \
    --epochs 150 \
    --img_size 1024 256 \
    --save_dir models/unet_custom \
    --seed 42 \
    --use_wandb
```

## Key Features

### Model Selection
- **UNet**: Standard U-Net architecture
- **UNet++**: Enhanced version with dense skip connections
- **Encoders**: ResNet34 (faster) or ResNet50 (more accurate)

### Data Preparation
- **Automatic loading** of Severstal dataset
- **Stratified sampling** for balanced training
- **TFM-specific augmentation** for steel defects
- **Horizontal aspect preservation** with padding

### Training Features
- **Mixed precision training (AMP)**
- **Gradient clipping** for stability
- **Early stopping** with patience
- **Cosine annealing** with warm restarts
- **Comprehensive checkpointing**

## Output Structure

```
models/unet_lr1e-3_seed42/
├── best_dice_0.8234_seed42.pth          # Best model by Dice
├── best_iou_0.7891_seed42.pth           # Best model by IoU
├── checkpoint_epoch_10_seed42.pth        # Periodic checkpoints
├── checkpoint_epoch_20_seed42.pth
├── metrics_seed42.csv                    # All epoch metrics
└── args.json                             # Experiment configuration
```

## Metrics CSV Columns

- `epoch`, `train_loss`, `val_loss`
- `val_iou_fixed`, `val_dice_fixed` (threshold=0.5)
- `val_iou_opt`, `val_dice_opt` (optimal threshold)
- `opt_threshold_f1`, `opt_threshold_dice`
- `iou_50`, `iou_75`, `iou_90` (IoU at thresholds)
- `auprc`, `recall_fp_image` (extra metrics)
- `learning_rate`, `epoch_time`

## TFM Comparison with SAM2

The system provides **all required metrics** for fair comparison:

1. **Segmentation Quality**: IoU, Dice, IoU@{0.50, 0.75, 0.90}
2. **Class Performance**: AUPRC, Precision, Recall, F1
3. **Reproducibility**: Fixed seed (42) for consistent results
4. **Efficiency**: Time/epoch, ms/img inference
5. **Threshold Optimization**: Learned vs fixed comparison

## Dependencies

- PyTorch (with AMP support)
- Albumentations
- OpenCV
- NumPy
- scikit-learn
- tqdm
- wandb (optional)
- pandas (for CSV handling)

## Reproducibility

- **Fixed seed (42)** for all experiments
- **Deterministic computation** (`torch.backends.cudnn.deterministic=True`)
- **Comprehensive logging** of all parameters
- **Checkpoint saving** with full state
- **Metrics export** for analysis

## Example Output

```
Model: UNetPlusPlus
Encoder: resnet34
Total parameters: 23,518,081

Starting training with:
  Model: UNetPlusPlus
  Encoder: resnet34
  Learning rate: 0.001
  Batch size: 8
  Weight decay: 0.0001
  Image size: [1024, 256]
  Early stopping patience: 7
  Seed: 42

Epoch 1/100
--------------------------------------------------
Training: 100%|██████████| 525/525 [02:34<00:00]
Validation: 100%|██████████| 59/59 [00:15<00:00]

Train Loss: 0.8234
Val Loss: 0.7123
Val IoU (fixed 0.5): 0.6234
Val Dice (fixed 0.5): 0.7123
Val IoU (opt thresh): 0.6789
Val Dice (opt thresh): 0.7456
Optimal threshold (F1): 0.450
Optimal threshold (Dice): 0.475
Epoch time: 169.45s
Learning Rate: 1.00e-03
```
