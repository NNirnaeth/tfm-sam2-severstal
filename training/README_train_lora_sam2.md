# SAM2 LoRA Training Script

## Overview

The `train_lora_sam2.py` script is a modified version of the original `train_lora_sam2_full_dataset.py` that supports training SAM2 with LoRA adapters on different dataset sizes.

## Key Changes

1. **Renamed**: `train_lora_sam2_full_dataset.py` → `train_lora_sam2.py`
2. **Added dataset size selection**: Support for full dataset and subsets (500, 1000, 2000 images)
3. **Flexible data loading**: Automatically selects the correct dataset path based on size
4. **Updated naming**: Model names and output directories include dataset size information

## Supported Dataset Sizes

- `full`: Full training dataset (`datasets/Data/splits/train_split`)
- `500`: 500-image subset (`datasets/Data/splits/subsets/500_subset`)
- `1000`: 1000-image subset (`datasets/Data/splits/subsets/1000_subset`)
- `2000`: 2000-image subset (`datasets/Data/splits/subsets/2000_subset`)

## Usage

### Basic Usage

```bash
# Train on full dataset (default)
python new_src/training/train_lora_sam2.py

# Train on 500-image subset
python new_src/training/train_lora_sam2.py --dataset_size 500

# Train on 1000-image subset
python new_src/training/train_lora_sam2.py --dataset_size 1000

# Train on 2000-image subset
python new_src/training/train_lora_sam2.py --dataset_size 2000
```

### Advanced Usage

```bash
python new_src/training/train_lora_sam2.py \
    --dataset_size 1000 \
    --steps 5000 \
    --learning_rate 1e-3 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --batch_size 1
```

## Arguments

- `--dataset_size`: Dataset size ("full", "500", "1000", "2000")
- `--learning_rate`: Learning rate for LoRA training (default: 1e-3)
- `--weight_decay`: Weight decay for optimizer (default: 0.01)
- `--steps`: Number of training steps (default: 10000)
- `--batch_size`: Batch size (default: 1, effective batch = batch_size * 4)
- `--lora_rank`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA scaling factor (default: 16)
- `--lora_dropout`: LoRA dropout rate (default: 0.05)

## Output Structure

Checkpoints are saved to:
```
new_src/training/training_results/sam2_lora/
└── sam2_large_lora_r8_a16_lr1e-3_{dataset_size}_{timestamp}/
    ├── best_lora_step{step}_iou{val_iou:.4f}_dice{val_dice:.4f}.torch
    ├── best_merged_step{step}_iou{val_iou:.4f}_dice{val_dice:.4f}.torch
    ├── latest_lora_step{step}.torch
    ├── final_lora_step{steps}.torch
    └── training_summary.json
```

## Training Features

- **LoRA Adaptation**: Only attention projections (Q, K, V, O) are fine-tuned
- **Early Stopping**: Patience of 7 validations with minimum delta of 1e-4
- **Mixed Precision**: Uses AMP for memory efficiency
- **Gradient Clipping**: Max norm of 1.0
- **Cosine Scheduler**: With warm restarts
- **Validation**: Every 500 steps on internal validation set
- **Checkpointing**: Saves best and latest models

## Evaluation

After training, use the evaluation script to test the model:

```bash
# Evaluate with GT points (3 points)
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --backbone_ckpt models/sam2_base_models/sam2_hiera_large.pt \
    --lora_checkpoint new_src/training/training_results/sam2_lora/sam2_large_lora_r8_a16_lr1e-3_1000_20241201_1200/best_lora_step5000_iou0.8500_dice0.9200.torch \
    --evaluation_mode gt_points \
    --num_gt_points 3
```

## Example Script

Use `example_train_lora_sam2.py` to see example configurations and commands:

```bash
python new_src/training/example_train_lora_sam2.py
```

## Notes

- The script uses the same LoRA implementation as the original
- Training uses 3 GT points (centroid + max distance + negative point)
- All hyperparameters are consistent with the original implementation
- Results are saved in CSV format for comparative analysis

