# SAM2 Large LoRA Training

This directory contains the LoRA (Low-Rank Adaptation) training script for SAM2 Large model on the Severstal dataset.

## Overview

The `train_lora_sam2_full_dataset.py` script implements efficient fine-tuning of SAM2 Large using LoRA adapters. This approach:

- **Freezes all base model weights** to preserve pre-trained knowledge
- **Adds low-rank adaptation matrices** only to attention projections (Q, K, V, O)
- **Reduces trainable parameters** by ~99% compared to full fine-tuning
- **Maintains model performance** while enabling efficient adaptation

## LoRA Implementation

### Architecture
- **LoRA Layer**: Implements low-rank decomposition with matrices A and B
- **LoRA Linear**: Wraps existing linear layers with LoRA adaptation
- **Target Modules**: Applied to attention projections in image encoder and mask decoder

### Hyperparameters
- **Rank (r)**: 8 (default), 4, or 16 for ablation studies
- **Alpha (α)**: 16 (scaling factor, rule of thumb: α ≈ 2×r)
- **Dropout**: 0.05 (0.1 if training is unstable)
- **Bias**: None (LoRA adapters don't add bias terms)

## Usage

### Basic Training
```bash
python3 train_lora_sam2_full_dataset.py
```

### Custom Parameters
```bash
python3 train_lora_sam2_full_dataset.py \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --steps 10000 \
    --batch_size 1 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05
```

### Ablation Studies
```bash
# Test different LoRA ranks
python3 train_lora_sam2_full_dataset.py --lora_rank 4
python3 train_lora_sam2_full_dataset.py --lora_rank 8
python3 train_lora_sam2_full_dataset.py --lora_rank 16
```

## Training Configuration

### Dataset
- **Training**: `datasets/Data/splits/train_split` (full training set)
- **Validation**: `datasets/Data/splits/val_split` (internal validation)
- **Test**: `datasets/Data/splits/test_split` (final evaluation)

### Training Parameters
- **Optimizer**: AdamW (only for LoRA parameters)
- **Learning Rate**: 1e-3 (higher than traditional fine-tuning)
- **Weight Decay**: 0.01
- **Scheduler**: Cosine annealing with warmup
- **Mixed Precision**: AMP enabled
- **Gradient Clipping**: 1.0

### Early Stopping
- **Patience**: 7 validation cycles
- **Monitor**: Validation IoU
- **Min Delta**: 1e-4 improvement threshold

## Output

### Checkpoints
- **LoRA Adapters**: `best_lora_step{step}_iou{val_iou}_dice{val_dice}.torch`
- **Merged Model**: `best_merged_step{step}_iou{val_iou}_dice{val_dice}.torch`
- **Latest**: `latest_lora_step{step}.torch`

### Metrics
- **Training**: IoU, Loss, Learning Rate (every 100 steps)
- **Validation**: IoU, Dice (every 500 steps)
- **Logging**: CSV format in `logs/training_metrics_lora_{timestamp}.csv`

### Directory Structure
```
models/severstal_lora/
└── sam2_large_lora_r8_a16_lr1e-3_{timestamp}/
    ├── best_lora_step{step}_iou{val_iou}_dice{val_dice}.torch
    ├── best_merged_step{step}_iou{val_iou}_dice{val_dice}.torch
    ├── latest_lora_step{step}.torch
    ├── final_lora_step{steps}.torch
    └── training_summary.json
```

## Testing

Run the test script to verify LoRA implementation:
```bash
python3 test_lora_implementation.py
```

This tests:
- LoRA layer forward pass
- LoRA linear wrapper
- Parameter counting and freezing

## Key Features

### Efficiency
- **Memory**: ~99% reduction in trainable parameters
- **Speed**: Faster training and inference
- **Storage**: Compact LoRA adapters (~1-2% of base model size)

### Flexibility
- **Modular**: Easy to apply/remove LoRA adapters
- **Configurable**: Adjustable rank, alpha, and dropout
- **Compatible**: Works with existing SAM2 pipeline

### Reproducibility
- **Seeds**: Fixed random seeds (42)
- **Deterministic**: CUDA deterministic mode enabled
- **Logging**: Comprehensive training logs and metrics

## Comparison with Full Fine-tuning

| Aspect | Full Fine-tuning | LoRA |
|--------|------------------|------|
| Trainable Parameters | 100% | ~1% |
| Memory Usage | High | Low |
| Training Speed | Slow | Fast |
| Storage | Large | Small |
| Performance | Baseline | Comparable |
| Flexibility | Low | High |

## Notes

- **Base Model**: Uses SAM2 Hiera Large (`sam2_hiera_large.pt`)
- **Prompt Encoder**: Kept frozen (no LoRA adaptation)
- **Image Encoder**: LoRA applied to attention projections only
- **Mask Decoder**: LoRA applied to attention and MLP layers
- **Validation**: Internal validation every 500 steps
- **Checkpointing**: Saves both LoRA adapters and merged models

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or LoRA rank
2. **Training Instability**: Increase LoRA dropout to 0.1
3. **Poor Performance**: Try different LoRA ranks (4, 8, 16)
4. **Import Errors**: Check SAM2 library path in `libs/sam2base`

### Performance Tips
- Use mixed precision (AMP) for faster training
- Monitor validation metrics for early stopping
- Save LoRA adapters separately for easy deployment
- Use merged models for inference to avoid LoRA overhead

