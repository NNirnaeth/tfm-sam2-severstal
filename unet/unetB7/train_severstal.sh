#!/bin/bash
# Training script for Severstal Steel Defect Dataset
# Uses separate train/val/test splits for proper evaluation

echo "🚀 Training UNet with EfficientNet-B7 on Severstal Dataset"
echo "=========================================================="

python train.py \
    --data_dir /home/javi/projects/SAM2/datasets/severstal/splits \
    --use_severstal_format \
    --train_split train_split \
    --val_split val_split \
    --test_split test_split \
    --image_dir img \
    --annotation_dir ann \
    --epochs 100 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --output_dir ./outputs \
    --experiment_name severstal_defect_detection \
    --patience 20 \
    --min_lr 1e-7 \
    --loss bce_dice \
    --metrics accuracy dice_coefficient \
    --random_seed 42

echo "✅ Training completed!"
echo "📊 Check outputs in ./outputs/severstal_defect_detection/"
