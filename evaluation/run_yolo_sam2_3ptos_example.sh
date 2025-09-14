#!/bin/bash

# Example script to run YOLO+SAM2+3Points evaluation
# This script shows how to use the eval_yolo+sam2+3ptos.py script

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_yolo+sam2+3ptos.py"

# Default paths (adjust according to your setup)
YOLO_PATH="${SCRIPT_DIR}/../training/training_results/yolo_detection/best.pt"
SAM2_CHECKPOINT="${SCRIPT_DIR}/../training/training_results/sam2_full_dataset/sam2_large_20241201-1234/best_step2000_iou0.7500_dice0.8200.torch"
GT_ANNOTATIONS="${SCRIPT_DIR}/../../datasets/severstal_full/annotations_test"
TEST_IMAGES_DIR="${SCRIPT_DIR}/../../datasets/severstal_full/images_test"
OUTPUT_DIR="${SCRIPT_DIR}/evaluation_results/yolo_sam2_gt_points"

# Alternative: Use YOLO predictions from .txt files instead of model
# YOLO_PATH="${SCRIPT_DIR}/evaluation_results/yolo_detection/predictions"

echo "🚀 Running YOLO+SAM2+3Points Evaluation"
echo "======================================="
echo "YOLO: $YOLO_PATH"
echo "SAM2: $SAM2_CHECKPOINT"
echo "GT:   $GT_ANNOTATIONS"
echo "Test: $TEST_IMAGES_DIR"
echo ""

# Check if files exist
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "❌ Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

if [ ! -f "$SAM2_CHECKPOINT" ]; then
    echo "❌ SAM2 checkpoint not found: $SAM2_CHECKPOINT"
    echo "💡 Use find_best_sam2_model.py to locate the best checkpoint"
    exit 1
fi

if [ ! -d "$GT_ANNOTATIONS" ]; then
    echo "❌ GT annotations directory not found: $GT_ANNOTATIONS"
    exit 1
fi

if [ ! -d "$TEST_IMAGES_DIR" ]; then
    echo "❌ Test images directory not found: $TEST_IMAGES_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "✅ All paths validated. Starting evaluation..."
echo ""

# Run evaluation with 3 GT points
python3 "$EVAL_SCRIPT" \
    --yolo_path "$YOLO_PATH" \
    --sam2_checkpoint "$SAM2_CHECKPOINT" \
    --sam2_model_type "sam2_hiera_l" \
    --gt_annotations "$GT_ANNOTATIONS" \
    --test_images_dir "$TEST_IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_gt_points 2 \
    --conf_threshold 0.25 \
    --iou_threshold 0.70 \
    --fill_holes \
    --morphology \
    --batch_size 50

echo ""
echo "🎉 Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"

# Optional: Run with different numbers of GT points for comparison
read -p "🤔 Run with 1 and 2 GT points for comparison? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running with 1 GT point..."
    python3 "$EVAL_SCRIPT" \
        --yolo_path "$YOLO_PATH" \
        --sam2_checkpoint "$SAM2_CHECKPOINT" \
        --sam2_model_type "sam2_hiera_l" \
        --gt_annotations "$GT_ANNOTATIONS" \
        --test_images_dir "$TEST_IMAGES_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_gt_points 1 \
        --conf_threshold 0.25 \
        --iou_threshold 0.70 \
        --fill_holes \
        --morphology \
        --batch_size 50

    echo "Running with 2 GT points..."
    python3 "$EVAL_SCRIPT" \
        --yolo_path "$YOLO_PATH" \
        --sam2_checkpoint "$SAM2_CHECKPOINT" \
        --sam2_model_type "sam2_hiera_l" \
        --gt_annotations "$GT_ANNOTATIONS" \
        --test_images_dir "$TEST_IMAGES_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_gt_points 2 \
        --conf_threshold 0.25 \
        --iou_threshold 0.70 \
        --fill_holes \
        --morphology \
        --batch_size 50
    
    echo "🎉 All evaluations completed!"
fi
