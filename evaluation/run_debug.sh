#!/bin/bash

# Debug script for YOLO+SAM2 pipeline
# This script runs the debug script with the existing data paths

echo "Running YOLO+SAM2 pipeline debug..."

# Check if we have the required data
YOLO_LABELS_DIR="evaluation_results/yolo_detection/predict_test/labels"
GT_ANNOTATIONS_DIR="datasets/yolo_detection_fixed/labels/test_split"
TEST_IMAGES_DIR="datasets/yolo_detection_fixed/images/test_split"
SAM2_CHECKPOINT="new_src/training/training_results/sam2_full_dataset/sam2_large_full_dataset_lr0.0001_20250829_1646/best_step11000_iou0.5124_dice0.6461.torch"

# Check if directories exist
if [ ! -d "$YOLO_LABELS_DIR" ]; then
    echo "Error: YOLO labels directory not found: $YOLO_LABELS_DIR"
    echo "Please run the YOLO evaluation first:"
    echo "python new_src/evaluation/eval_yolo_detect_full_dataset.py --model_path <path_to_yolo_model>"
    exit 1
fi

if [ ! -d "$GT_ANNOTATIONS_DIR" ]; then
    echo "Error: Ground truth annotations directory not found: $GT_ANNOTATIONS_DIR"
    exit 1
fi

if [ ! -d "$TEST_IMAGES_DIR" ]; then
    echo "Error: Test images directory not found: $TEST_IMAGES_DIR"
    exit 1
fi

if [ ! -f "$SAM2_CHECKPOINT" ]; then
    echo "Error: SAM2 checkpoint not found: $SAM2_CHECKPOINT"
    echo "Please check the path or train SAM2 first"
    exit 1
fi

echo "Found required directories and files:"
echo "  YOLO labels: $YOLO_LABELS_DIR"
echo "  GT annotations: $GT_ANNOTATIONS_DIR"
echo "  Test images: $TEST_IMAGES_DIR"
echo "  SAM2 checkpoint: $SAM2_CHECKPOINT"

# Run the debug script
echo ""
echo "Starting debug analysis..."
python new_src/evaluation/debug_yolo_sam2_pipeline.py \
    --yolo_labels_dir "$YOLO_LABELS_DIR" \
    --sam2_checkpoint "$SAM2_CHECKPOINT" \
    --sam2_model_type "sam2_hiera_l" \
    --gt_annotations "$GT_ANNOTATIONS_DIR" \
    --test_images_dir "$TEST_IMAGES_DIR" \
    --output_dir "new_src/evaluation/debug_results" \
    --num_images 3

echo ""
echo "Debug analysis completed!"
echo "Check the debug_results directory for visualizations and console output for detailed analysis."
