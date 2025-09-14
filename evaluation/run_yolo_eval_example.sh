#!/bin/bash
# Example script to run YOLO detection evaluation on corrected dataset

# Set paths
MODEL_PATH="/home/ptp/sam2/new_src/training/training_results/yolo_detection/yolo_detect_corrected/weights/best.pt"
DATASET_YAML="/home/ptp/sam2/datasets/yolo_detection_fixed/yolo_detection.yaml"
OUTPUT_DIR="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection"

# Run evaluation
python new_src/evaluation/eval_yolo_detect_corrected.py \
    --model-path "$MODEL_PATH" \
    --dataset-yaml "$DATASET_YAML" \
    --output-dir "$OUTPUT_DIR" \
    --conf 0.25 \
    --iou 0.70 \
    --max-det 300 \
    --agnostic-nms \
    --imgsz 1024 \
    --save-dir "$OUTPUT_DIR"

echo "Evaluation completed! Check the output directory for results."
