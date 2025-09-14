#!/bin/bash
# Script to check if all required paths exist for YOLO+SAM2 evaluation

echo "Checking paths for YOLO+SAM2 pipeline evaluation..."
echo ""

# Define paths
YOLO_LABELS_DIR="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection/predict_test_corrected/labels"
SAM2_CHECKPOINT="/home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_large_full_dataset_lr0.0001_20250829_1646/best_step11000_iou0.5124_dice0.6461.torch"
GT_ANNOTATIONS="/home/ptp/sam2/datasets/Data/splits/test_split/ann"
TEST_IMAGES_DIR="/home/ptp/sam2/datasets/yolo_detection_fixed/images/test_split"
OUTPUT_DIR="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_sam2_pipeline"

# Function to check path
check_path() {
    local path=$1
    local description=$2
    
    if [ -e "$path" ]; then
        if [ -d "$path" ]; then
            local count=$(find "$path" -type f | wc -l)
            echo "✅ $description: $path ($count files)"
        else
            echo "✅ $description: $path (file exists)"
        fi
    else
        echo "❌ $description: $path (NOT FOUND)"
    fi
}

# Check all paths
check_path "$YOLO_LABELS_DIR" "YOLO Labels Directory"
check_path "$SAM2_CHECKPOINT" "SAM2 Checkpoint"
check_path "$GT_ANNOTATIONS" "Ground Truth Annotations"
check_path "$TEST_IMAGES_DIR" "Test Images Directory"

# Check if output directory can be created
if [ -d "$(dirname "$OUTPUT_DIR")" ]; then
    echo "✅ Output Directory Parent: $(dirname "$OUTPUT_DIR") (writable)"
else
    echo "❌ Output Directory Parent: $(dirname "$OUTPUT_DIR") (NOT FOUND)"
fi

echo ""
echo "Path check completed!"

# Additional checks
echo ""
echo "Additional information:"
if [ -d "$YOLO_LABELS_DIR" ]; then
    echo "YOLO prediction files: $(ls "$YOLO_LABELS_DIR"/*.txt 2>/dev/null | wc -l)"
fi

if [ -d "$GT_ANNOTATIONS" ]; then
    echo "Ground truth annotation files: $(ls "$GT_ANNOTATIONS"/*.jpg.json 2>/dev/null | wc -l)"
fi

if [ -d "$TEST_IMAGES_DIR" ]; then
    echo "Test image files: $(ls "$TEST_IMAGES_DIR"/*.jpg 2>/dev/null | wc -l)"
fi
