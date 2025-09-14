#!/bin/bash

# Script to clean up old result directories and organize everything

echo "🧹 Cleaning up old YOLO+SAM2 result directories..."

BASE_DIR="/home/ptp/sam2/new_src/evaluation/evaluation_results"
MAIN_DIR="$BASE_DIR/yolo_sam2_pipeline"

# Create main directory if it doesn't exist
mkdir -p "$MAIN_DIR/results"
mkdir -p "$MAIN_DIR/incremental"

# Move any existing results to main directory
echo "Moving results from old directories..."

# From yolo_sam2_pipeline_test
if [ -d "$BASE_DIR/yolo_sam2_pipeline_test/results" ]; then
    cp "$BASE_DIR/yolo_sam2_pipeline_test/results"/*.csv "$MAIN_DIR/results/" 2>/dev/null || true
    echo "  Moved results from yolo_sam2_pipeline_test"
fi

# From yolo_sam2_pipeline_full  
if [ -d "$BASE_DIR/yolo_sam2_pipeline_full" ]; then
    cp "$BASE_DIR/yolo_sam2_pipeline_full"/*.log "$MAIN_DIR/" 2>/dev/null || true
    echo "  Moved logs from yolo_sam2_pipeline_full"
fi

# Remove old directories
echo "Removing old directories..."
rm -rf "$BASE_DIR/yolo_sam2_pipeline_test" 2>/dev/null || true
rm -rf "$BASE_DIR/yolo_sam2_pipeline_full" 2>/dev/null || true

echo "✅ Cleanup completed!"
echo "📁 All YOLO+SAM2 results are now in: $MAIN_DIR"
echo ""
echo "Directory structure:"
ls -la "$MAIN_DIR"









