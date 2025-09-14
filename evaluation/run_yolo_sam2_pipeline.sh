#!/bin/bash

# YOLO+SAM2 Pipeline Evaluation - MAIN SCRIPT
# Optimized for full dataset (2000 images) with memory management
# This is the ONLY script you need to run the YOLO+SAM2 pipeline

echo "🚀 YOLO+SAM2 Pipeline Evaluation (Full Dataset)"
echo "==============================================="

# SAM2 Model Options (define paths first)
SAM2_FINETUNED_LARGE="/home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_large_full_dataset_lr0.0001_20250829_1646/best_step11000_iou0.5124_dice0.6461.torch"
SAM2_BASE_LARGE="/home/ptp/sam2/models/base/sam2/sam2_hiera_large.pt"
SAM2_BASE_SMALL="/home/ptp/sam2/models/base/sam2/sam2_hiera_small.pt"

# Parse command line arguments
MODEL_TYPE="finetuned_large"  # Default

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--model MODEL_TYPE]"
            echo ""
            echo "Available models:"
            echo "  finetuned_large  - SAM2 Large fine-tuned on Severstal (default)"
            echo "  base_large       - SAM2 Large base model"
            echo "  base_small       - SAM2 Small base model"
            echo ""
            echo "Examples:"
            echo "  $0                           # Use fine-tuned large (default)"
            echo "  $0 --model base_large        # Use base large"
            echo "  $0 --model base_small        # Use base small"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set model configuration based on argument
case $MODEL_TYPE in
    "finetuned_large")
        SAM2_CHECKPOINT="${SAM2_FINETUNED_LARGE}"
        SAM2_MODEL_TYPE="sam2_hiera_l"
        MODEL_DESC="SAM2 Large Fine-tuned"
        ;;
    "base_large")
        SAM2_CHECKPOINT="${SAM2_BASE_LARGE}"
        SAM2_MODEL_TYPE="sam2_hiera_l"
        MODEL_DESC="SAM2 Large Base"
        ;;
    "base_small")
        SAM2_CHECKPOINT="${SAM2_BASE_SMALL}"
        SAM2_MODEL_TYPE="sam2_hiera_s"
        MODEL_DESC="SAM2 Small Base"
        ;;
    *)
        echo "❌ Error: Unknown model type '$MODEL_TYPE'"
        echo "Available options: finetuned_large, base_large, base_small"
        exit 1
        ;;
esac

echo "Selected model: $MODEL_DESC"
echo "Checkpoint: $SAM2_CHECKPOINT"
echo "Model type: $SAM2_MODEL_TYPE"
echo ""

# Configuration
YOLO_LABELS_DIR="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection/predict_test_corrected/labels"

GT_ANNOTATIONS="/home/ptp/sam2/datasets/Data/splits/test_split/ann"
TEST_IMAGES_DIR="/home/ptp/sam2/datasets/yolo_detection_fixed/images/test_split"
OUTPUT_DIR="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_sam2_pipeline"
MEMORY_LOG="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_sam2_pipeline/memory_usage.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Model: $MODEL_DESC"
echo "  YOLO labels: $YOLO_LABELS_DIR"
echo "  SAM2 checkpoint: $SAM2_CHECKPOINT"
echo "  SAM2 model type: $SAM2_MODEL_TYPE"
echo "  Ground truth: $GT_ANNOTATIONS"
echo "  Test images: $TEST_IMAGES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Memory log: $MEMORY_LOG"
echo ""

# Check if required files exist
echo "Checking required files..."
if [ ! -d "$YOLO_LABELS_DIR" ]; then
    echo "❌ Error: YOLO labels directory not found: $YOLO_LABELS_DIR"
    exit 1
fi

if [ ! -f "$SAM2_CHECKPOINT" ]; then
    echo "❌ Error: SAM2 checkpoint not found: $SAM2_CHECKPOINT"
    exit 1
fi

if [ ! -d "$GT_ANNOTATIONS" ]; then
    echo "❌ Error: Ground truth annotations directory not found: $GT_ANNOTATIONS"
    exit 1
fi

if [ ! -d "$TEST_IMAGES_DIR" ]; then
    echo "❌ Error: Test images directory not found: $TEST_IMAGES_DIR"
    exit 1
fi

echo "✅ All required files found"
echo ""

# Count images
TOTAL_IMAGES=$(find "$TEST_IMAGES_DIR" -name "*.jpg" | wc -l)
echo "📊 Dataset info:"
echo "  Total images: $TOTAL_IMAGES"
echo "  Batch size: 25 (to avoid OOM)"
echo "  Estimated batches: $(( (TOTAL_IMAGES + 24) / 25 ))"
echo ""

# Start memory monitoring in background
echo "🔍 Starting memory monitoring..."
python new_src/evaluation/monitor_memory.py --interval 10 --log "$MEMORY_LOG" &
MONITOR_PID=$!

# Wait a moment for monitoring to start
sleep 2

echo "🎯 Starting YOLO+SAM2 pipeline evaluation..."
echo "   (Memory monitoring running in background, PID: $MONITOR_PID)"
echo ""

# Run YOLO+SAM2 pipeline evaluation with memory optimization
python new_src/evaluation/eval_yolo+sam2_full_dataset.py \
    --yolo_labels_dir "$YOLO_LABELS_DIR" \
    --sam2_checkpoint "$SAM2_CHECKPOINT" \
    --sam2_model_type "sam2_hiera_l" \
    --gt_annotations "$GT_ANNOTATIONS" \
    --test_images_dir "$TEST_IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --conf_threshold 0.25 \
    --iou_threshold 0.70 \
    --max_detections 300 \
    --mask_threshold 0.5 \
    --fill_holes \
    --morphology \
    --iou_merge 0.85 \
    --topk_boxes 20 \
    --min_w_px 8 \
    --pad_px 6 \
    --opt_threshold 0.5 \
    --batch_size 25 \
    --save_batch_results

# Capture exit code
EXIT_CODE=$?

# Stop memory monitoring
echo ""
echo "🛑 Stopping memory monitoring..."
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

# Check results
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ YOLO+SAM2 pipeline evaluation completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo "📊 Memory usage log: $MEMORY_LOG"
    
    # Show final results
    if [ -d "$OUTPUT_DIR/results" ]; then
        echo ""
        echo "📈 Final Results:"
        ls -la "$OUTPUT_DIR/results/"*.csv | tail -1 | while read line; do
            echo "  Latest results: $(basename $(echo $line | awk '{print $NF}'))"
        done
    fi
else
    echo ""
    echo "❌ YOLO+SAM2 pipeline evaluation failed with exit code: $EXIT_CODE"
    echo "📊 Check memory usage log: $MEMORY_LOG"
fi

echo ""
echo "🏁 Evaluation completed!"
