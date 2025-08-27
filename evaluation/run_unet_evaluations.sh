#!/bin/bash

# Script to run individual UNet model evaluations
# This script provides examples of how to run the evaluation for different models

# Set default paths
CHECKPOINTS_DIR="/home/ptp/sam2/new_src/training/results"
TEST_IMG_DIR="/home/ptp/sam2/datasets/Data/splits/test_split/img"
TEST_ANN_DIR="/home/ptp/sam2/datasets/Data/splits/test_split/ann"
OUTPUT_DIR="runs/unet_evaluation"
DEVICE="cuda:0"

# Function to run evaluation
run_evaluation() {
    local model_ckpt=$1
    local arch=$2
    local encoder=$3
    local lr=$4
    
    echo "Evaluating $arch with $encoder encoder, lr=$lr"
    echo "Checkpoint: $model_ckpt"
    echo "Output: $OUTPUT_DIR"
    echo "----------------------------------------"
    
    python eval_unet_full_dataset.py \
        --model_ckpt "$model_ckpt" \
        --arch "$arch" \
        --encoder "$encoder" \
        --img_dir "$TEST_IMG_DIR" \
        --ann_dir "$TEST_ANN_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --threshold 0.5 \
        --save_preds \
        --seed 42
    
    echo "----------------------------------------"
    echo ""
}

# Function to run comprehensive evaluation
run_comprehensive() {
    echo "Running comprehensive evaluation of all UNet models..."
    echo "Checkpoints directory: $CHECKPOINTS_DIR"
    echo "Output directory: $OUTPUT_DIR"
    echo "----------------------------------------"
    
    python eval_all_unet_models.py \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --img_dir "$TEST_IMG_DIR" \
        --ann_dir "$TEST_ANN_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --threshold 0.5 \
        --seed 42
    
    echo "----------------------------------------"
    echo ""
}

# Main execution
echo "UNet Models Evaluation Script"
echo "============================="
echo ""

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Running comprehensive evaluation..."
    echo ""
    run_comprehensive
else
    case $1 in
        "comprehensive"|"all")
            run_comprehensive
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [comprehensive|help]"
            echo ""
            echo "Options:"
            echo "  comprehensive, all  - Run comprehensive evaluation of all models"
            echo "  help, -h, --help   - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 comprehensive                    # Run all models"
            echo "  $0                                 # Run all models (default)"
            echo ""
            echo "For individual model evaluation, use:"
            echo "  python eval_unet_full_dataset.py --model_ckpt /path/to/checkpoint --arch unet --encoder resnet34"
            echo ""
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
fi

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
