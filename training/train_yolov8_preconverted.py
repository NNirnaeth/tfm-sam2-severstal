#!/usr/bin/env python3
"""
Train YOLOv8 Segmentation models using pre-converted YOLO format datasets

This script assumes annotations have been pre-converted using convert_to_yolo_format.py
and trains YOLO models on the converted datasets.

Usage:
    # Train on specific subset
    python train_yolov8_preconverted.py --subset 1000 --lr 1e-4
    
    # Train on all subsets with both learning rates
    python train_yolov8_preconverted.py --all-subsets --both-lr
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_yolo_training(lr, subset_name, dataset_yaml, project_dir="new_src/training/training_results/yolo_segmentation", model_size="s"):
    """
    Execute YOLO training command with specified parameters
    
    Args:
        lr (float): Learning rate
        subset_name (str): Name of the subset for naming
        dataset_yaml (str): Path to dataset YAML file
        project_dir (str): Project directory for saving results
        model_size (str): YOLO model size (s, m, l, x)
    """
    
    # Format learning rate for naming
    if lr == 0.001:
        lr_str = "lr1e-3"
    elif lr == 0.0005:
        lr_str = "lr5e-4"
    elif lr == 0.0001:
        lr_str = "lr1e-4"
    else:
        lr_str = f"lr{lr}"
    
    # Construct YOLO command
    cmd = [
        "yolo", "task=segment", "mode=train",
        f"model=yolov8{model_size}-seg.pt",
        f"data={dataset_yaml}",
        "epochs=150", "batch=8", "imgsz=1024", "rect=True", "seed=42",
        "patience=30", "single_cls=True",
        "optimizer=AdamW", "weight_decay=0.0001",
        f"lr0={lr}", "lrf=0.01", "cos_lr=True",
        "mosaic=0.0", "mixup=0.0", "degrees=10.0", "translate=0.10",
        "scale=0.10", "shear=0.0", "fliplr=0.5", "flipud=0.0",
        "hsv_h=0.0", "hsv_s=0.0", "hsv_v=0.0",
        f"project={project_dir}",
        f"name=yolov8{model_size}_seg_{subset_name}_{lr_str}"
    ]
    
    print(f"Executing YOLO training:")
    print(f"  Subset: {subset_name}")
    print(f"  Learning rate: {lr}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        # Execute YOLO command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Training completed successfully for subset {subset_name}, lr={lr}")
        if result.stdout:
            # Print only the last few lines of output to avoid spam
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                print("Training output (last 10 lines):")
                for line in lines[-10:]:
                    print(f"  {line}")
            else:
                print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed for subset {subset_name}, lr={lr}")
        print(f"Error: {e.stderr}")
        return False

def find_dataset_yaml(converted_dir, subset_name):
    """Find the YAML file for a given subset"""
    if subset_name == "full":
        yaml_path = os.path.join(converted_dir, "full_dataset", f"yolo_segmentation_{subset_name}.yaml")
    else:
        yaml_path = os.path.join(converted_dir, f"subset_{subset_name}", f"yolo_segmentation_{subset_name}.yaml")
    
    if os.path.exists(yaml_path):
        return yaml_path
    else:
        print(f"ERROR: Dataset YAML not found at {yaml_path}")
        print("Please run convert_to_yolo_format.py first")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation on pre-converted datasets")
    parser.add_argument("--subset", type=str, choices=["500", "1000", "2000", "full"], 
                       help="Dataset subset to use (500, 1000, 2000, or full)")
    parser.add_argument("--lr", type=float, choices=[0.001, 0.0005, 0.0001], 
                       help="Learning rate to use (1e-3, 5e-4, or 1e-4)")
    parser.add_argument("--both-lr", action="store_true", 
                       help="Train with both lr=5e-4 and lr=1e-4")
    parser.add_argument("--all-subsets", action="store_true",
                       help="Train on all subsets (500, 1000, 2000, full)")
    parser.add_argument("--converted-dir", type=str, default="/home/ptp/sam2/datasets/yolo_segmentation_subsets",
                       help="Directory containing pre-converted YOLO datasets")
    parser.add_argument("--project", type=str, default="new_src/training/training_results/yolo_segmentation",
                       help="Project directory for saving results")
    parser.add_argument("--model-size", type=str, default="s", choices=["s", "m", "l", "x"],
                       help="YOLO model size")
    
    args = parser.parse_args()
    
    # Set default behavior
    if not args.subset and not args.all_subsets:
        print("ERROR: Must specify either --subset or --all-subsets")
        return
    
    if not args.lr and not args.both_lr:
        args.both_lr = True
    
    print("YOLOv8 Segmentation Training (Pre-converted datasets)")
    print("=" * 60)
    print(f"Project directory: {args.project}")
    print(f"Model size: yolov8{args.model_size}-seg")
    print(f"Converted datasets directory: {args.converted_dir}")
    
    # Determine subsets to process
    if args.all_subsets:
        print("Training mode: All subsets (500, 1000, 2000, full)")
        subsets = ["500", "1000", "2000", "full"]
    else:
        print(f"Training mode: Subset {args.subset}")
        subsets = [args.subset]
    
    # Determine learning rates
    if args.both_lr:
        print("Learning rates: 5e-4, 1e-4")
        learning_rates = [0.0005, 0.0001]
    else:
        print(f"Learning rate: {args.lr}")
        learning_rates = [args.lr]
    
    print("-" * 60)
    
    start_time = time.time()
    
    # Verify converted datasets exist
    print("\nVerifying pre-converted datasets...")
    missing_datasets = []
    for subset in subsets:
        yaml_path = find_dataset_yaml(args.converted_dir, subset)
        if yaml_path is None:
            missing_datasets.append(subset)
        else:
            print(f"✓ Found dataset for subset {subset}: {yaml_path}")
    
    if missing_datasets:
        print(f"\nERROR: Missing datasets for subsets: {missing_datasets}")
        print("Please run convert_to_yolo_format.py first to convert these subsets")
        return
    
    print("\nAll datasets found. Starting training...")
    
    # Process each subset
    total_trainings = len(subsets) * len(learning_rates)
    current_training = 0
    
    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"PROCESSING SUBSET: {subset}")
        print(f"{'='*60}")
        
        # Get dataset YAML path
        dataset_yaml = find_dataset_yaml(args.converted_dir, subset)
        
        # Train with each learning rate
        for lr in learning_rates:
            current_training += 1
            print(f"\n{'-'*40}")
            print(f"Training {current_training}/{total_trainings}: {subset} with lr={lr}")
            print(f"{'-'*40}")
            
            success = run_yolo_training(lr, subset, dataset_yaml, args.project, args.model_size)
            if success:
                print(f"✓ Training {current_training}/{total_trainings} completed successfully")
            else:
                print(f"✗ Training {current_training}/{total_trainings} failed")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TOTAL EXECUTION TIME: {total_time/3600:.2f} hours")
    print("All training completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

