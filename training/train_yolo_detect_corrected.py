#!/usr/bin/env python3
"""
Train YOLOv8 Detection model on corrected Severstal dataset
Training data: datasets/yolo_detection_fixed/images/train (full dataset)
           or: datasets/yolo_detection_fixed/subset_XXX/images/train (subset)
Validation data: datasets/yolo_detection_fixed/images/val (full dataset)
             or: datasets/yolo_detection_fixed/subset_XXX/images/val (subset)

Object detection: defect instances as bounding boxes
Image size: 1024x256 (maintaining aspect ratio)

This script trains YOLO detection model on the corrected dataset
with proper coordinate alignment between GT and predictions.
Supports both full dataset and subset training.
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
import tempfile
from datetime import datetime
import time
from pathlib import Path
import yaml

def train_yolo_model(dataset_yaml, epochs=100, imgsz=1024, batch=16, lr=0.01, 
                     weight_decay=1e-4, patience=30, save_dir="runs/detect", subset_size=None):
    """Train YOLO detection model with specified parameters"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate experiment name based on subset size
    if subset_size is not None:
        exp_name = f"yolo_detect_corrected_subset_{subset_size}"
    else:
        exp_name = "yolo_detect_corrected_full"
    
    # YOLO training command
    cmd = [
        "yolo", "task=detect", "mode=train",
        f"data={dataset_yaml}",
        "model=yolov8s.pt",  # Start with small model
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"lr0={lr}",
        f"weight_decay={weight_decay}",
        "patience=30",
        "save=True",
        "save_period=10",
        "cache=True",
        "device=0",  # Use GPU
        f"project={save_dir}",
        f"name={exp_name}",
        "exist_ok=True",
        "cos_lr=True",  # Cosine learning rate scheduling
        "amp=True",     # Mixed precision
        "overlap_mask=True",
        "mask_ratio=4",
        "single_cls=True",  # Single class (defect)
        "optimizer=AdamW"
    ]
    
    print(f"Training YOLO detection model with command:")
    print(" ".join(cmd))
    
    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Training completed successfully!")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr

def main():
    parser = argparse.ArgumentParser(description="Train YOLO detection on corrected dataset")
    parser.add_argument("--dataset-yaml", type=str, default=None,
                       help="Path to dataset YAML file (auto-detected if not provided)")
    parser.add_argument("--subset-size", type=int, default=None,
                       help="Subset size for training (e.g., 500, 1000, 2000). Uses full dataset if not specified.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--save-dir", type=str, default="/home/ptp/sam2/new_src/training/training_results/yolo_detection", help="Save directory")
    
    args = parser.parse_args()
    
    # Auto-detect dataset YAML if not provided
    if args.dataset_yaml is None:
        if args.subset_size is not None:
            # Use subset dataset
            args.dataset_yaml = f"datasets/yolo_detection_fixed/subset_{args.subset_size}/yolo_detection_{args.subset_size}.yaml"
        else:
            # Use full dataset
            args.dataset_yaml = "datasets/yolo_detection_fixed/yolo_detection.yaml"
    
    # Check if dataset YAML exists
    if not os.path.exists(args.dataset_yaml):
        print(f"Error: Dataset YAML file not found: {args.dataset_yaml}")
        sys.exit(1)
    
    # Check if dataset directories exist
    dataset_dir = os.path.dirname(args.dataset_yaml)
    train_dir = os.path.join(dataset_dir, "images", "train_split")
    val_dir = os.path.join(dataset_dir, "images", "val_split")
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        sys.exit(1)
    
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found: {val_dir}")
        sys.exit(1)
    
    print(f"Dataset YAML: {args.dataset_yaml}")
    print(f"Training images: {train_dir}")
    print(f"Validation images: {val_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    if args.subset_size:
        print(f"Subset size: {args.subset_size}")
    else:
        print(f"Using full dataset")
    
    # Start training
    print("\nStarting YOLO detection training...")
    start_time = time.time()
    
    success, output = train_yolo_model(
        dataset_yaml=args.dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=args.save_dir,
        subset_size=args.subset_size
    )
    
    if success:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nTraining completed successfully in {duration:.2f} seconds")
        
        # Generate experiment name for output path
        if args.subset_size is not None:
            exp_name = f"yolo_detect_corrected_subset_{args.subset_size}"
        else:
            exp_name = "yolo_detect_corrected_full"
        
        print(f"Results saved to: {args.save_dir}/{exp_name}")
        
        # Find best model
        best_model = os.path.join(args.save_dir, exp_name, "weights", "best.pt")
        if os.path.exists(best_model):
            print(f"Best model saved to: {best_model}")
        else:
            print("Warning: Best model not found")
    else:
        print("\nTraining failed!")
        print("Output:", output)
        sys.exit(1)

if __name__ == "__main__":
    main()
