#!/usr/bin/env python3
"""
Train YOLO-Detect on Severstal dataset
Training data: data/severstal_det/images/train (from prepared dataset)
Validation data: data/severstal_det/images/val (from prepared dataset)

Object detection: defect instances as bounding boxes
Image size: 1024x1024 (YOLO standard)
Loss: YOLO detection loss (classification + regression)

This script uses modular components:
- YOLOv8 model for detection
- Pre-prepared Severstal detection dataset

TFM Requirements:
- AdamW with weight_decay=1e-4
- Compare lr = 1e-3 vs 1e-4
- YOLO detection loss
- Save best model by val_mAP
- Comprehensive metrics logging
- 3 seeds for stability
- Mixed precision, gradient clipping, stability
- Comprehensive logging and metrics
"""

import os
import sys
import numpy as np
import torch
import argparse
import subprocess
import csv
from datetime import datetime
import random

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_yolo_training(lr, weight_decay, batch_size, epochs, patience, img_size, seed, save_dir):
    """Run YOLO training with specified parameters"""
    
    # Create project and run names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    project_name = "new_src/training/training_results/yolo_detection"
    run_name = f"yolov8s_det_lr{lr:.0e}_seed{seed}"
    
    # Get absolute path to dataset YAML
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dataset_yaml = os.path.join(project_root, "datasets", "yolo_detection", "yolo_detection.yaml")
    
    # YOLO training command
    cmd = [
        "yolo", "task=detect", "mode=train",
        "model=yolov8s.pt",
        f"data={dataset_yaml}",  # Use absolute path
        f"epochs={epochs}",
        f"batch={batch_size}",
        f"imgsz={img_size}",
        "rect=True",
        f"seed={seed}",
        f"patience={patience}",
        "single_cls=True",
        "optimizer=AdamW",
        f"weight_decay={weight_decay}",
        f"lr0={lr}",
        "lrf=0.01",
        "cos_lr=True",  # Better learning rate schedule
        "mosaic=0",
        "mixup=0",
        "degrees=0",  # No rotation for detection (better for elongated defects)
        "translate=0.1",
        "scale=0.1",
        "shear=0.0",
        "fliplr=0.5",
        "flipud=0.0",
        "hsv_h=0.0",
        "hsv_s=0.0",
        "hsv_v=0.0",
        f"project={project_name}",
        f"name={run_name}"
    ]
    
    print(f"Running YOLO training command:")
    print(" ".join(cmd))
    
    # Run training
    try:
        # Don't capture output to avoid buffer overflow with long training
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return True, "Training completed"
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False, f"Training failed with return code {e.returncode}"


def parse_yolo_results(run_path):
    """Parse YOLO training results from the run directory - find best epoch by mAP50"""
    results_file = os.path.join(run_path, "results.csv")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    # Read all results and find best epoch by mAP50
    best_metrics = None
    best_mAP50 = -1
    
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mAP50 = float(row.get('metrics/mAP50(B)', 0))
                if mAP50 > best_mAP50:
                    best_mAP50 = mAP50
                    best_metrics = row
            except (ValueError, KeyError):
                continue
    
    if best_metrics is None:
        print("Could not find valid metrics in results.csv")
        return None
    
    print(f"Best epoch found with mAP50: {best_mAP50:.4f}")
    return best_metrics


def save_experiment_results(exp_id, model, subset_size, variant, prompt_type, img_size, 
                           batch_size, epochs, lr, wd, seed, val_metrics, ckpt_path, 
                           save_dir, timestamp):
    """Save experiment results to CSV following TFM format"""
    
    # Create results directory
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare results data with correct YOLO detection metrics
    results_data = {
        'exp_id': exp_id,
        'model': model,
        'subset_size': subset_size,
        'variant': variant,
        'prompt_type': prompt_type,
        'img_size': img_size,
        'batch_size': batch_size,
        'steps': epochs,  # YOLO uses epochs, not steps
        'lr': lr,
        'wd': wd,
        'seed': seed,
        'val_mIoU': val_metrics.get('metrics/mAP50(B)', 0.0),
        'val_Dice': val_metrics.get('metrics/mAP50-95(B)', 0.0),
        'test_mIoU': 0.0,  # Will be filled later
        'test_Dice': 0.0,  # Will be filled later
        'IoU@50': val_metrics.get('metrics/mAP50(B)', 0.0),
        'IoU@75': 0.0,  # Not directly available in YOLO
        'IoU@90': 0.0,  # Not directly available in YOLO
        'IoU@95': 0.0,  # Not directly available in YOLO
        'Precision': val_metrics.get('metrics/precision(B)', 0.0),
        'Recall': val_metrics.get('metrics/recall(B)', 0.0),
        'F1': 0.0,  # Will calculate if possible
        'ckpt_path': ckpt_path,
        'timestamp': timestamp
    }
    
    # Calculate F1 if precision and recall are available
    try:
        precision_val = float(results_data['Precision'])
        recall_val = float(results_data['Recall'])
        if precision_val > 0 and recall_val > 0:
            results_data['F1'] = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    except (ValueError, TypeError):
        results_data['F1'] = 0.0
    
    # Save to CSV
    csv_filename = f"{timestamp}_{model}_{variant}_{lr}_{seed}.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results_data)
    
    print(f"Results saved to: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='Train YOLO-Detect on Severstal dataset (requires pre-prepared dataset)')
    parser.add_argument('--lr', type=float, default=0.0005,
                       choices=[1e-3, 1e-4, 0.0005],
                       help='Learning rate (1e-3, 1e-4, or 0.0005)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--img_size', type=int, default=1024,
                       help='Image size (square)')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, 
                       default='new_src/training/training_results/yolo_detection',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check if YOLO is available
    try:
        subprocess.run(["yolo", "--version"], capture_output=True, check=True)
        print("YOLO is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: YOLO command not found. Please install ultralytics: pip install ultralytics")
        return
    
    # Check if dataset YAML exists
    # Use absolute path to work from any directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to project root
    dataset_yaml = os.path.join(project_root, "datasets", "yolo_detection", "yolo_detection.yaml")
    
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset YAML file not found: {dataset_yaml}")
        print("Please run the dataset preparation script first:")
        print("python new_src/experiments/prepare_yolo_detect_dataset.py")
        return
    
    # Check if dataset directories exist
    required_dirs = [
        os.path.join(project_root, "datasets", "yolo_detection", "images", "train_split"),
        os.path.join(project_root, "datasets", "yolo_detection", "images", "val_split"),
        os.path.join(project_root, "datasets", "yolo_detection", "labels", "train_split"),
        os.path.join(project_root, "datasets", "yolo_detection", "labels", "val_split")
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    if missing_dirs:
        print(f"Error: Missing dataset directories: {missing_dirs}")
        print("Please run the dataset preparation script first:")
        print("python new_src/experiments/prepare_yolo_detect_dataset.py")
        return
    
    # Print training configuration
    print(f"\nYOLO Detection Training Configuration:")
    print(f"  Model: YOLOv8s")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Image size: {args.img_size}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Seed: {args.seed}")
    print(f"  Dataset: {dataset_yaml}")
    
    # Run training
    print(f"\nStarting YOLO training...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    success, output = run_yolo_training(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        img_size=args.img_size,
        seed=args.seed,
        save_dir=args.save_dir
    )
    
    if not success:
        print("Training failed. Exiting.")
        return
    
    # Find the run directory
    project_dir = "new_src/training/training_results/yolo_detection"
    run_name = f"yolov8s_det_lr{args.lr:.0e}_seed{args.seed}"
    run_path = os.path.join(project_dir, run_name)
    
    if not os.path.exists(run_path):
        print(f"Run directory not found: {run_path}")
        return
    
    # Parse results
    print(f"\nParsing training results from: {run_path}")
    val_metrics = parse_yolo_results(run_path)
    
    if val_metrics is None:
        print("Could not parse validation metrics")
        return
    
    # Find best checkpoint
    weights_dir = os.path.join(run_path, "weights")
    if os.path.exists(weights_dir):
        best_ckpt = os.path.join(weights_dir, "best.pt")
        last_ckpt = os.path.join(weights_dir, "last.pt")
        
        if os.path.exists(best_ckpt):
            ckpt_path = best_ckpt
        elif os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
        else:
            ckpt_path = "Not found"
    else:
        ckpt_path = "Not found"
    
    # Save experiment results
    exp_id = f"exp_yolo_detect_full_dataset_{args.lr}_{timestamp}"
    
    csv_path = save_experiment_results(
        exp_id=exp_id,
        model="yolov8s",
        subset_size="full_dataset",
        variant=f"lr{args.lr:.0e}",
        prompt_type="detection",
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.weight_decay,
        seed=args.seed,
        val_metrics=val_metrics,
        ckpt_path=ckpt_path,
        save_dir=args.save_dir,
        timestamp=timestamp
    )
    
    # Print final results
    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {csv_path}")
    print(f"Best checkpoint: {ckpt_path}")
    
    if val_metrics:
        print(f"Best validation metrics:")
        for key, value in val_metrics.items():
            if 'mAP' in key or 'precision' in key or 'recall' in key:
                try:
                    print(f"  {key}: {float(value):.4f}")
                except:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

