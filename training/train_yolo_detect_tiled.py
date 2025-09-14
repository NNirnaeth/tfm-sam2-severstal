#!/usr/bin/env python3
"""
Train YOLOv8 Detection model on tiled Severstal dataset
Training data: tiled dataset with 800x256 tiles
Validation data: tiled validation set

Features:
- Optimized config for tiled data (800x256 native resolution)
- Threshold calibration on validation set
- Both strict and benevolent evaluation metrics
- Early stopping and checkpoint management
- Comprehensive logging and results tracking

This script trains YOLO detection model on tiled dataset
with proper hyperparameter optimization and threshold calibration.
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
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

def train_yolo_model(dataset_yaml, model_size="s", epochs=150, imgsz=896, batch=16, 
                     lr0=0.01, weight_decay=5e-4, patience=30, save_dir="runs/detect",
                     cos_lr=True, augment_level="light"):
    """Train YOLO detection model with optimized parameters for tiled data"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Model selection
    model_map = {
        "s": "yolov8s.pt",
        "m": "yolov8m.pt", 
        "l": "yolov8l.pt",
        "n": "yolov8n.pt"
    }
    model_pt = model_map.get(model_size, "yolov8s.pt")
    
    # Augmentation settings based on level
    if augment_level == "light":
        # Light augmentation to preserve fine geometries
        augment_args = [
            "hsv_h=0.010",      # Slight hue variation
            "hsv_s=0.5",        # Moderate saturation
            "hsv_v=0.3",        # Moderate value/brightness
            "degrees=0.0",      # No rotation (preserves defect orientation)
            "translate=0.1",    # Slight translation
            "scale=0.2",        # Moderate scaling
            "shear=0.0",        # No shear
            "perspective=0.0",  # No perspective
            "flipud=0.0",       # No vertical flip
            "fliplr=0.5",       # Horizontal flip OK for steel defects
            "mosaic=0.0",       # Disable mosaic (can break fine geometries)
            "mixup=0.0"         # Disable mixup
        ]
    else:
        # Standard augmentation
        augment_args = []
    
    # YOLO training command
    cmd = [
        "yolo", "task=detect", "mode=train",
        f"data={dataset_yaml}",
        f"model={model_pt}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"lr0={lr0}",
        f"weight_decay={weight_decay}",
        f"patience={patience}",
        "save=True",
        "save_period=20",  # Save every 20 epochs
        "cache=True",
        "device=0",  # Use GPU
        f"project={save_dir}",
        "name=yolo_detect_tiled",
        "exist_ok=True",
        "single_cls=False",  # Multi-class detection
        "optimizer=AdamW",
        "verbose=True"
    ]
    
    # Add learning rate scheduling
    if cos_lr:
        cmd.append("cos_lr=True")
    
    # Add augmentation parameters
    cmd.extend(augment_args)
    
    # Add mixed precision training
    cmd.append("amp=True")
    
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

def calibrate_thresholds(model_path, dataset_yaml, output_dir, imgsz=896):
    """Calibrate confidence and IoU thresholds on validation set"""
    print("\nCalibrating thresholds on validation set...")
    
    # Threshold ranges to test
    conf_thresholds = np.arange(0.05, 0.75, 0.05)  # 0.05 to 0.70
    iou_thresholds = np.arange(0.50, 0.80, 0.05)   # 0.50 to 0.75
    
    results = []
    
    # Create calibration output directory
    calib_dir = os.path.join(output_dir, "threshold_calibration")
    os.makedirs(calib_dir, exist_ok=True)
    
    print(f"Testing {len(conf_thresholds)} × {len(iou_thresholds)} = {len(conf_thresholds) * len(iou_thresholds)} threshold combinations...")
    
    for conf_thresh in tqdm(conf_thresholds, desc="Confidence thresholds"):
        for iou_thresh in iou_thresholds:
            # Run validation with specific thresholds
            cmd = [
                "yolo", "task=detect", "mode=val",
                f"model={model_path}",
                f"data={dataset_yaml}",
                "split=val",
                f"imgsz={imgsz}",
                f"conf={conf_thresh}",
                f"iou={iou_thresh}",
                "save_json=False",
                "save_txt=False",
                "plots=False",
                "verbose=False"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Parse metrics from output
                metrics = parse_validation_metrics(result.stdout)
                if metrics:
                    metrics.update({
                        'conf_threshold': conf_thresh,
                        'iou_threshold': iou_thresh
                    })
                    results.append(metrics)
                    
            except subprocess.CalledProcessError as e:
                print(f"Warning: Validation failed for conf={conf_thresh}, iou={iou_thresh}")
                continue
    
    if not results:
        print("Warning: No valid calibration results obtained")
        return None
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    # Save all results
    calib_csv = os.path.join(calib_dir, "threshold_calibration_results.csv")
    df_results.to_csv(calib_csv, index=False)
    print(f"Calibration results saved to: {calib_csv}")
    
    # Find optimal thresholds for different objectives
    optimal_thresholds = {}
    
    # 1. Maximize F1 (balanced)
    if 'f1' in df_results.columns:
        best_f1_idx = df_results['f1'].idxmax()
        optimal_thresholds['f1_optimal'] = {
            'conf': df_results.loc[best_f1_idx, 'conf_threshold'],
            'iou': df_results.loc[best_f1_idx, 'iou_threshold'],
            'f1': df_results.loc[best_f1_idx, 'f1'],
            'precision': df_results.loc[best_f1_idx, 'precision'],
            'recall': df_results.loc[best_f1_idx, 'recall'],
            'mAP50': df_results.loc[best_f1_idx, 'mAP50']
        }
    
    # 2. Maximize Recall (for SAM2 pipeline - high recall preferred)
    if 'recall' in df_results.columns:
        # Filter by minimum F1 to avoid too low precision
        min_f1_threshold = 0.3
        high_f1_results = df_results[df_results['f1'] >= min_f1_threshold] if 'f1' in df_results.columns else df_results
        
        if not high_f1_results.empty:
            best_recall_idx = high_f1_results['recall'].idxmax()
            optimal_thresholds['recall_optimal'] = {
                'conf': high_f1_results.loc[best_recall_idx, 'conf_threshold'],
                'iou': high_f1_results.loc[best_recall_idx, 'iou_threshold'],
                'f1': high_f1_results.loc[best_recall_idx, 'f1'],
                'precision': high_f1_results.loc[best_recall_idx, 'precision'],
                'recall': high_f1_results.loc[best_recall_idx, 'recall'],
                'mAP50': high_f1_results.loc[best_recall_idx, 'mAP50']
            }
    
    # 3. Maximize mAP@50
    if 'mAP50' in df_results.columns:
        best_map_idx = df_results['mAP50'].idxmax()
        optimal_thresholds['mAP50_optimal'] = {
            'conf': df_results.loc[best_map_idx, 'conf_threshold'],
            'iou': df_results.loc[best_map_idx, 'iou_threshold'],
            'f1': df_results.loc[best_map_idx, 'f1'],
            'precision': df_results.loc[best_map_idx, 'precision'],
            'recall': df_results.loc[best_map_idx, 'recall'],
            'mAP50': df_results.loc[best_map_idx, 'mAP50']
        }
    
    # Save optimal thresholds
    optimal_json = os.path.join(calib_dir, "optimal_thresholds.json")
    with open(optimal_json, 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    # Print results
    print(f"\nThreshold Calibration Results:")
    for obj_name, thresholds in optimal_thresholds.items():
        print(f"\n{obj_name.upper()}:")
        print(f"  Conf: {thresholds['conf']:.3f}, IoU: {thresholds['iou']:.3f}")
        print(f"  F1: {thresholds['f1']:.4f}, P: {thresholds['precision']:.4f}, R: {thresholds['recall']:.4f}")
        print(f"  mAP@50: {thresholds['mAP50']:.4f}")
    
    return optimal_thresholds, calib_csv

def parse_validation_metrics(stdout_text):
    """Parse YOLO validation metrics from stdout"""
    lines = stdout_text.split('\n')
    metrics = {}
    
    for line in lines:
        # Look for metrics line (usually contains "all" and numerical values)
        if 'all' in line and any(char.isdigit() for char in line):
            # Try to extract metrics - YOLO format varies
            parts = line.strip().split()
            
            # Common YOLO validation output format
            try:
                # Find numeric values
                numeric_parts = []
                for part in parts:
                    try:
                        val = float(part)
                        numeric_parts.append(val)
                    except ValueError:
                        continue
                
                # Typical order: precision, recall, mAP@50, mAP@50-95
                if len(numeric_parts) >= 4:
                    metrics = {
                        'precision': numeric_parts[0],
                        'recall': numeric_parts[1], 
                        'mAP50': numeric_parts[2],
                        'mAP50_95': numeric_parts[3]
                    }
                    
                    # Calculate F1 score
                    if metrics['precision'] > 0 and metrics['recall'] > 0:
                        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                    else:
                        metrics['f1'] = 0.0
                    
                    break
            except (IndexError, ValueError):
                continue
    
    return metrics

def save_training_results(exp_id, model_size, dataset_path, training_dir, optimal_thresholds,
                         epochs, batch_size, lr, weight_decay, save_dir, timestamp):
    """Save training results to CSV following TFM format"""
    
    # Create results directory
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get best validation metrics from training
    results_csv = os.path.join(training_dir, "results.csv")
    val_metrics = {}
    
    if os.path.exists(results_csv):
        try:
            df = pd.read_csv(results_csv)
            if not df.empty:
                # Get best epoch metrics
                best_idx = df['metrics/mAP50(B)'].idxmax() if 'metrics/mAP50(B)' in df.columns else -1
                val_metrics = {
                    'val_mAP50': df.loc[best_idx, 'metrics/mAP50(B)'] if 'metrics/mAP50(B)' in df.columns else 0.0,
                    'val_mAP50_95': df.loc[best_idx, 'metrics/mAP50-95(B)'] if 'metrics/mAP50-95(B)' in df.columns else 0.0,
                    'val_precision': df.loc[best_idx, 'metrics/precision(B)'] if 'metrics/precision(B)' in df.columns else 0.0,
                    'val_recall': df.loc[best_idx, 'metrics/recall(B)'] if 'metrics/recall(B)' in df.columns else 0.0
                }
        except Exception as e:
            print(f"Warning: Could not parse training results: {e}")
    
    # Prepare results for each threshold optimization objective
    results_data = []
    
    for obj_name, thresholds in optimal_thresholds.items():
        result_row = {
            'exp_id': f"{exp_id}_{obj_name}",
            'model': f"yolov8{model_size}",
            'subset_size': 'tiled_full',
            'variant': 'tiled_detect',
            'prompt_type': 'detection',
            'img_size': 896,  # Training image size
            'batch_size': batch_size,
            'steps': epochs,
            'lr': lr,
            'wd': weight_decay,
            'seed': 42,
            'val_mIoU': 0.0,  # Not applicable for detection
            'val_Dice': 0.0,  # Not applicable for detection
            'val_mAP50': val_metrics.get('val_mAP50', 0.0),
            'val_mAP50_95': val_metrics.get('val_mAP50_95', 0.0),
            'val_precision': val_metrics.get('val_precision', 0.0),
            'val_recall': val_metrics.get('val_recall', 0.0),
            'test_mIoU': 0.0,  # Will be filled by evaluation
            'test_Dice': 0.0,  # Will be filled by evaluation
            'IoU@50': 0.0,     # Will be filled by evaluation
            'IoU@75': 0.0,     # Will be filled by evaluation
            'IoU@90': 0.0,     # Will be filled by evaluation
            'IoU@95': 0.0,     # Will be filled by evaluation
            'Precision': thresholds['precision'],
            'Recall': thresholds['recall'],
            'F1': thresholds['f1'],
            'mAP50': thresholds['mAP50'],
            'mAP50_95': 0.0,   # Not always available from calibration
            'ckpt_path': os.path.join(training_dir, "weights", "best.pt"),
            'timestamp': timestamp,
            'conf_threshold': thresholds['conf'],
            'iou_threshold': thresholds['iou'],
            'optimization_objective': obj_name,
            'dataset_path': dataset_path
        }
        
        results_data.append(result_row)
    
    # Save to CSV
    csv_filename = f"{timestamp}_{exp_id}_tiled_yolo_training.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        if results_data:
            fieldnames = results_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_data)
    
    print(f"Training results saved to: {csv_path}")
    return csv_path

def main():
    parser = argparse.ArgumentParser(description="Train YOLO detection on tiled dataset")
    parser.add_argument("--dataset-yaml", type=str, required=True,
                       help="Path to tiled dataset YAML file")
    parser.add_argument("--model-size", type=str, default="s", choices=["n", "s", "m", "l"],
                       help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=896, help="Input image size (use 896 for 800x256 tiles)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--save-dir", type=str, 
                       default="/home/ptp/sam2/new_src/training/training_results/yolo_tiled_detection", 
                       help="Save directory")
    parser.add_argument("--augment-level", type=str, default="light", choices=["light", "standard"],
                       help="Augmentation level")
    parser.add_argument("--skip-calibration", action="store_true",
                       help="Skip threshold calibration (faster but less optimal)")
    parser.add_argument("--exp-id", type=str, default=None,
                       help="Experiment ID (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Check if dataset YAML exists
    if not os.path.exists(args.dataset_yaml):
        print(f"Error: Dataset YAML file not found: {args.dataset_yaml}")
        sys.exit(1)
    
    # Validate dataset structure
    with open(args.dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    dataset_path = dataset_config.get('path', os.path.dirname(args.dataset_yaml))
    train_dir = os.path.join(dataset_path, "images", "train")
    val_dir = os.path.join(dataset_path, "images", "val")
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        sys.exit(1)
    
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found: {val_dir}")
        sys.exit(1)
    
    # Generate experiment ID if not provided
    if args.exp_id is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        args.exp_id = f"exp_yolo{args.model_size}_tiled_detect_{timestamp}"
    
    print(f"Tiled YOLO Detection Training Configuration:")
    print(f"  Dataset YAML: {args.dataset_yaml}")
    print(f"  Model size: YOLOv8{args.model_size}")
    print(f"  Training images: {train_dir}")
    print(f"  Validation images: {val_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr0}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Augmentation: {args.augment_level}")
    print(f"  Experiment ID: {args.exp_id}")
    
    # Start training
    print("\nStarting YOLO detection training...")
    start_time = time.time()
    
    success, output = train_yolo_model(
        dataset_yaml=args.dataset_yaml,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=args.save_dir,
        augment_level=args.augment_level
    )
    
    if not success:
        print("\nTraining failed!")
        print("Output:", output)
        sys.exit(1)
    
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"\nTraining completed successfully in {training_duration:.2f} seconds")
    
    # Find training results directory
    training_dir = os.path.join(args.save_dir, "yolo_detect_tiled")
    if not os.path.exists(training_dir):
        print("Warning: Training directory not found, looking for alternative...")
        # Try to find the most recent training directory
        possible_dirs = [d for d in os.listdir(args.save_dir) if d.startswith("yolo_detect_tiled")]
        if possible_dirs:
            training_dir = os.path.join(args.save_dir, sorted(possible_dirs)[-1])
    
    best_model = os.path.join(training_dir, "weights", "best.pt")
    if not os.path.exists(best_model):
        print(f"Error: Best model not found at {best_model}")
        sys.exit(1)
    
    print(f"Best model saved to: {best_model}")
    
    # Threshold calibration
    optimal_thresholds = {}
    if not args.skip_calibration:
        print("\nStarting threshold calibration...")
        calib_start = time.time()
        
        optimal_thresholds, calib_csv = calibrate_thresholds(
            model_path=best_model,
            dataset_yaml=args.dataset_yaml,
            output_dir=training_dir,
            imgsz=args.imgsz
        )
        
        calib_end = time.time()
        calib_duration = calib_end - calib_start
        print(f"Threshold calibration completed in {calib_duration:.2f} seconds")
        
        if optimal_thresholds:
            print(f"Calibration results saved to: {calib_csv}")
        else:
            print("Warning: Threshold calibration failed")
    else:
        print("Skipping threshold calibration (--skip-calibration)")
        # Use default thresholds
        optimal_thresholds = {
            'default': {
                'conf': 0.25,
                'iou': 0.70,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0
            }
        }
    
    # Save training results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_path = save_training_results(
        exp_id=args.exp_id,
        model_size=args.model_size,
        dataset_path=args.dataset_yaml,
        training_dir=training_dir,
        optimal_thresholds=optimal_thresholds,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr0,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        timestamp=timestamp
    )
    
    total_duration = time.time() - start_time
    print(f"\nTraining and calibration completed successfully!")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"Results saved to: {csv_path}")
    print(f"Best model: {best_model}")
    print(f"Ready for evaluation!")

if __name__ == "__main__":
    main()




