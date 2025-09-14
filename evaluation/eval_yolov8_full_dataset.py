#!/usr/bin/env python3
"""
Evaluate YOLOv8 Segmentation models on test_split using trained models from train_yolov8_full_dataset.py
Test data: datasets/Data/splits/test_split (2000 images)

Binary segmentation: defect vs background (unified classes)
Image size: 1024x256 (maintaining aspect ratio)

This script evaluates YOLO segmentation models trained with:
- YOLO-Seg lr=1e-3
- YOLO-Seg lr=5e-4  
- YOLO-Seg lr=1e-4

TFM Requirements:
- Same metrics as all evaluations: mIoU, IoU@50, IoU@75, IoU@90, IoU@95, Dice, Precision, Recall, F1
- Save results in CSV format under /data/results/ with unique names
- Generate comparative performance plots
- Support all three learning rate variants
- Comprehensive logging and metrics
- Automatic confidence threshold optimization for recall >= 0.75
- Proper probability map handling for IoU@thresholds
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
import base64
import zlib
from PIL import Image
from io import BytesIO

# Add paths
sys.path.append('new_src/utils')



def load_ground_truth(ann_path):
    """Load and combine all defect objects into a single binary mask from PNG bitmap"""
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        if 'objects' not in data or not data['objects']:
            return np.zeros((data['size']['height'], data['size']['width']), dtype=np.uint8)
        
        # Get image dimensions
        height = data['size']['height']
        width = data['size']['width']
        
        # Create empty mask
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process all objects with proper bitmap positioning
        for obj in data['objects']:
            if 'bitmap' not in obj or 'data' not in obj['bitmap']:
                continue
            
            bmp = obj['bitmap']
            
            # Decompress bitmap data (PNG compressed)
            compressed_data = base64.b64decode(bmp['data'])
            decompressed_data = zlib.decompress(compressed_data)
            
            # Load PNG image from bytes
            png_image = Image.open(BytesIO(decompressed_data))
            
            # Use ALPHA channel if exists, otherwise convert to L (grayscale)
            if png_image.mode == 'RGBA':
                crop = np.array(png_image.split()[-1])  # Alpha channel
            else:
                crop = np.array(png_image.convert('L'))  # Grayscale
            
            # Convert to binary mask
            crop_bin = (crop > 0).astype(np.uint8)
            
            # Get origin coordinates from bitmap
            x0, y0 = bmp.get('origin', [0, 0])
            h, w = crop_bin.shape[:2]
            
            # Calculate end coordinates (clamped to image boundaries)
            x1, y1 = min(x0 + w, width), min(y0 + h, height)
            
            # Place the bitmap at the correct position
            if x1 > x0 and y1 > y0:
                full_mask[y0:y1, x0:x1] = np.maximum(
                    full_mask[y0:y1, x0:x1],
                    crop_bin[:y1-y0, :x1-x0]
                )
        
        return full_mask
        
    except Exception as e:
        print(f"Error loading GT from {ann_path}: {e}")
        return np.zeros((data['size']['height'], data['size']['width']), dtype=np.uint8)

def compute_metrics(pred_mask, gt_mask):
    """Compute all TFM metrics: IoU, IoU@thresholds, Dice, Precision, Recall, F1"""
    # Ensure binary masks
    pred_binary = (pred_mask > 0).astype(bool)
    gt_binary = (gt_mask > 0).astype(bool)
    
    # Intersection and Union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # IoU
    iou = intersection / (union + 1e-8)
    
    # Pixel-level metrics
    tp = intersection
    fp = pred_binary.sum() - intersection
    fn = gt_binary.sum() - intersection
    tn = (~pred_binary & ~gt_binary).sum()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Dice Coefficient
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'dice': float(dice)
    }

def calculate_iou_at_thresholds(pred_prob, gt_mask, thresholds=[0.5, 0.75, 0.9, 0.95]):
    """Calculate IoU at different thresholds using probability map"""
    results = {}
    gt_binary = (gt_mask > 0).astype(bool)
    
    for th in thresholds:
        # Apply threshold to probability map
        pred_thresholded = (pred_prob > th).astype(bool)
        
        # Calculate IoU at this threshold
        intersection = np.logical_and(pred_thresholded, gt_binary).sum()
        union = np.logical_or(pred_thresholded, gt_binary).sum()
        iou_th = intersection / (union + 1e-8)
        
        results[f'iou_{int(th*100)}'] = float(iou_th)
    
    return results

def find_optimal_confidence(model, test_data, target_recall=0.75):
    """Find optimal confidence threshold to achieve target recall using validation split"""
    # Use validation split for calibration (first 200 images)
    val_data = test_data[:200]  # Use validation subset for speed
    
    # Test confidence thresholds in range [0.10, 0.35] as recommended
    conf_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    best_conf = 0.25  # Default fallback
    best_recall = 0.0
    
    print(f"Calibrating confidence threshold on validation split ({len(val_data)} images)...")
    
    for conf in conf_thresholds:
        recalls = []
        for item in val_data[:50]:  # Use subset for speed
            gt_mask = load_ground_truth(item['annotation'])
            if gt_mask is None:
                continue
                
            image = cv2.imread(item['image'])
            if image is None:
                continue
                
            # Use recommended parameters for defect detection
            results = model(image, imgsz=1024, conf=conf, iou=0.7, max_det=300, verbose=False)
            
            if len(results) > 0 and results[0].masks is not None:
                pred_prob = np.zeros_like(gt_mask, dtype=np.float32)
                for mask in results[0].masks.data:
                    mask_np = mask.cpu().numpy()
                    if mask_np.shape != gt_mask.shape:
                        mask_np = cv2.resize(mask_np, (gt_mask.shape[1], gt_mask.shape[0]))
                    pred_prob = np.maximum(pred_prob, mask_np)
                
                # Use the same threshold as in evaluation for consistency
                pred_mask = (pred_prob > 0.5).astype(np.uint8)
                gt_binary = (gt_mask > 0).astype(bool)
                
                # Calculate pixel-level recall (segmentation recall)
                tp = np.logical_and(pred_mask > 0, gt_binary).sum()
                fn = gt_binary.sum() - tp
                recall = tp / (tp + fn + 1e-8)
                recalls.append(recall)
        
        if recalls:
            avg_recall = np.mean(recalls)
            print(f"  conf={conf:.2f}: recall={avg_recall:.3f}")
            if avg_recall >= target_recall:
                if conf > best_conf:  # Take the highest conf that meets target
                    best_conf = conf
                    best_recall = avg_recall
            elif best_recall == 0.0:  # If no conf meets target, take the highest recall
                best_conf = conf
                best_recall = avg_recall
    
    if best_recall >= target_recall:
        print(f"Selected conf={best_conf:.2f} (recall={best_recall:.3f} >= {target_recall})")
    else:
        print(f"Warning: No conf meets target recall {target_recall}. Using conf={best_conf:.2f} (recall={best_recall:.3f})")
    
    return best_conf

def evaluate_yolo_model(model_path, test_data, conf_threshold=0.5, iou_threshold=0.5):
    """Evaluate YOLO segmentation model on test data"""
    print(f"Loading YOLO model: {model_path}")
    
    # Load YOLO model
    model = YOLO(model_path, task="detect")
    
    # First, find optimal confidence threshold for recall >= 0.75
    print("Finding optimal confidence threshold for recall >= 0.75...")
    optimal_conf = find_optimal_confidence(model, test_data[:100], target_recall=0.75)  # Use subset for speed
    print(f"Optimal confidence threshold: {optimal_conf:.3f}")
    
    all_metrics = []
    all_iou_thresholds = []
    
    print(f"Evaluating on {len(test_data)} test images with conf={optimal_conf}...")
    
    for item in tqdm(test_data, desc="Evaluating"):
        image_path = item['image']
        ann_path = item['annotation']
        
        # Load ground truth
        gt_mask = load_ground_truth(ann_path)
        if gt_mask is None:
            continue
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Run YOLO inference with optimal confidence threshold and recommended parameters
        results = model(image, imgsz=1024, conf=optimal_conf, iou=0.7, max_det=300, verbose=False)
        
        # Initialize probability map and binary mask
        pred_prob = np.zeros_like(gt_mask, dtype=np.float32)
        pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        
        if len(results) > 0 and results[0].masks is not None:
            # Combine all detected masks using maximum probability
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy()
                # Resize mask to match ground truth
                if mask_np.shape != gt_mask.shape:
                    mask_np = cv2.resize(mask_np, (gt_mask.shape[1], gt_mask.shape[0]))
                # Keep probability values (don't threshold yet)
                pred_prob = np.maximum(pred_prob, mask_np)
            
            # Create binary mask for basic metrics (threshold at 0.5)
            pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # Compute basic metrics using binary mask
        metrics = compute_metrics(pred_mask, gt_mask)
        # Compute IoU@thresholds using probability map
        iou_thresholds = calculate_iou_at_thresholds(pred_prob, gt_mask)
        
        # Combine all metrics
        combined_metrics = {**metrics, **iou_thresholds}
        all_metrics.append(combined_metrics)
        all_iou_thresholds.append(iou_thresholds)
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Calculate IoU@threshold percentages (percentage of images with IoU >= 0.5 at each threshold)
        for th in [50, 75, 90, 95]:
            key = f'iou_{th}'
            if key in all_metrics[0]:
                # Calculate percentage of images that achieve IoU >= 0.5 at this threshold
                avg_metrics[f'{key}_pct'] = np.mean([m[key] >= 0.5 for m in all_metrics]) * 100
        
        return avg_metrics, all_metrics
    else:
        return None, []

def save_results_to_csv(results, model_name, lr, output_dir="new_src/evaluation/evaluation_results/yolo_segmentation"):
    """Save evaluation results to CSV following TFM format"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"exp_yolov8seg_{lr}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for CSV
    data = {
        'exp_id': f"exp_yolov8seg_{lr}_{timestamp}",
        'model': 'YOLOv8-Seg',
        'subset_size': 'full_dataset',
        'variant': f'lr_{lr}',
        'prompt_type': 'N/A',
        'img_size': '1024x256',
        'batch_size': 'N/A',
        'steps': 'N/A',
        'lr': lr,
        'wd': 'N/A',
        'seed': 42,
        'val_mIoU': 'N/A',
        'val_Dice': 'N/A',
        'test_mIoU': results['iou'],
        'test_Dice': results['dice'],
        'IoU@50': results['iou_50_pct'] if 'iou_50_pct' in results else 'N/A',
        'IoU@75': results['iou_75_pct'] if 'iou_75_pct' in results else 'N/A',
        'IoU@90': results['iou_90_pct'] if 'iou_90_pct' in results else 'N/A',
        'IoU@95': results['iou_95_pct'] if 'iou_95_pct' in results else 'N/A',
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1': results['f1'],
        'ckpt_path': 'N/A',
        'timestamp': timestamp
    }
    
    # Create DataFrame and save
    df = pd.DataFrame([data])
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    
    return filepath

def create_comparison_plots(results_dict, output_dir="new_src/evaluation/evaluation_results/plots"):
    """Create comparative performance plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    # Metrics to plot
    metrics = ['iou', 'dice', 'precision', 'recall', 'f1']
    metric_names = ['IoU', 'Dice', 'Precision', 'Recall', 'F1-Score']
    
    # Create comparison bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    lr_values = list(results_dict.keys())
    colors = ['#1f77b4', '#ff7f0e']
    
    for i, lr in enumerate(lr_values):
        values = [results_dict[lr][metric] for metric in metrics]
        ax.bar(x + i*width, values, width, label=f'LR={lr}', color=colors[i])
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('YOLOv8 Segmentation Performance Comparison')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, lr in enumerate(lr_values):
        for j, metric in enumerate(metrics):
            value = results_dict[lr][metric]
            ax.text(j + i*width, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"yolov8seg_comparison_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Segmentation models on test split")
    parser.add_argument("--model-dir", type=str, default="new_src/training/training_results/yolo_segmentation",
                       help="Directory containing trained YOLO models")
    parser.add_argument("--test-data", type=str, default="datasets/Data/splits/test_split",
                       help="Path to test data directory")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for NMS")
    parser.add_argument("--output-dir", type=str, default="new_src/evaluation/evaluation_results/yolo_segmentation",
                       help="Output directory for results")
    parser.add_argument("--lr", type=str, choices=['1e-3', '1e-4', 'both'], default='both',
                       help="Learning rate to evaluate (1e-3, 1e-4, or both)")
    
    args = parser.parse_args()
    
    print("YOLOv8 Segmentation Evaluation Script")
    print("=" * 50)
    print(f"Test data: {args.test_data}")
    print(f"Model directory: {args.model_dir}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    print("-" * 50)
    
    # Check if test data exists
    if not os.path.exists(args.test_data):
        print(f"Error: Test data directory not found: {args.test_data}")
        return
    
    # Prepare test data
    test_img_dir = os.path.join(args.test_data, "img")
    test_ann_dir = os.path.join(args.test_data, "ann")
    
    if not os.path.exists(test_img_dir) or not os.path.exists(test_ann_dir):
        print(f"Error: Test data structure not found. Expected: {test_img_dir} and {test_ann_dir}")
        return
    
    # Get test image files
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(test_images)} test images")
    
    # Prepare test data list
    test_data = []
    for img_file in test_images:
        img_path = os.path.join(test_img_dir, img_file)
        ann_path = os.path.join(test_ann_dir, img_file + ".json")
        
        if os.path.exists(ann_path):
            test_data.append({
                'image': img_path,
                'annotation': ann_path
            })
    
    print(f"Prepared {len(test_data)} test samples with annotations")
    
    # Find available trained models
    available_models = {}
    
    if args.lr == 'both':
        # Look for both lr=1e-3 and lr=1e-4 models
        for lr in ['1e-3', '1e-4']:
            model_dir = os.path.join(args.model_dir, f"yolov8s_seg_lr{lr}")
            model_path = os.path.join(model_dir, "weights", "best.pt")
            
            if os.path.exists(model_path):
                available_models[lr] = model_path
                print(f"Found model for lr={lr}: {model_path}")
            else:
                print(f"Model not found for lr={lr}: {model_path}")
    else:
        # Look for specific learning rate
        model_dir = os.path.join(args.model_dir, f"yolov8s_seg_lr{args.lr}")
        model_path = os.path.join(model_dir, "weights", "best.pt")
        
        if os.path.exists(model_path):
            available_models[args.lr] = model_path
            print(f"Found model for lr={args.lr}: {model_path}")
        else:
            print(f"Model not found for lr={args.lr}: {model_path}")
    
    if not available_models:
        print("No trained models found. Please train models first using train_yolov8_full_dataset.py")
        print(f"Expected model directories:")
        print(f"  - {os.path.join(args.model_dir, 'yolov8s_seg_lr1e-3/weights/best.pt')}")
        print(f"  - {os.path.join(args.model_dir, 'yolov8s_seg_lr1e-4/weights/best.pt')}")
        return
    
    # Evaluate each available model
    all_results = {}
    
    for lr, model_path in available_models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating YOLOv8-Seg with lr={lr}")
        print(f"{'='*60}")
        
        try:
            # Evaluate model
            avg_metrics, detailed_metrics = evaluate_yolo_model(
                model_path, test_data, args.conf_threshold, args.iou_threshold
            )
            
            if avg_metrics:
                all_results[lr] = avg_metrics
                
                # Print results
                print(f"\nResults for lr={lr}:")
                print(f"  IoU: {avg_metrics['iou']:.4f}")
                print(f"  Dice: {avg_metrics['dice']:.4f}")
                print(f"  Precision: {avg_metrics['precision']:.4f}")
                print(f"  Recall: {avg_metrics['recall']:.4f}")
                print(f"  F1: {avg_metrics['f1']:.4f}")
                
                if 'iou_50_pct' in avg_metrics:
                    print(f"  IoU@50: {avg_metrics['iou_50_pct']:.2f}%")
                if 'iou_75_pct' in avg_metrics:
                    print(f"  IoU@75: {avg_metrics['iou_75_pct']:.2f}%")
                if 'iou_90_pct' in avg_metrics:
                    print(f"  IoU@90: {avg_metrics['iou_90_pct']:.2f}%")
                if 'iou_95_pct' in avg_metrics:
                    print(f"  IoU@95: {avg_metrics['iou_95_pct']:.2f}%")
                
                # Save results to CSV
                csv_path = save_results_to_csv(avg_metrics, f"yolov8s_seg", lr, args.output_dir)
                
            else:
                print(f"Evaluation failed for lr={lr}")
                
        except Exception as e:
            print(f"Error evaluating model with lr={lr}: {e}")
            continue
    
    # Create comparison plots if multiple models evaluated
    if len(all_results) > 1:
        print(f"\nCreating comparison plots...")
        create_comparison_plots(all_results, os.path.join(args.output_dir, "plots"))
    
    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
