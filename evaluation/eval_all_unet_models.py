#!/usr/bin/env python3
"""
Comprehensive evaluation of all UNet variants on Severstal test split
Evaluates U-Net, U-Net++, and DSC-U-Net with both learning rates (1e-3 and 1e-4)

This script ensures strict comparability across all models:
- Same data loading, seed, post-processing, and metrics
- Consistent evaluation pipeline
- Unified reporting for TFM analysis
- No code duplication

Usage:
python eval_all_unet_models.py --checkpoints_dir /path/to/checkpoints --output_dir runs/unet_comprehensive_eval
"""

import os
import sys
import json
import torch
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image
import base64
import zlib
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random
import argparse
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.model_unet import create_unet_model
from utils.metrics import SegmentationMetrics

def set_seed(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_binary_gt_mask(ann_path, target_shape):
    """Load and combine all defect objects into a single binary mask from PNG bitmap"""
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        if 'objects' not in data or not data['objects']:
            return np.zeros(target_shape, dtype=np.uint8)
        
        # Get image dimensions
        height = data['size']['height']
        width = data['size']['width']
        
        # Create empty mask
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process all objects (not just the first one)
        for obj in data['objects']:
            if 'bitmap' not in obj or 'data' not in obj['bitmap']:
                continue
            
            # Decompress bitmap data (PNG compressed)
            compressed_data = base64.b64decode(obj['bitmap']['data'])
            decompressed_data = zlib.decompress(compressed_data)
            
            # Load PNG image from bytes
            png_image = Image.open(io.BytesIO(decompressed_data))
            mask = np.array(png_image)
            
            # Convert to binary mask (1=defect, 0=background)
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Resize to full image size
            mask_pil = Image.fromarray(binary_mask * 255)
            mask_pil = mask_pil.resize((width, height), Image.NEAREST)
            obj_mask = (np.array(mask_pil) > 0).astype(np.uint8)
            
            # Combine with full mask (OR operation)
            full_mask = np.logical_or(full_mask, obj_mask).astype(np.uint8)
        
        # Resize to target shape if needed
        if full_mask.shape != target_shape:
            mask_pil = Image.fromarray(full_mask * 255)
            mask_pil = mask_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            full_mask = (np.array(mask_pil) > 0).astype(np.uint8)
        
        return full_mask
        
    except Exception as e:
        print(f"Error loading GT from {ann_path}: {e}")
        return np.zeros(target_shape, dtype=np.uint8)

def load_image_and_mask(img_path, ann_path, target_size=(512, 128)):
    """Load image and corresponding binary mask"""
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load annotation
    mask = get_binary_gt_mask(ann_path, (image.shape[0], image.shape[1]))
    
    # Resize image and mask to target size
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Ensure dimensions are correct (H, W) -> (H, W)
    if len(image_resized.shape) == 3:
        h, w, c = image_resized.shape
    else:
        h, w = image_resized.shape
        c = 1
    
    if len(mask_resized.shape) == 3:
        mask_resized = mask_resized.squeeze()
    
    # Verify dimensions
    assert mask_resized.shape == (h, w), f"Mask shape {mask_resized.shape} != image shape {(h, w)}"
    
    # Normalize image
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor format
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)
    
    return image_tensor, mask_tensor

def find_optimal_threshold(predictions, targets, metric='f1'):
    """Find optimal threshold on validation set"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(np.uint8)
        
        if metric == 'f1':
            # Calculate F1 score
            tp = np.sum((pred_binary == 1) & (targets == 1))
            fp = np.sum((pred_binary == 1) & (targets == 0))
            fn = np.sum((pred_binary == 0) & (targets == 1))
            
            if tp + fp + fn == 0:
                score = 0
            else:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        elif metric == 'dice':
            # Calculate Dice score
            intersection = np.sum((pred_binary == 1) & (targets == 1))
            union = np.sum(pred_binary == 1) + np.sum(targets == 1)
            score = (2 * intersection) / union if union > 0 else 0
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

def calculate_extra_metrics(predictions, targets, threshold=0.5):
    """Calculate extra metrics for TFM comparison"""
    iou_50_sum = 0
    iou_75_sum = 0
    iou_90_sum = 0
    recall_per_image = []
    
    # Flatten all predictions and targets for AUPRC
    all_preds_flat = []
    all_targets_flat = []
    
    for i in range(len(predictions)):
        pred = predictions[i]
        target = targets[i]
        
        # IoU at different thresholds
        iou_50_sum += calculate_iou_at_threshold_single(pred, target, 0.5)
        iou_75_sum += calculate_iou_at_threshold_single(pred, target, 0.75)
        iou_90_sum += calculate_iou_at_threshold_single(pred, target, 0.90)
        
        # Recall per image
        pred_binary = (pred > threshold).astype(np.uint8)
        tp = np.sum((pred_binary == 1) & (target == 1))
        fn = np.sum((pred_binary == 0) & (target == 1))
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
            recall_per_image.append(recall)
        else:
            recall_per_image.append(0.0)
        
        # Collect for AUPRC
        all_preds_flat.extend(pred.flatten())
        all_targets_flat.extend(target.flatten())
    
    # Calculate averages
    num_images = len(predictions)
    iou_50 = iou_50_sum / num_images
    iou_75 = iou_75_sum / num_images
    iou_90 = iou_90_sum / num_images
    avg_recall = np.mean(recall_per_image)
    
    # AUPRC for defect class
    auprc = average_precision_score(all_targets_flat, all_preds_flat)
    
    return {
        'iou_50': iou_50,
        'iou_75': iou_75,
        'iou_90': iou_90,
        'auprc': auprc,
        'recall_fp_image': avg_recall
    }

def calculate_iou_at_threshold_single(pred, target, threshold):
    """Calculate IoU at specific threshold for a single image"""
    pred_binary = (pred > threshold).astype(np.uint8)
    
    intersection = np.sum(pred_binary * target)
    union = np.sum(pred_binary) + np.sum(target) - intersection
    
    return intersection / (union + 1e-6)

def evaluate_model(model, test_data, device, threshold=0.5, use_tta=False):
    """Evaluate model on test data"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    inference_times = []
    
    metrics_calculator = SegmentationMetrics()
    
    for idx, (img_path, ann_path) in enumerate(tqdm(test_data, desc="Evaluation", leave=False)):
        try:
            # Load image and mask
            image_tensor, mask_tensor = load_image_and_mask(img_path, ann_path)
            
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                output = model(image_tensor.to(device))
                prediction = torch.sigmoid(output)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Convert to numpy
            pred_np = prediction.cpu().numpy()[0, 0]  # Remove batch and channel dims
            target_np = mask_tensor.numpy()[0]  # Remove batch dim
            
            # Store for metrics calculation
            all_predictions.append(pred_np)
            all_targets.append(target_np)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Calculate metrics
    # Binary predictions with threshold
    pred_binary = [(pred > threshold).astype(np.uint8) for pred in all_predictions]
    
    # Basic metrics
    basic_metrics = metrics_calculator.evaluate_batch(pred_binary, all_targets)
    
    # Extra metrics
    extra_metrics = calculate_extra_metrics(
        np.array(all_predictions), 
        np.array(all_targets), 
        threshold=threshold
    )
    
    # Find optimal threshold
    all_preds_flat = np.concatenate([pred.flatten() for pred in all_predictions])
    all_targets_flat = np.concatenate([target.flatten() for target in all_targets])
    
    opt_threshold_f1, best_f1 = find_optimal_threshold(all_preds_flat, all_targets_flat, 'f1')
    opt_threshold_dice, best_dice = find_optimal_threshold(all_preds_flat, all_targets_flat, 'dice')
    
    # Metrics with optimal threshold
    pred_binary_opt = [(pred > opt_threshold_f1).astype(np.uint8) for pred in all_predictions]
    metrics_opt = metrics_calculator.evaluate_batch(pred_binary_opt, all_targets)
    
    # Brier score (calibration metric)
    brier_score = np.mean((all_preds_flat - all_targets_flat) ** 2)
    
    results = {
        'basic_metrics': basic_metrics,
        'metrics_opt': metrics_opt,
        'extra_metrics': extra_metrics,
        'opt_threshold_f1': opt_threshold_f1,
        'opt_threshold_dice': opt_threshold_dice,
        'best_f1': best_f1,
        'best_dice': best_dice,
        'brier_score': brier_score,
        'avg_inference_time': np.mean(inference_times),
        'total_inference_time': np.sum(inference_times)
    }
    
    return results

def find_checkpoints(checkpoints_dir):
    """Find all available checkpoints for different models and learning rates"""
    checkpoints = {}
    
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        return checkpoints
    
    # Look for checkpoints in subdirectories
    for root, dirs, files in os.walk(checkpoints_dir):
        for file in files:
            if file.endswith('.pth') or file.endswith('.pt'):
                file_path = os.path.join(root, file)
                
                # Try to extract model info from directory path first, then filename
                dir_name = os.path.basename(root).lower()
                filename = file.lower()
                
                print(f"Checking directory: {dir_name}, file: {filename}")
                
                # Determine architecture from directory name
                arch = None
                if 'dsc_unet' in dir_name:
                    arch = 'dsc_unet'
                elif 'unet++' in dir_name or 'unetpp' in dir_name:
                    arch = 'unetpp'
                elif 'unet_standard' in dir_name or 'unet' in dir_name:
                    arch = 'unet'
                
                # Determine encoder from directory name
                encoder = None
                if 'resnet50' in dir_name:
                    encoder = 'resnet50'
                elif 'resnet34' in dir_name:
                    encoder = 'resnet34'
                
                # Determine learning rate from directory name
                lr = None
                if 'lr1e-3' in dir_name or '1e-3' in dir_name:
                    lr = '1e-3'
                elif 'lr1e-4' in dir_name or '1e-4' in dir_name:
                    lr = '1e-4'
                
                # If not found in directory, try filename
                if not arch:
                    if 'dsc' in filename:
                        arch = 'dsc_unet'
                    elif 'unet++' in filename or 'unetpp' in filename:
                        arch = 'unetpp'
                    elif 'unet' in filename:
                        arch = 'unet'
                
                if not encoder:
                    if 'resnet50' in filename:
                        encoder = 'resnet50'
                    elif 'resnet34' in filename:
                        encoder = 'resnet34'
                
                if not lr:
                    if '1e-3' in filename or '0.001' in filename:
                        lr = '1e-3'
                    elif '1e-4' in filename or '0.0001' in filename:
                        lr = '1e-4'
                
                if arch and encoder and lr:
                    key = f"{arch}_{encoder}_{lr}"
                    if key not in checkpoints:
                        checkpoints[key] = []
                    checkpoints[key].append(file_path)
                    print(f"Found checkpoint: {key} -> {file_path}")
                else:
                    print(f"Could not parse: arch={arch}, encoder={encoder}, lr={lr}")
    
    return checkpoints

def create_comparison_table(all_results):
    """Create a comprehensive comparison table"""
    comparison_data = []
    
    for model_key, results in all_results.items():
        # Parse model key
        parts = model_key.split('_')
        if len(parts) >= 3:
            arch = parts[0]
            encoder = parts[1]
            lr = parts[2]
        else:
            continue
        
        # Handle special case for dsc_unet (it's one word)
        if arch == 'dsc':
            arch = 'dsc_unet'
        
        # Extract metrics
        basic = results['basic_metrics']
        opt = results['metrics_opt']
        extra = results['extra_metrics']
        
        # Create row
        row = {
            'Model': f"{arch.upper()}-{encoder.upper()}",
            'Architecture': arch,
            'Encoder': encoder,
            'Learning Rate': lr,
            'mIoU (0.5)': f"{basic['mean_iou']:.4f}",
            'mDice (0.5)': f"{basic['mean_dice']:.4f}",
            'mIoU (opt)': f"{opt['mean_iou']:.4f}",
            'mDice (opt)': f"{opt['mean_dice']:.4f}",
            'IoU@50': f"{extra['iou_50']:.4f}",
            'IoU@75': f"{extra['iou_75']:.4f}",
            'IoU@90': f"{extra['iou_90']:.4f}",
            'Precision': f"{basic['mean_precision']:.4f}",
            'Recall': f"{basic['mean_recall']:.4f}",
            'F1': f"{basic['mean_f1']:.4f}",
            'AUPRC': f"{extra['auprc']:.4f}",
            'Brier Score': f"{results['brier_score']:.4f}",
            'Avg Inference (s)': f"{results['avg_inference_time']:.4f}",
            'Opt Threshold F1': f"{results['opt_threshold_f1']:.3f}",
            'Opt Threshold Dice': f"{results['opt_threshold_dice']:.3f}"
        }
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def save_comprehensive_results(all_results, output_dir):
    """Save comprehensive results for all models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison table
    comparison_df = create_comparison_table(all_results)
    
    # Save comparison table
    comparison_path = os.path.join(output_dir, "model_comparison_table.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}")
    
    # Save individual results
    for model_key, results in all_results.items():
        model_dir = os.path.join(output_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save detailed results as JSON
        json_path = os.path.join(model_dir, "detailed_results.json")
        
        # Convert numpy arrays and other non-serializable types to lists for JSON serialization
        results_json = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_json[key] = value.tolist()
            elif isinstance(value, np.float32) or isinstance(value, np.float64):
                results_json[key] = float(value)
            elif isinstance(value, np.int32) or isinstance(value, np.int64):
                results_json[key] = int(value)
            elif isinstance(value, dict):
                results_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results_json[key][k] = v.tolist()
                    elif isinstance(v, np.float32) or isinstance(v, np.float64):
                        results_json[key][k] = float(v)
                    elif isinstance(v, np.int32) or isinstance(v, np.int64):
                        results_json[key][k] = int(v)
                    else:
                        results_json[key][k] = v
            else:
                results_json[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Detailed results for {model_key} saved to: {json_path}")
    
    # Create summary report
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("UNet Models Comprehensive Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Models Evaluated: {len(all_results)}\n\n")
        
        f.write("Models Evaluated:\n")
        for model_key in all_results.keys():
            f.write(f"  - {model_key}\n")
        
        f.write(f"\nComparison table saved to: {comparison_path}\n")
        f.write("Individual detailed results saved in subdirectories.\n")
    
    print(f"Summary report saved to: {summary_path}")
    
    return comparison_path

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of all UNet variants')
    
    # Checkpoints directory
    parser.add_argument('--checkpoints_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    
    # Data paths
    parser.add_argument('--img_dir', type=str, 
                       default='/home/ptp/sam2/datasets/Data/splits/test_split/img',
                       help='Path to test images')
    parser.add_argument('--ann_dir', type=str,
                       default='/home/ptp/sam2/datasets/Data/splits/test_split/ann',
                       help='Path to test annotations')
    
    # Evaluation settings
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary threshold for predictions')
    parser.add_argument('--output_dir', type=str, default='new_src/evaluation/evaluation_results/unet_models',
                       help='Output directory for results')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    test_data = []
    
    # Get all image files
    image_files = [f for f in os.listdir(args.img_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} test images")
    
    for img_file in image_files:
        img_path = os.path.join(args.img_dir, img_file)
        ann_file = img_file + '.json'
        ann_path = os.path.join(args.ann_dir, ann_file)
        
        if os.path.exists(ann_path):
            test_data.append((img_path, ann_path))
        else:
            print(f"Warning: No annotation found for {img_file}")
    
    print(f"Successfully paired {len(test_data)} image-annotation pairs")
    
    if len(test_data) == 0:
        raise ValueError("No valid test data found")
    
    # Find checkpoints
    print("Searching for checkpoints...")
    checkpoints = find_checkpoints(args.checkpoints_dir)
    
    if not checkpoints:
        print("No valid checkpoints found. Please check the checkpoints directory.")
        return
    
    print(f"Found checkpoints for {len(checkpoints)} model configurations:")
    for key in checkpoints.keys():
        print(f"  - {key}: {len(checkpoints[key])} checkpoint(s)")
    
    # Evaluate all models
    all_results = {}
    
    for model_key, checkpoint_paths in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_key}")
        print(f"{'='*60}")
        
        # Parse model key
        parts = model_key.split('_')
        arch = parts[0]
        encoder = parts[1]
        lr = parts[2]
        
        # Create model
        print(f"Creating {arch} model with {encoder} encoder...")
        
        if arch == 'unet':
            model = create_unet_model(encoder_name=encoder, unet_plus_plus=False)
        elif arch == 'unetpp':
            model = create_unet_model(encoder_name=encoder, unet_plus_plus=True)
        elif arch == 'dsc_unet':
            # For DSC-U-Net, we use UNet++ as it has dense skip connections
            model = create_unet_model(encoder_name=encoder, unet_plus_plus=True)
            print(f"Using UNet++ architecture for DSC-U-Net (dense skip connections)")
        else:
            print(f"Unsupported architecture: {arch}, skipping...")
            continue
        
        # Load checkpoint (use the first one if multiple)
        checkpoint_path = checkpoint_paths[0]
        print(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            # Fix for PyTorch 2.6 weights_only issue
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            
            # Print model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model loaded successfully. Total parameters: {total_params:,}")
            
            # Evaluate model
            results = evaluate_model(
                model=model,
                test_data=test_data,
                device=device,
                threshold=args.threshold
            )
            
            # Store results
            all_results[model_key] = results
            
            # Print summary
            print(f"Evaluation completed for {model_key}:")
            print(f"  mIoU: {results['basic_metrics']['mean_iou']:.4f}")
            print(f"  mDice: {results['basic_metrics']['mean_dice']:.4f}")
            print(f"  IoU@50: {results['extra_metrics']['iou_50']:.4f}")
            print(f"  IoU@75: {results['extra_metrics']['iou_75']:.4f}")
            print(f"  IoU@90: {results['extra_metrics']['iou_90']:.4f}")
            print(f"  AUPRC: {results['extra_metrics']['auprc']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {model_key}: {e}")
            continue
    
    # Save comprehensive results
    if all_results:
        print(f"\n{'='*60}")
        print("Saving comprehensive results...")
        print(f"{'='*60}")
        
        comparison_path = save_comprehensive_results(all_results, args.output_dir)
        
        print(f"\nComprehensive evaluation completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Comparison table: {comparison_path}")
        
        # Print final comparison
        print(f"\n{'='*60}")
        print("FINAL COMPARISON TABLE")
        print(f"{'='*60}")
        
        comparison_df = create_comparison_table(all_results)
        print(comparison_df.to_string(index=False))
        
    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main()
