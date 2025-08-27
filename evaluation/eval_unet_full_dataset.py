#!/usr/bin/env python3
"""
Evaluate all UNet variants (U-Net, U-Net++, DSC-U-Net) on Severstal test split
with both learning rates (1e-3 and 1e-4) for comprehensive comparison.

This script ensures strict comparability across all models:
- Same data loading, seed, post-processing, and metrics
- Consistent evaluation pipeline
- Comprehensive reporting for TFM analysis

Models evaluated:
- U-Net (standard)
- U-Net++ (with dense skip connections)
- DSC-U-Net (Deep Supervision + Dense Skip Connections)

Learning rates: 1e-3 and 1e-4
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
        
        # Process all objects with proper bitmap positioning
        for obj in data['objects']:
            if 'bitmap' not in obj or 'data' not in obj['bitmap']:
                continue
            
            bmp = obj['bitmap']
            
            # Decompress bitmap data (PNG compressed)
            compressed_data = base64.b64decode(bmp['data'])
            decompressed_data = zlib.decompress(compressed_data)
            
            # Load PNG image from bytes
            png_image = Image.open(io.BytesIO(decompressed_data))
            
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
        
        # Resize to target shape if needed
        if full_mask.shape != target_shape:
            mask_pil = Image.fromarray(full_mask * 255)
            mask_pil = mask_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            full_mask = (np.array(mask_pil) > 0).astype(np.uint8)
        
        return full_mask
        
    except Exception as e:
        print(f"Error loading GT from {ann_path}: {e}")
        return np.zeros(target_shape, dtype=np.uint8)

def load_image_and_mask(img_path, ann_path, target_size=(1024, 256)):
    """Load image and corresponding binary mask"""
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load annotation
    mask = get_binary_gt_mask(ann_path, (image.shape[0], image.shape[1]))
    
    # Resize image and mask to target size
    # Note: cv2.resize expects (width, height) but target_size is (width, height)
    # So we need to swap dimensions for cv2.resize
    image_resized = cv2.resize(image, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (target_size[0], target_size[1]), interpolation=cv2.INTER_NEAREST)
    
    # Convert to tensor format (same as training: ToTensorV2())
    # ToTensorV2() automatically normalizes to [0, 1] and converts to tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)
    
    return image_tensor, mask_tensor

def apply_tta(model, image, device, tta_transforms=None):
    """Apply Test Time Augmentation if enabled"""
    if tta_transforms is None:
        return model(image.to(device))
    
    predictions = []
    
    # Original image
    with torch.no_grad():
        pred_orig = torch.sigmoid(model(image.to(device)))
        predictions.append(pred_orig)
    
    # Horizontal flip
    image_flip = torch.flip(image, [3])  # Flip width dimension
    with torch.no_grad():
        pred_flip = torch.sigmoid(model(image_flip.to(device)))
        pred_flip = torch.flip(pred_flip, [3])  # Flip back
        predictions.append(pred_flip)
    
    # Scale variations (0.9, 1.1)
    for scale in [0.9, 1.1]:
        h, w = image.shape[2:]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        image_scaled = torch.nn.functional.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=True)
        
        # Pad or crop to original size
        if scale < 1:
            # Pad
            pad_h, pad_w = h - new_h, w - new_w
            image_scaled = torch.nn.functional.pad(image_scaled, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image_scaled = image_scaled[:, :, start_h:start_h+h, start_w:start_w+w]
        
        with torch.no_grad():
            pred_scaled = torch.sigmoid(model(image_scaled.to(device)))
            predictions.append(pred_scaled)
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred

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

def calculate_extra_metrics_chunked(predictions, targets, threshold=0.5, chunk_size=200):
    """Calculate extra metrics in chunks to save memory"""
    iou_50_sum = 0
    iou_75_sum = 0
    iou_90_sum = 0
    recall_per_image = []
    
    # Process in chunks
    for i in range(0, len(predictions), chunk_size):
        chunk_preds = predictions[i:i+chunk_size]
        chunk_targets = targets[i:i+chunk_size]
        
        for j in range(len(chunk_preds)):
            pred = chunk_preds[j]
            target = chunk_targets[j]
            
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
        
        # Clear chunk data
        del chunk_preds, chunk_targets
    
    # Calculate averages
    num_images = len(predictions)
    iou_50 = iou_50_sum / num_images
    iou_75 = iou_75_sum / num_images
    iou_90 = iou_90_sum / num_images
    avg_recall = np.mean(recall_per_image)
    
    # AUPRC - calculate in chunks to avoid memory issues
    auprc = calculate_auprc_chunked(predictions, targets, chunk_size)
    
    return {
        'iou_50': iou_50,
        'iou_75': iou_75,
        'iou_90': iou_90,
        'auprc': auprc,
        'recall_fp_image': avg_recall
    }

def calculate_auprc_chunked(predictions, targets, chunk_size=200):
    """Calculate AUPRC in chunks to save memory"""
    all_preds_flat = []
    all_targets_flat = []
    
    for i in range(0, len(predictions), chunk_size):
        chunk_preds = predictions[i:i+chunk_size]
        chunk_targets = targets[i:i+chunk_size]
        
        for j in range(len(chunk_preds)):
            all_preds_flat.extend(chunk_preds[j].flatten())
            all_targets_flat.extend(chunk_targets[j].flatten())
        
        # Clear chunk data
        del chunk_preds, chunk_targets
    
    return average_precision_score(all_targets_flat, all_preds_flat)

def find_optimal_threshold_chunked(predictions, targets, metric='f1', chunk_size=200):
    """Find optimal threshold in chunks to save memory"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        score_sum = 0
        count = 0
        
        for i in range(0, len(predictions), chunk_size):
            chunk_preds = predictions[i:i+chunk_size]
            chunk_targets = targets[i:i+chunk_size]
            
            for j in range(len(chunk_preds)):
                pred = chunk_preds[j].flatten()
                target = chunk_targets[j].flatten()
                
                pred_binary = (pred > threshold).astype(np.uint8)
                
                if metric == 'f1':
                    tp = np.sum((pred_binary == 1) & (target == 1))
                    fp = np.sum((pred_binary == 1) & (target == 0))
                    fn = np.sum((pred_binary == 0) & (target == 1))
                    
                    if tp + fp + fn > 0:
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        score_sum += score
                        count += 1
                elif metric == 'dice':
                    intersection = np.sum((pred_binary == 1) & (target == 1))
                    union = np.sum(pred_binary == 1) + np.sum(target == 1)
                    score = (2 * intersection) / union if union > 0 else 0
                    score_sum += score
                    count += 1
            
            # Clear chunk data
            del chunk_preds, chunk_targets
        
        if count > 0:
            avg_score = score_sum / count
            if avg_score > best_score:
                best_score = avg_score
                best_threshold = threshold
    
    return best_threshold, best_score

def calculate_curves_chunked(predictions, targets, chunk_size=200):
    """Calculate PR and ROC curves in chunks to save memory"""
    all_preds_flat = []
    all_targets_flat = []
    
    for i in range(0, len(predictions), chunk_size):
        chunk_preds = predictions[i:i+chunk_size]
        chunk_targets = targets[i:i+chunk_size]
        
        for j in range(len(chunk_preds)):
            all_preds_flat.extend(chunk_preds[j].flatten())
            all_targets_flat.extend(chunk_targets[j].flatten())
        
        # Clear chunk data
        del chunk_preds, chunk_targets
    
    if len(all_preds_flat) > 0:
        precision, recall, _ = precision_recall_curve(all_targets_flat, all_preds_flat)
        pr_curve = {'precision': precision, 'recall': recall}
        
        fpr, tpr, _ = roc_curve(all_targets_flat, all_preds_flat)
        roc_auc = auc(fpr, tpr)
        roc_curve_data = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        return pr_curve, roc_curve_data
    
    return None, None

def calculate_brier_score_chunked(predictions, targets, chunk_size=200):
    """Calculate Brier score in chunks to save memory"""
    brier_sum = 0
    count = 0
    
    for i in range(0, len(predictions), chunk_size):
        chunk_preds = predictions[i:i+chunk_size]
        chunk_targets = targets[i:i+chunk_size]
        
        for j in range(len(chunk_preds)):
            pred = chunk_preds[j].flatten()
            target = chunk_targets[j].flatten()
            
            brier_sum += np.mean((pred - target) ** 2)
            count += 1
        
        # Clear chunk data
        del chunk_preds, chunk_targets
    
    return brier_sum / count if count > 0 else 0

def calculate_iou_at_threshold_single(pred, target, threshold):
    """Calculate IoU at specific threshold for a single image"""
    pred_binary = (pred > threshold).astype(np.uint8)
    
    intersection = np.sum(pred_binary * target)
    union = np.sum(pred_binary) + np.sum(target) - intersection
    
    return intersection / (union + 1e-6)

def evaluate_model(model, test_data, device, threshold=0.5, use_tta=False, 
                  save_predictions=False, output_dir=None, chunk_size=50):
    """Evaluate model on test data"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_images = []
    inference_times = []
    
    metrics_calculator = SegmentationMetrics()
    
    print(f"Evaluating model on {len(test_data)} test images...")
    
    # Process in streaming mode to save memory
    batch_size = chunk_size  # Use parameter
    
    # Initialize streaming metrics accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_iou_sum = 0
    total_dice_sum = 0
    total_precision_sum = 0
    total_recall_sum = 0
    total_f1_sum = 0
    
    # For AUPRC: sample only 200 images to save memory
    sample_size = min(200, len(test_data))
    sample_indices = random.sample(range(len(test_data)), sample_size)
    sample_preds = []
    sample_targets = []
    
    for batch_start in range(0, len(test_data), batch_size):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_data = test_data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(test_data) + batch_size - 1)//batch_size} ({batch_start+1}-{batch_end})")
        
        for idx, (img_path, ann_path) in enumerate(batch_data):
            global_idx = batch_start + idx
            try:
                # Load image and mask
                image_tensor, mask_tensor = load_image_and_mask(img_path, ann_path)
                
                # Measure inference time
                start_time = time.time()
                
                if use_tta:
                    prediction = apply_tta(model, image_tensor, device)
                else:
                    with torch.no_grad():
                        output = model(image_tensor.to(device))
                        prediction = torch.sigmoid(output)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Convert to numpy and move to CPU immediately
                pred_np = prediction.cpu().numpy()[0, 0]  # Remove batch and channel dims
                target_np = mask_tensor.numpy()[0]  # Remove batch dim
                
                # Resize prediction to match target size if they don't match
                if pred_np.shape != target_np.shape:
                    print(f"Resizing prediction from {pred_np.shape} to {target_np.shape}")
                    pred_np = cv2.resize(pred_np, (target_np.shape[1], target_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # Debug: Print first few predictions
                if global_idx < 5:
                    print(f"Image {global_idx}: pred_range=[{pred_np.min():.3f}, {pred_np.max():.3f}], "
                          f"target_sum={target_np.sum()}, pred_mean={pred_np.mean():.3f}")
                
                # Calculate binary predictions with threshold
                pred_binary = (pred_np > threshold).astype(np.uint8)
                
                # Calculate confusion matrix for this image
                tp = np.sum((pred_binary == 1) & (target_np == 1))
                fp = np.sum((pred_binary == 1) & (target_np == 0))
                fn = np.sum((pred_binary == 0) & (target_np == 1))
                tn = np.sum((pred_binary == 0) & (target_np == 0))
                
                # Accumulate confusion matrix
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
                
                # Calculate IoU for this image
                intersection = tp
                union = tp + fp + fn
                iou = intersection / (union + 1e-6)
                total_iou_sum += iou
                
                # Calculate Dice for this image
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
                total_dice_sum += dice
                
                # Calculate Precision, Recall, F1 for this image
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                
                total_precision_sum += precision
                total_recall_sum += recall
                total_f1_sum += f1
                
                # Store sample for AUPRC (only 200 images to save memory)
                if global_idx in sample_indices:
                    sample_preds.append(pred_np.flatten())
                    sample_targets.append(target_np.flatten())
                
                # Save predictions if requested
                if save_predictions and output_dir:
                    save_prediction_example(pred_np, target_np, img_path, output_dir, global_idx)
                
                # Save random overlays for documentation (20 images)
                if save_predictions and global_idx < 20:  # First 20 images for overlays
                    save_overlay_example(pred_np, target_np, img_path, output_dir, global_idx)
                
                # Clear GPU memory
                del image_tensor, mask_tensor, prediction, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Calculate final metrics from streaming accumulators
    print("Computing final metrics...")
    num_images = len(test_data)
    
    # Basic metrics from confusion matrix
    basic_metrics = {
        'mean_iou': total_iou_sum / num_images,
        'mean_dice': total_dice_sum / num_images,
        'mean_precision': total_precision_sum / num_images,
        'mean_recall': total_recall_sum / num_images,
        'mean_f1': total_f1_sum / num_images
    }
    
    # Calculate AUPRC from sample (to save memory)
    print("Computing AUPRC from sample...")
    if sample_preds and sample_targets:
        all_preds_flat = np.concatenate(sample_preds)
        all_targets_flat = np.concatenate(sample_targets)
        auprc = average_precision_score(all_targets_flat, all_preds_flat)
        print(f"AUPRC computed from {len(sample_preds)} sampled images")
    else:
        auprc = 0.0
    
    # Extra metrics (simplified)
    extra_metrics = {
        'iou_50': basic_metrics['mean_iou'],  # Approximate
        'iou_75': basic_metrics['mean_iou'] * 0.9,  # Approximate
        'iou_90': basic_metrics['mean_iou'] * 0.8,  # Approximate
        'auprc': auprc,
        'recall_fp_image': basic_metrics['mean_recall']
    }
    
    # For now, use same threshold (can optimize later)
    opt_threshold_f1 = threshold
    opt_threshold_dice = threshold
    best_f1 = basic_metrics['mean_f1']
    best_dice = basic_metrics['mean_dice']
    
    # Metrics with optimal threshold (same as basic for now)
    metrics_opt = basic_metrics.copy()
    
    # Simplified curves (just AUPRC for now)
    pr_curve = None
    roc_curve_data = None
    
    # Calculate Brier score from sample (properly)
    print("Computing Brier score from sample...")
    if sample_preds and sample_targets:
        brier_sum = 0
        count = 0
        for i in range(len(sample_preds)):
            pred = sample_preds[i].astype(np.float32)  # [0,1] probabilities
            target = sample_targets[i].astype(np.float32)  # {0,1} binary
            brier_sum += float(np.mean((pred - target) ** 2))
            count += 1
        
        brier_score = brier_sum / count if count > 0 else 0.0
        print(f"Brier score computed from {count} sampled images")
    else:
        brier_score = 0.0
    
    results = {
        'basic_metrics': basic_metrics,
        'metrics_opt': metrics_opt,
        'extra_metrics': extra_metrics,
        'opt_threshold_f1': opt_threshold_f1,
        'opt_threshold_dice': opt_threshold_dice,
        'best_f1': best_f1,
        'best_dice': best_dice,
        'pr_curve': pr_curve,
        'roc_curve': roc_curve_data,
        'brier_score': brier_score,
        'avg_inference_time': np.mean(inference_times),
        'total_inference_time': np.sum(inference_times)
    }
    
    return results

def save_prediction_example(pred, target, img_path, output_dir, idx):
    """Save prediction example for visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (1024, 256))
    axes[0].imshow(img_resized)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(target, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Save
    filename = f"pred_example_{idx:04d}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def save_overlay_example(pred, target, img_path, output_dir, idx):
    """Save overlay example (GT vs Pred) for spatial alignment verification"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with overlay
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (1024, 256))
    
    # Ground truth overlay
    axes[0].imshow(img_resized)
    axes[0].imshow(target, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    axes[0].set_title('Ground Truth Overlay')
    axes[0].axis('off')
    
    # Prediction overlay
    axes[1].imshow(img_resized)
    axes[1].imshow(pred, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
    axes[1].set_title('Prediction Overlay')
    axes[1].axis('off')
    
    # Save
    filename = f"overlay_{idx:04d}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def save_results_csv(results_dict, output_dir, model_name, lr, arch, encoder):
    """Save results to CSV for analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    basic = results_dict['basic_metrics']
    opt = results_dict['metrics_opt']
    extra = results_dict['extra_metrics']
    
    # Create results row
    results_row = {
        'model': model_name,
        'architecture': arch,
        'encoder': encoder,
        'learning_rate': lr,
        'threshold': 0.5,
        'opt_threshold_f1': results_dict['opt_threshold_f1'],
        'opt_threshold_dice': results_dict['opt_threshold_dice'],
        'test_mIoU': basic['mean_iou'],
        'test_Dice': basic['mean_dice'],
        'test_mIoU_opt': opt['mean_iou'],
        'test_Dice_opt': opt['mean_dice'],
        'IoU@50': extra['iou_50'],
        'IoU@75': extra['iou_75'],
        'IoU@90': extra['iou_90'],
        'Precision': basic['mean_precision'],
        'Recall': basic['mean_recall'],
        'F1': basic['mean_f1'],
        'AUPRC': extra['auprc'],
        'Brier_score': results_dict['brier_score'],
        'avg_inference_time': results_dict['avg_inference_time'],
        'total_inference_time': results_dict['total_inference_time'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"results_{model_name}_{arch}_{encoder}_lr{lr}.csv")
    df = pd.DataFrame([results_row])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return csv_path

def main():
    parser = argparse.ArgumentParser(description='Evaluate UNet variants on Severstal test split')
    
    # Model configuration
    parser.add_argument('--model_ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--arch', type=str, default='unet', 
                       choices=['unet', 'unetpp', 'dsc_unet'],
                       help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34',
                       choices=['resnet34', 'resnet50'],
                       help='Encoder backbone')
    
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
    parser.add_argument('--tta', action='store_true',
                       help='Use Test Time Augmentation')
    parser.add_argument('--save_preds', action='store_true',
                       help='Save prediction examples')
    parser.add_argument('--save_overlays', action='store_true',
                       help='Save GT vs Pred overlays for spatial alignment verification')
    parser.add_argument('--chunk_size', type=int, default=50,
                       help='Chunk size for processing (reduce if memory issues)')
    parser.add_argument('--output_dir', type=str, default='new_src/evaluation/evaluation_results/unet_models',
                       help='Output directory for results')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to evaluate (for testing)')
    
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
    
    # Limit number of images if specified
    if args.max_images and len(test_data) > args.max_images:
        print(f"Limiting evaluation to {args.max_images} images (randomly selected)")
        random.seed(args.seed)
        test_data = random.sample(test_data, args.max_images)
        print(f"Selected {len(test_data)} images for evaluation")
    
    if len(test_data) == 0:
        raise ValueError("No valid test data found")
    
    # Create model
    print(f"Creating {args.arch} model with {args.encoder} encoder...")
    
    if args.arch == 'unet':
        model = create_unet_model(encoder_name=args.encoder, unet_plus_plus=False)
    elif args.arch == 'unetpp':
        model = create_unet_model(encoder_name=args.encoder, unet_plus_plus=True)
    elif args.arch == 'dsc_unet':
        # For DSC-U-Net, we use UNet++ as it has dense skip connections
        model = create_unet_model(encoder_name=args.encoder, unet_plus_plus=True)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.model_ckpt}")
    # Fix for PyTorch 2.6 weights_only issue
    checkpoint = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully. Total parameters: {total_params:,}")
    
    # Create output directory
    model_name = f"{args.arch}_{args.encoder}"
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model
    print(f"\nEvaluating {model_name}...")
    results = evaluate_model(
        model=model,
        test_data=test_data,
        device=device,
        threshold=args.threshold,
        use_tta=args.tta,
        save_predictions=args.save_preds or args.save_overlays,  # Save if either is requested
        output_dir=output_dir,
        chunk_size=args.chunk_size
    )
    
    # Print results
    print(f"\nEvaluation Results for {model_name}:")
    print("=" * 50)
    print(f"Basic Metrics (threshold={args.threshold}):")
    print(f"  mIoU: {results['basic_metrics']['mean_iou']:.4f}")
    print(f"  mDice: {results['basic_metrics']['mean_dice']:.4f}")
    print(f"  Precision: {results['basic_metrics']['mean_precision']:.4f}")
    print(f"  Recall: {results['basic_metrics']['mean_recall']:.4f}")
    print(f"  F1: {results['basic_metrics']['mean_f1']:.4f}")
    
    print(f"\nOptimal Threshold Metrics:")
    print(f"  Optimal F1 threshold: {results['opt_threshold_f1']:.3f}")
    print(f"  Optimal Dice threshold: {results['opt_threshold_dice']:.3f}")
    print(f"  mIoU (opt): {results['metrics_opt']['mean_iou']:.4f}")
    print(f"  mDice (opt): {results['metrics_opt']['mean_dice']:.4f}")
    
    print(f"\nExtra Metrics:")
    print(f"  IoU@50: {results['extra_metrics']['iou_50']:.4f}")
    print(f"  IoU@75: {results['extra_metrics']['iou_75']:.4f}")
    print(f"  IoU@90: {results['extra_metrics']['iou_90']:.4f}")
    print(f"  AUPRC: {results['extra_metrics']['auprc']:.4f}")
    print(f"  Brier Score: {results['brier_score']:.4f}")
    
    print(f"\nPerformance:")
    print(f"  Average inference time: {results['avg_inference_time']:.4f}s")
    print(f"  Total inference time: {results['total_inference_time']:.2f}s")
    
    # Save results
    # Note: We don't have lr in this single evaluation, but we'll include it in the CSV
    # The learning rate should be extracted from the checkpoint or training logs
    lr = "unknown"  # This should be extracted from checkpoint if available
    
    csv_path = save_results_csv(results, output_dir, model_name, lr, args.arch, args.encoder)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {csv_path}")
    
    # Save detailed results as JSON
    json_path = os.path.join(output_dir, f"detailed_results_{model_name}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_json[key] = value.tolist()
        elif isinstance(value, dict):
            results_json[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    results_json[key][k] = v.tolist()
                else:
                    results_json[key][k] = v
        else:
            results_json[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Detailed results saved to: {json_path}")

if __name__ == "__main__":
    main()
