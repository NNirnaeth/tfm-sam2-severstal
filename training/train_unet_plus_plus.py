#!/usr/bin/env python3
"""
Train UNet++ model on full Severstal dataset
This is a separate script to avoid interfering with running UNet experiments

Training data: datasets/Data/splits/train_split (4200 images)
Validation data: datasets/Data/splits/val_split (466 images)

Binary segmentation: defect vs background (unified classes)
Image size: 1024x256 (maintaining aspect ratio)

TFM Requirements:
- AdamW with weight_decay=1e-4
- Compare lr = 1e-3 vs 1e-4
- BCE + Dice loss combination
- Save best model by val_mIoU/Dice
- Learn optimal threshold on validation
- Extra metrics: IoU@{0.50, 0.75, 0.90}, AUPRC, Recall@FP/image
- 3 seeds for stability
- Stratification by defect presence
- Horizontal aspect preservation
- Specific augmentation (no vertical flip, limited rotation)
- Mixed precision, gradient clipping, stability
- Comprehensive logging and metrics
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import cv2
import base64
import zlib
from PIL import Image
from io import BytesIO
from datetime import datetime
from tqdm import tqdm
import random
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import csv
from sklearn.metrics import average_precision_score, precision_recall_curve
import time

# Add new_src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import modular components
from experiments.model_unet import create_unet_model
from experiments.prepare_data_unet import create_data_loaders
from utils.metrics import SegmentationMetrics

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation"""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice


def bce_loss(pred, target):
    """Binary Cross Entropy loss with logits (safe for autocast)"""
    return F.binary_cross_entropy_with_logits(pred, target)


def combined_loss(pred, target, bce_weight=1.0, dice_weight=1.0):
    """Combined BCE + Dice loss for stability with minority classes"""
    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)
    
    return bce_weight * bce + dice_weight * dice


def find_optimal_threshold(predictions, targets, metric='f1'):
    """Find optimal threshold on validation set"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        pred_binary = (predictions > thresh).astype(np.uint8)
        
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
            intersection = np.sum(pred_binary * targets)
            score = 2 * intersection / (np.sum(pred_binary) + np.sum(targets) + 1e-6)
        else:
            continue
            
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def calculate_extra_metrics(predictions, targets, threshold=0.5):
    """Calculate extra metrics for TFM comparison with SAM2 - memory efficient"""
    # Process images one by one to avoid memory issues
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


def calculate_iou_at_threshold(predictions, targets, threshold):
    """Calculate IoU at specific threshold - deprecated, use calculate_iou_at_threshold_single"""
    return calculate_iou_at_threshold_single(predictions, targets, threshold)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    metrics_calculator = SegmentationMetrics()
    
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            # UNet++ outputs have different spatial dimensions, need to resize to match masks
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=True)
            loss = criterion(outputs, masks.unsqueeze(1))
        
        # Backward pass
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Calculate metrics every 100 batches
        if batch_idx % 100 == 0:
            with torch.no_grad():
                pred_masks = torch.sigmoid(outputs) > 0.5
                batch_metrics = metrics_calculator.compute_pixel_metrics(
                    pred_masks.cpu().numpy().flatten(),
                    masks.cpu().numpy().flatten()
                )
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}, IoU={batch_metrics['iou']:.4f}, Dice={batch_metrics['dice']:.4f}")
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            # UNet++ outputs have different spatial dimensions, need to resize to match masks
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=True)
            loss = criterion(outputs, masks.unsqueeze(1))
            total_loss += loss.item()
            
            # Get predictions (raw logits for threshold optimization)
            pred_logits = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.extend(pred_logits)
            all_targets.extend(masks.cpu().numpy())
    
    # Calculate metrics
    metrics_calculator = SegmentationMetrics()
    
    # Metrics with fixed threshold 0.5
    pred_binary_fixed = [(pred > 0.5).astype(np.uint8) for pred in all_predictions]
    metrics_fixed = metrics_calculator.evaluate_batch(pred_binary_fixed, all_targets)
    
    # Find optimal threshold
    all_preds_flat = np.concatenate([pred.flatten() for pred in all_predictions])
    all_targets_flat = np.concatenate([target.flatten() for target in all_targets])
    
    opt_threshold_f1, best_f1 = find_optimal_threshold(all_preds_flat, all_targets_flat, 'f1')
    opt_threshold_dice, best_dice = find_optimal_threshold(all_preds_flat, all_targets_flat, 'dice')
    
    # Metrics with optimal threshold
    pred_binary_opt = [(pred > opt_threshold_f1).astype(np.uint8) for pred in all_predictions]
    metrics_opt = metrics_calculator.evaluate_batch(pred_binary_opt, all_targets)
    
    # Extra metrics for TFM comparison
    extra_metrics = calculate_extra_metrics(
        np.array(all_predictions), 
        np.array(all_targets), 
        threshold=opt_threshold_f1
    )
    
    return {
        'loss': total_loss / len(dataloader),
        'metrics_fixed': metrics_fixed,
        'metrics_opt': metrics_opt,
        'opt_threshold_f1': opt_threshold_f1,
        'opt_threshold_dice': opt_threshold_dice,
        'best_f1': best_f1,
        'best_dice': best_dice,
        'extra_metrics': extra_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Train UNet++ on Severstal dataset')
    parser.add_argument('--train_path', type=str, 
                       default='/home/ptp/sam2/datasets/Data/splits/train_split',
                       help='Path to training data')
    parser.add_argument('--val_path', type=str,
                       default='/home/ptp/sam2/datasets/Data/splits/val_split',
                       help='Path to validation data')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[1024, 256],
                       help='Image size [width, height] (default: [1024, 256])')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=7,
                       help='Early stopping patience (default: 7)')
    parser.add_argument('--save_dir', type=str, default='new_src/training/training_results/unet_models',
                       help='Directory to save models')
    parser.add_argument('--encoder', type=str, default='resnet34',
                       choices=['resnet34', 'resnet50'],
                       help='Encoder backbone (default: resnet34)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Custom run name for logging')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="unet++-severstal-tfm",
            name=args.run_name or f"unet++_{args.encoder}_lr{args.lr}_seed{args.seed}",
            config=vars(args)
        )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders using modular data preparation
    train_loader, val_loader, train_size, val_size = create_data_loaders(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        num_workers=args.num_workers
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create UNet++ model
    model = create_unet_model(
        encoder_name=args.encoder,
        unet_plus_plus=True,  # Force UNet++
        in_channels=3,
        num_classes=1
    ).to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Encoder: {model.encoder_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function (BCE + Dice for stability with minority classes)
    criterion = combined_loss
    
    # Optimizer (AdamW with weight_decay=1e-4 as per TFM requirements)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop
    best_val_dice = 0
    best_val_iou = 0
    patience_counter = 0
    
    # Metrics tracking
    all_metrics = []
    
    print(f"\nStarting training with:")
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Image size: {args.img_size}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Seed: {args.seed}")
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validation
        val_results = validate_epoch(model, val_loader, criterion, device)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}")
        print(f"Val IoU (fixed 0.5): {val_results['metrics_fixed']['mean_iou']:.4f}")
        print(f"Val Dice (fixed 0.5): {val_results['metrics_fixed']['mean_dice']:.4f}")
        print(f"Val IoU (opt thresh): {val_results['metrics_opt']['mean_iou']:.4f}")
        print(f"Val Dice (opt thresh): {val_results['metrics_opt']['mean_dice']:.4f}")
        print(f"Optimal threshold (F1): {val_results['opt_threshold_f1']:.3f}")
        print(f"Optimal threshold (Dice): {val_results['opt_threshold_dice']:.3f}")
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_results['loss'],
                'val_iou_fixed': val_results['metrics_fixed']['mean_iou'],
                'val_dice_fixed': val_results['metrics_fixed']['mean_dice'],
                'val_iou_opt': val_results['metrics_opt']['mean_iou'],
                'val_dice_opt': val_results['metrics_opt']['mean_iou'],
                'opt_threshold_f1': val_results['opt_threshold_f1'],
                'opt_threshold_dice': val_results['opt_threshold_dice'],
                'learning_rate': current_lr,
                'epoch_time': epoch_time,
                **val_results['extra_metrics']
            })
        
        # Save metrics to CSV
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_results['loss'],
            'val_iou_fixed': val_results['metrics_fixed']['mean_iou'],
            'val_dice_fixed': val_results['metrics_fixed']['mean_dice'],
            'val_iou_opt': val_results['metrics_opt']['mean_iou'],
            'val_dice_opt': val_results['metrics_opt']['mean_dice'],
            'opt_threshold_f1': val_results['opt_threshold_f1'],
            'opt_threshold_dice': val_results['opt_threshold_dice'],
            'learning_rate': current_lr,
            'epoch_time': epoch_time,
            'iou_50': val_results['extra_metrics']['iou_50'],
            'iou_75': val_results['extra_metrics']['iou_75'],
            'iou_90': val_results['extra_metrics']['iou_90'],
            'auprc': val_results['extra_metrics']['auprc'],
            'recall_fp_image': val_results['extra_metrics']['recall_fp_image']
        }
        all_metrics.append(epoch_metrics)
        
        # Save best model by validation Dice (primary metric)
        val_dice = val_results['metrics_opt']['mean_dice']
        val_iou = val_results['metrics_opt']['mean_iou']
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_ckpt_path = os.path.join(args.save_dir, f"best_dice_{val_dice:.4f}_seed{args.seed}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'opt_threshold_f1': val_results['opt_threshold_f1'],
                'opt_threshold_dice': val_results['opt_threshold_dice'],
                'args': args
            }, best_ckpt_path)
            print(f"New best Dice model saved: {best_ckpt_path}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_iou_ckpt_path = os.path.join(args.save_dir, f"best_iou_{val_iou:.4f}_seed{args.seed}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'opt_threshold_f1': val_results['opt_threshold_f1'],
                'opt_threshold_dice': val_results['opt_threshold_dice'],
                'args': args
            }, best_iou_ckpt_path)
            print(f"New best IoU model saved: {best_iou_ckpt_path}")
        
        # Save checkpoint every 10 epochs (secondary priority)
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}_seed{args.seed}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'opt_threshold_f1': val_results['opt_threshold_f1'],
                'opt_threshold_dice': val_results['opt_threshold_dice'],
                'args': args
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement")
            break
        
        # Update scheduler
        scheduler.step()
    
    # Save final metrics CSV
    metrics_csv_path = os.path.join(args.save_dir, f"metrics_seed{args.seed}.csv")
    with open(metrics_csv_path, 'w', newline='') as csvfile:
        fieldnames = all_metrics[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    
    print(f"\nTraining completed!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Metrics saved to: {metrics_csv_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

