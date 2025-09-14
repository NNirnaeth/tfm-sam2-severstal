#!/usr/bin/env python3
"""
Evaluate UNet + SAM2 Pipeline on Segmentation Datasets

This script provides a unified evaluation pipeline that:
1. Uses UNet for initial mask prediction
2. Generates prompts from UNet masks (bbox, center point, random points)
3. Uses SAM2 for refined mask generation from prompts
4. Evaluates final masks against ground truth
5. Supports multiple dataset formats (Supervisely bitmap, COCO)

Workflow:
1. Load image and ground truth annotation
2. Run UNet inference → initial mask
3. Generate prompts from UNet mask → bbox/points
4. Use prompts for SAM2 → refined masks
5. Calculate segmentation metrics (IoU, Dice, etc.)
6. Save results and visualizations

Usage:
    python eval_unet-sam2_pipeline.py --unet_model path/to/unet.pth --sam2_model path/to/sam2.pth --img_dir path/to/images --ann_dir path/to/annotations
"""

import os
import sys
import json
import torch
import numpy as np
import time
import cv2
import random
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
import base64
import zlib
from PIL import Image
from io import BytesIO
from sklearn.metrics import average_precision_score
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import queue
warnings.filterwarnings('ignore')

# SAM2 imports
sys.path.append('libs/sam2base')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.segmentation_metrics import SegmentationMetrics

# UNet imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'unet'))
from model_unet import create_unet_model


def set_seed(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def optimize_gpu_settings():
    """Optimize GPU settings for maximum performance"""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.cuda.set_per_process_memory_fraction(0.90)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        print(f"GPU optimization enabled. Available GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_image_optimized(img_path):
    """Optimized image loading with caching"""
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = np.array(img)
        return image
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def load_annotation_optimized(ann_path, img_shape):
    """Optimized annotation loading"""
    try:
        return load_bitmap_annotation(ann_path, img_shape)
    except Exception as e:
        print(f"Error loading annotation {ann_path}: {e}")
        return None


def preload_batch_data(batch_paths, num_workers=2):
    """Optimized batch data loading with reduced workers"""
    def load_single_item(item):
        img_path, ann_path = item
        image = load_image_optimized(img_path)
        if image is not None:
            gt_mask = load_annotation_optimized(ann_path, image.shape[:2])
            if gt_mask is not None:
                return (image, gt_mask, img_path, ann_path)
        return None
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_single_item, batch_paths))
    
    return [item for item in results if item is not None]


class VisualizationSaver:
    """Async visualization saver to avoid blocking inference"""
    
    def __init__(self, max_workers=2):
        self.queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        """Worker thread that processes visualization queue"""
        while self.running:
            try:
                task = self.queue.get(timeout=1.0)
                if task is None:
                    break
                
                image, gt_mask, unet_mask, prompts, pred_mask, vis_path, image_name = task
                save_visualization(image, gt_mask, unet_mask, prompts, pred_mask, vis_path, image_name)
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving visualization: {e}")
                self.queue.task_done()
    
    def save_async(self, image, gt_mask, unet_mask, prompts, pred_mask, vis_path, image_name):
        """Add visualization to save queue"""
        if self.running:
            self.queue.put((image, gt_mask, unet_mask, prompts, pred_mask, vis_path, image_name))
    
    def shutdown(self):
        """Shutdown the visualization saver"""
        self.running = False
        self.queue.put(None)
        self.worker_thread.join(timeout=2.0)
        if self.worker_thread.is_alive():
            print("Warning: Visualization thread still running, forcing shutdown")
        self.executor.shutdown(wait=False)


def load_unet_model(model_ckpt_path, arch='unet', encoder='resnet34', device='cuda'):
    """Load UNet model from checkpoint"""
    print(f"Loading UNet model from: {model_ckpt_path}")
    print(f"Architecture: {arch}, Encoder: {encoder}")
    
    # Create model
    if arch == 'unet':
        model = create_unet_model(encoder_name=encoder, unet_plus_plus=False)
    elif arch == 'unetpp':
        model = create_unet_model(encoder_name=encoder, unet_plus_plus=True)
    elif arch == 'dsc_unet':
        model = create_unet_model(encoder_name=encoder, unet_plus_plus=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Load checkpoint
    checkpoint = torch.load(model_ckpt_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"UNet model loaded successfully")
    return model


def load_sam2_model(checkpoint_path, model_type="sam2_hiera_l", device='cuda'):
    """Load SAM2 model from checkpoint"""
    print(f"Loading SAM2 model from: {checkpoint_path}")
    print(f"Model type: {model_type}")
    
    config_file = f"configs/sam2/{model_type}.yaml"
    
    sam2 = build_sam2(config_file=config_file, ckpt_path=checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2)
    
    print(f"SAM2 model loaded successfully")
    return sam2, predictor


def load_bitmap_annotation(ann_path, target_shape):
    """Load bitmap annotation from Supervisely format JSON"""
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        if 'objects' not in data or not data['objects']:
            return np.zeros(target_shape, dtype=np.uint8)
        
        height = data['size']['height']
        width = data['size']['width']
        
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        for obj in data['objects']:
            if 'bitmap' not in obj or 'data' not in obj['bitmap']:
                continue
            
            bmp = obj['bitmap']
            
            compressed_data = base64.b64decode(bmp['data'])
            decompressed_data = zlib.decompress(compressed_data)
            
            png_image = Image.open(BytesIO(decompressed_data))
            
            if png_image.mode == 'RGBA':
                crop = np.array(png_image.split()[-1])
            else:
                crop = np.array(png_image.convert('L'))
            
            crop_bin = (crop > 0).astype(np.uint8)
            
            x0, y0 = bmp.get('origin', [0, 0])
            h, w = crop_bin.shape[:2]
            
            x1, y1 = min(x0 + w, width), min(y0 + h, height)
            
            if x1 > x0 and y1 > y0:
                full_mask[y0:y1, x0:x1] = np.maximum(
                    full_mask[y0:y1, x0:x1],
                    crop_bin[:y1-y0, :x1-x0]
                )
        
        if full_mask.shape != target_shape:
            mask_pil = Image.fromarray(full_mask * 255)
            mask_pil = mask_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            full_mask = (np.array(mask_pil) > 0).astype(np.uint8)
        
        return full_mask
        
    except Exception as e:
        print(f"Error loading GT from {ann_path}: {e}")
        return np.zeros(target_shape, dtype=np.uint8)


def load_test_data(img_dir, ann_dir, max_samples=None):
    """Load test data from image and annotation directories"""
    print(f"Loading test data from: {img_dir}")
    
    test_data = []
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        
        ann_path = None
        for ext in ['.json', '.jpg.json']:
            potential_ann = os.path.join(ann_dir, img_file + ext)
            if os.path.exists(potential_ann):
                ann_path = potential_ann
                break
        
        if ann_path and os.path.exists(ann_path):
            test_data.append((img_path, ann_path))
        else:
            print(f"Warning: No annotation found for {img_file}")
    
    print(f"Found {len(test_data)} valid image-annotation pairs")
    
    if max_samples and len(test_data) > max_samples:
        print(f"Limiting to {max_samples} samples")
        test_data = random.sample(test_data, max_samples)
    
    return test_data


def run_unet_inference(model, image, device, target_size=(1024, 256)):
    """Run UNet inference on single image"""
    # Store original image dimensions
    original_h, original_w = image.shape[:2]
    
    # Resize image to target size
    image_resized = cv2.resize(image, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor format
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    with torch.no_grad():
        output = model(image_tensor.to(device))
        prediction = torch.sigmoid(output)
    
    # Convert to numpy and resize back to original size
    pred_np = prediction.cpu().numpy()[0, 0]
    pred_resized = cv2.resize(pred_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    
    return pred_resized


def generate_prompts_from_mask(mask, prompt_type='bbox', num_random_points=5, threshold=0.5):
    """Generate prompts from UNet mask"""
    binary_mask = (mask > threshold).astype(np.uint8)
    
    if np.sum(binary_mask) == 0:
        return []
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    prompts = []
    
    if prompt_type == 'bbox':
        # Generate bounding boxes for each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:  # Filter small regions
                prompts.append([x, y, x + w, y + h])
    
    elif prompt_type == 'center_point':
        # Generate center points for each contour
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                prompts.append([cx, cy])
    
    elif prompt_type == 'random_points':
        # Generate random points within each contour
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                # Generate random points within the contour
                points_in_contour = []
                attempts = 0
                max_attempts = num_random_points * 10
                
                while len(points_in_contour) < num_random_points and attempts < max_attempts:
                    px = random.randint(x, x + w - 1)
                    py = random.randint(y, y + h - 1)
                    
                    # Check if point is inside contour
                    if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                        points_in_contour.append([px, py])
                    
                    attempts += 1
                
                prompts.extend(points_in_contour)
    
    return prompts


def run_sam2_inference(predictor, image, prompts, prompt_type='bbox'):
    """Run SAM2 inference using prompts from UNet mask"""
    if not prompts:
        return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)
    
    # Set image once per image
    predictor.set_image(image)
    
    if prompt_type == 'bbox':
        # Process all bboxes at once
        all_boxes = np.array(prompts)
        
        masks_pred, scores_pred, _ = predictor.predict(
            box=all_boxes,
            multimask_output=False
        )
        
        if len(prompts) == 1:
            masks_pred = [masks_pred]
            scores_pred = [scores_pred]
        
        # Merge masks
        merged_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        merged_prob = np.zeros(image.shape[:2], dtype=np.float32)
        
        for mask, score in zip(masks_pred, scores_pred):
            merged_mask = np.maximum(merged_mask, mask.astype(np.uint8))
            merged_prob = np.maximum(merged_prob, mask.astype(np.float32) * score)
        
        return merged_mask, merged_prob
    
    elif prompt_type in ['center_point', 'random_points']:
        # Process points
        masks = []
        scores = []
        
        for prompt in prompts:
            if len(prompt) == 2:  # Single point
                input_point = np.array([prompt])
                input_label = np.array([1])
            else:  # Multiple points
                input_point = np.array(prompt)
                input_label = np.array([1] * len(prompt))
            
            masks_pred, scores_pred, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            
            masks.append(masks_pred)
            scores.append(scores_pred)
        
        # Merge masks
        if masks:
            merged_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            merged_prob = np.zeros(image.shape[:2], dtype=np.float32)
            
            for mask, score in zip(masks, scores):
                merged_mask = np.maximum(merged_mask, mask.astype(np.uint8))
                merged_prob = np.maximum(merged_prob, mask.astype(np.float32) * score)
            
            return merged_mask, merged_prob
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)


def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """Calculate segmentation metrics"""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    
    intersection = np.sum(pred_binary * gt_mask)
    union = np.sum(pred_binary) + np.sum(gt_mask) - intersection
    
    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (np.sum(pred_binary) + np.sum(gt_mask) + 1e-8)
    
    tp = intersection
    fp = np.sum(pred_binary) - intersection
    fn = np.sum(gt_mask) - intersection
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_visualization(image, gt_mask, unet_mask, prompts, pred_mask, output_path, image_name):
    """Save visualization with original, GT, UNet, prompts, and SAM2 prediction"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Ensure all masks have the same shape as the image
        h, w = image.shape[:2]
        
        # Validate and fix mask dimensions
        if gt_mask is None or gt_mask.size == 0:
            gt_mask = np.zeros((h, w), dtype=np.uint8)
        elif gt_mask.shape != (h, w):
            if len(gt_mask.shape) > 2:
                gt_mask = gt_mask.squeeze()
            if gt_mask.size > 0 and min(gt_mask.shape) > 0:
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                gt_mask = np.zeros((h, w), dtype=np.uint8)
        
        if unet_mask is None or unet_mask.size == 0:
            unet_mask = np.zeros((h, w), dtype=np.float32)
        elif unet_mask.shape != (h, w):
            if len(unet_mask.shape) > 2:
                unet_mask = unet_mask.squeeze()
            if unet_mask.size > 0 and min(unet_mask.shape) > 0:
                unet_mask = cv2.resize(unet_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                unet_mask = np.zeros((h, w), dtype=np.float32)
        
        if pred_mask is None or pred_mask.size == 0:
            pred_mask = np.zeros((h, w), dtype=np.uint8)
        elif pred_mask.shape != (h, w):
            if len(pred_mask.shape) > 2:
                pred_mask = pred_mask.squeeze()
            if pred_mask.size > 0 and min(pred_mask.shape) > 0:
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                pred_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(image)
        axes[0, 1].imshow(gt_mask, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # UNet prediction
        axes[0, 2].imshow(image)
        axes[0, 2].imshow(unet_mask, cmap='Greens', alpha=0.7, vmin=0, vmax=1)
        axes[0, 2].set_title('UNet Prediction')
        axes[0, 2].axis('off')
        
        # Prompts visualization
        axes[1, 0].imshow(image)
        if prompts:
            if len(prompts) > 0 and len(prompts[0]) == 4:  # Bbox
                for i, bbox in enumerate(prompts):
                    x1, y1, x2, y2 = bbox
                    # Ensure bbox coordinates are within image bounds
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    if x2 > x1 and y2 > y1:
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                               linewidth=2, edgecolor='blue', facecolor='none')
                        axes[1, 0].add_patch(rect)
                        axes[1, 0].text(x1, y1-5, f'{i+1}', fontsize=8, color='blue',
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            elif len(prompts) > 0 and len(prompts[0]) == 2:  # Points
                for i, point in enumerate(prompts):
                    if len(point) == 2:
                        px, py = int(point[0]), int(point[1])
                        # Ensure point is within image bounds
                        if 0 <= px < w and 0 <= py < h:
                            axes[1, 0].plot(px, py, 'bo', markersize=8)
                            axes[1, 0].text(px, py-10, f'{i+1}', fontsize=8, color='blue',
                                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        axes[1, 0].set_title(f'Generated Prompts ({len(prompts)})')
        axes[1, 0].axis('off')
        
        # SAM2 prediction
        axes[1, 1].imshow(image)
        axes[1, 1].imshow(pred_mask, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
        axes[1, 1].set_title('SAM2 Refined Prediction')
        axes[1, 1].axis('off')
        
        # Comparison
        axes[1, 2].imshow(image)
        axes[1, 2].imshow(gt_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        axes[1, 2].imshow(pred_mask, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        axes[1, 2].set_title('GT (Red) vs SAM2 (Blue)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in visualization for {image_name}: {e}")
        # Create a simple error visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Visualization Error\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Error: {image_name}')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def evaluate_pipeline(unet_model, sam2_predictor, test_data, device, 
                     prompt_type='bbox', threshold=0.5, num_random_points=5,
                     save_visualizations=False, output_dir=None, max_samples=None, 
                     batch_size=4, num_workers=4):
    """Evaluate the UNet+SAM2 pipeline"""
    print(f"Evaluating pipeline with {len(test_data)} images...")
    print(f"Prompt type: {prompt_type}")
    print(f"Threshold: {threshold}")
    print(f"Number of random points: {num_random_points}")
    print(f"Batch size: {batch_size}")
    
    all_metrics = []
    inference_times = []
    
    # Create output directory for visualizations
    vis_saver = None
    if save_visualizations and output_dir:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        vis_saver = VisualizationSaver(max_workers=2)
        print("Async visualization saver initialized")
    
    for batch_idx in tqdm(range(0, len(test_data), batch_size), desc="Processing batches"):
        batch_data = test_data[batch_idx:batch_idx + batch_size]
        
        # Pre-load batch data
        batch_items = preload_batch_data(batch_data, num_workers)
        
        if not batch_items:
            continue
            
        batch_images = [item[0] for item in batch_items]
        batch_gt_masks = [item[1] for item in batch_items]
        batch_paths = [(item[2], item[3]) for item in batch_items]
        
        # Process each image
        for idx, (image, gt_mask, (img_path, ann_path)) in enumerate(zip(batch_images, batch_gt_masks, batch_paths)):
            try:
                batch_start_time = time.time()
                
                # Step 1: UNet inference
                unet_mask = run_unet_inference(unet_model, image, device)
                
                # Step 2: Generate prompts from UNet mask
                prompts = generate_prompts_from_mask(unet_mask, prompt_type, num_random_points, threshold)
                
                # Step 3: SAM2 inference
                pred_mask, pred_prob = run_sam2_inference(sam2_predictor, image, prompts, prompt_type)
                
                # Calculate metrics
                metrics = calculate_metrics(pred_mask, gt_mask, threshold)
                all_metrics.append(metrics)
                
                # Record inference time
                inference_time = time.time() - batch_start_time
                inference_times.append(inference_time)
                
                # Save visualization asynchronously (only first 10 for speed)
                if save_visualizations and output_dir and (batch_idx + idx) < 10:
                    image_name = os.path.basename(img_path).split('.')[0]
                    vis_path = os.path.join(vis_dir, f"{image_name}_unet_sam2_pipeline.jpg")
                    # Add debugging info for problematic cases
                    if gt_mask is None or unet_mask is None or pred_mask is None:
                        print(f"Warning: None mask detected for {image_name} - gt:{gt_mask is not None}, unet:{unet_mask is not None}, pred:{pred_mask is not None}")
                    elif gt_mask.size == 0 or unet_mask.size == 0 or pred_mask.size == 0:
                        print(f"Warning: Empty mask detected for {image_name} - gt:{gt_mask.size}, unet:{unet_mask.size}, pred:{pred_mask.size}")
                    vis_saver.save_async(image, gt_mask, unet_mask, prompts, pred_mask, vis_path, image_name)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Clear GPU cache less frequently
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        avg_metrics['avg_inference_time'] = np.mean(inference_times)
        avg_metrics['total_inference_time'] = np.sum(inference_times)
        avg_metrics['num_images'] = len(all_metrics)
        
        # Shutdown visualization saver
        if vis_saver:
            print("Shutting down visualization saver...")
            vis_saver.shutdown()
        
        return avg_metrics, all_metrics
    else:
        if vis_saver:
            vis_saver.shutdown()
        return None, []


def save_results(metrics, output_dir, model_name):
    """Save evaluation results to CSV"""
    if not metrics:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_row = {
        'model': model_name,
        'test_mIoU': metrics['iou'],
        'test_Dice': metrics['dice'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'avg_inference_time': metrics['avg_inference_time'],
        'total_inference_time': metrics['total_inference_time'],
        'num_images': metrics['num_images'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    csv_path = os.path.join(output_dir, f"results_{model_name}.csv")
    df = pd.DataFrame([results_row])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return csv_path


def print_results(metrics):
    """Print evaluation results"""
    if not metrics:
        print("No results to display")
        return
    
    print("\n" + "="*60)
    print("UNET+SAM2 PIPELINE EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nSegmentation Metrics:")
    print(f"  mIoU:           {metrics['iou']:.4f}")
    print(f"  mDice:          {metrics['dice']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1:             {metrics['f1']:.4f}")
    
    print(f"\nPerformance:")
    print(f"  Number of images:   {metrics['num_images']}")
    print(f"  Avg inference time: {metrics['avg_inference_time']:.4f}s")
    print(f"  Total time:         {metrics['total_inference_time']:.2f}s")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate UNet + SAM2 Pipeline')
    
    # Model paths
    parser.add_argument('--unet_model', type=str, required=True,
                       help='Path to UNet model checkpoint')
    parser.add_argument('--sam2_model', type=str, required=True,
                       help='Path to SAM2 model checkpoint')
    parser.add_argument('--unet_arch', type=str, default='unet',
                       choices=['unet', 'unetpp', 'dsc_unet'],
                       help='UNet architecture')
    parser.add_argument('--unet_encoder', type=str, default='resnet50',
                       choices=['resnet34', 'resnet50'],
                       help='UNet encoder backbone')
    parser.add_argument('--sam2_model_type', type=str, default='sam2_hiera_l',
                       choices=['sam2_hiera_l', 'sam2_hiera_b', 'sam2_hiera_t'],
                       help='SAM2 model type')
    
    # Data paths
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Path to test images directory')
    parser.add_argument('--ann_dir', type=str, required=True,
                       help='Path to test annotations directory')
    
    # Evaluation settings
    parser.add_argument('--prompt_type', type=str, default='bbox',
                       choices=['bbox', 'center_point', 'random_points'],
                       help='Prompt type generated from UNet mask')
    parser.add_argument('--num_random_points', type=int, default=5,
                       help='Number of random points when using random_points prompt type')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary threshold for mask evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./unet_sam2_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default='unet_sam2_pipeline',
                       help='Model name for output files')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save prediction visualizations')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Optimize GPU settings
    optimize_gpu_settings()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    unet_model = load_unet_model(args.unet_model, args.unet_arch, args.unet_encoder, device)
    sam2_model, sam2_predictor = load_sam2_model(args.sam2_model, args.sam2_model_type, device)
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(args.img_dir, args.ann_dir, args.max_samples)
    
    if not test_data:
        print("No test data found!")
        return
    
    # Evaluate pipeline
    print(f"\nEvaluating {args.model_name}...")
    metrics, per_image_metrics = evaluate_pipeline(
        unet_model=unet_model,
        sam2_predictor=sam2_predictor,
        test_data=test_data,
        device=device,
        prompt_type=args.prompt_type,
        threshold=args.threshold,
        num_random_points=args.num_random_points,
        save_visualizations=args.save_visualizations,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Print and save results
    print_results(metrics)
    if metrics:
        csv_path = save_results(metrics, args.output_dir, args.model_name)
        
        # Save detailed results as JSON
        json_path = os.path.join(args.output_dir, f"detailed_results_{args.model_name}.json")
        results_json = {
            'summary_metrics': metrics,
            'per_image_metrics': per_image_metrics,
            'config': vars(args)
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {csv_path}")
        print(f"Detailed results saved to: {json_path}")
        
        if args.save_visualizations:
            vis_dir = os.path.join(args.output_dir, "visualizations")
            print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
