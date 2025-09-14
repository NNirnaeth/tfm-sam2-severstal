#!/usr/bin/env python3
"""
Evaluate SAM2 models with optional LoRA on Severstal test split (2000 images).

This script evaluates SAM2 models with flexible backbone and LoRA combinations:
- Backbone can be original Meta SAM2 (.pt) or fine-tuned SAM2 (.torch)
- LoRA adapters are optional and applied when checkpoint is provided
- Always uses LoRA structure when LoRA checkpoint is available

Supported combinations:
- SAM2-Base: Original Meta SAM2 without LoRA
- SAM2-FT: Fine-tuned SAM2 without LoRA  
- SAM2-Base+LoRA: Original Meta SAM2 with LoRA adapters
- SAM2-FT+LoRA: Fine-tuned SAM2 with LoRA adapters

Supported evaluation modes:
1. Auto-prompt: SAM2-AutoPrompt (no GT, no external detector) - for reference/experiment
2. GT Points: SAM2 with 1-3 GT points (centroid + max distance + negative point)
3. 30 GT Points: SAM2 with 30 GT points (for comparison with previous results)

Test data: datasets/Data/splits/test_split (2000 images)
Evaluation metrics: mIoU, IoU@50, IoU@75, IoU@90, IoU@95, Dice, Precision, Recall, F1, AUPRC
Results saved in CSV format under data/results/ for comparative analysis.

Usage examples:
- Base only: --backbone_ckpt models/base/sam2/sam2_hiera_large.pt --evaluation_mode auto_prompt
- Base+LoRA: --backbone_ckpt models/base/sam2/sam2_hiera_large.pt --lora_checkpoint /path/to/lora.torch
- FT+LoRA: --backbone_ckpt /path/to/ft_model.torch --lora_checkpoint /path/to/lora.torch

"""

import os
import sys
import json
import torch
import numpy as np
import time
import random
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import cv2
from PIL import Image
import base64
import zlib
import io
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('/home/ptp/sam2/libs/sam2base')
sys.path.append('/home/ptp/sam2/new_src/utils')  # To access metrics.py
from metrics import SegmentationMetrics

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Constants for hyperparameters (ensure consistency)
POINT_BATCH_SIZE = 4
BOX_BATCH_SIZE = 8
GRID_SPACING = 128
BOX_SCALES = [128, 256]
CONFIDENCE_THRESHOLD = 0.7  # Less strict than 0.8
NMS_IOU_THRESHOLD = 0.65   # Less strict than 0.8
TOP_K_FILTER = 200

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LoRALayer(torch.nn.Module):
    """LoRA layer implementation (same as training script)"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout)
        
        # Initialize B with zeros for stable training
        torch.nn.init.zeros_(self.lora_B)
        
        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        # Ensure LoRA matrices are on the same device as input
        device = x.device
        lora_A = self.lora_A.to(device)
        lora_B = self.lora_B.to(device)
        
        # Apply LoRA: output = B @ A @ x * scaling
        if x.dim() > 2:
            # For batched inputs, reshape to 2D for matrix multiplication
            batch_shape = x.shape[:-1]
            x_2d = x.reshape(-1, x.shape[-1])
            lora_output = self.dropout(lora_B @ (lora_A @ x_2d.T)).T
            lora_output = lora_output.reshape(*batch_shape, -1) * self.scaling
        else:
            # For 2D inputs, use direct matrix multiplication
            lora_output = self.dropout(lora_B @ (lora_A @ x.T)).T * self.scaling
        return lora_output

class LoRALinear(torch.nn.Module):
    """Linear layer with LoRA adaptation (same as training script)"""
    
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features, 
            linear_layer.out_features, 
            rank, alpha, dropout
        )
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original output + LoRA adaptation
        original_output = self.linear(x)
        lora_output = self.lora(x)
        return original_output + lora_output

def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.05):
    """Apply LoRA adapters to SAM2 model (same as training script)"""
    lora_params = []
    
    # Apply LoRA only to mask decoder attention projections
    if hasattr(model, 'sam_mask_decoder'):
        decoder = model.sam_mask_decoder
        for layer_name, layer in decoder.named_modules():
            # Only apply to attention layers with q_proj attribute
            if 'attn' in layer_name and hasattr(layer, 'q_proj'):
                try:
                    # Apply LoRA to Q, K, V, O projections
                    layer.q_proj = LoRALinear(layer.q_proj, rank, alpha, dropout)
                    layer.k_proj = LoRALinear(layer.k_proj, rank, alpha, dropout)
                    layer.v_proj = LoRALinear(layer.v_proj, rank, alpha, dropout)
                    layer.out_proj = LoRALinear(layer.out_proj, rank, alpha, dropout)
                    
                    # Collect LoRA parameters
                    lora_params.extend(list(layer.q_proj.lora.parameters()))
                    lora_params.extend(list(layer.k_proj.lora.parameters()))
                    lora_params.extend(list(layer.v_proj.lora.parameters()))
                    lora_params.extend(list(layer.out_proj.lora.parameters()))
                    
                except Exception as e:
                    print(f"Warning: Could not apply LoRA to {layer_name}: {e}")
                    continue
    
    return model, lora_params

def load_backbone_with_optional_lora(backbone_ckpt, config_path, lora_ckpt=None, device="cuda"):
    """Load backbone (base or fine-tuned) with optional LoRA adapters"""
    print(f"Loading backbone: {backbone_ckpt}")
    
    # Build backbone (base or FT, doesn't matter the origin)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    sam2_model = build_sam2(config_name, backbone_ckpt, device=device)
    
    # If there's LoRA, wrap projections and load adapters
    if lora_ckpt is not None:
        print("Applying LoRA structure...")
        sam2_model, _ = apply_lora_to_model(sam2_model, rank=8, alpha=16, dropout=0.05)
        
        print(f"Loading LoRA checkpoint: {lora_ckpt}")
        sd = torch.load(lora_ckpt, map_location=device)
        missing_keys, unexpected_keys = sam2_model.load_state_dict(sd, strict=False)
        
        if missing_keys:
            print(f"Missing keys (expected for base model): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print(f"Backbone + LoRA model loaded successfully")
    else:
        print(f"Backbone model loaded successfully (no LoRA)")
    
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def get_binary_gt_mask(ann_path, target_shape):
    """Load and combine all defect objects into a single binary mask from PNG bitmap"""
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        if 'objects' not in data or not data['objects']:
            return None
        
        # Get image dimensions
        height = data['size']['height']
        width = data['size']['width']
        
        # Create empty mask
        full_mask = np.zeros((height, width), dtype=bool)
        
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
            
            # Convert to binary mask
            binary_mask = mask > 0
            
            # Resize to full image size
            mask_pil = Image.fromarray(binary_mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((width, height), Image.NEAREST)
            obj_mask = np.array(mask_pil) > 0
            
            # Combine with full mask
            full_mask = np.logical_or(full_mask, obj_mask)
        
        # Resize to target shape if needed
        if full_mask.shape != target_shape:
            mask_pil = Image.fromarray(full_mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            full_mask = np.array(mask_pil) > 0
        
        return full_mask
        
    except Exception as e:
        print(f"Error loading GT from {ann_path}: {e}")
        return None

def generate_auto_prompts(image, grid_spacing=GRID_SPACING, box_scales=BOX_SCALES):
    """Generate automatic prompts from image without GT information (controlled complexity)"""
    height, width = image.shape[:2]
    
    # Strategy 1: Grid-based point sampling (limited density)
    grid_points = []
    grid_labels = []
    
    # Sample points in a regular grid with controlled spacing
    for y in range(grid_spacing, height - grid_spacing, grid_spacing):
        for x in range(grid_spacing, width - grid_spacing, grid_spacing):
            grid_points.append([x, y])
            grid_labels.append(1)  # All points are positive prompts
    
    # Strategy 2: Multi-scale box sweeping (limited scales and density)
    auto_boxes = []
    for scale in box_scales:
        stride = scale  # Full scale stride to reduce density
        for y in range(0, height - scale, stride):
            for x in range(0, width - scale, stride):
                # Ensure box is within image boundaries
                if x + scale <= width and y + scale <= height:
                    auto_boxes.append([x, y, x + scale, y + scale])
    
    return {
        'points': np.array(grid_points) if grid_points else np.zeros((0, 2)),
        'point_labels': np.array(grid_labels) if grid_labels else np.zeros((0,)),
        'boxes': np.array(auto_boxes) if auto_boxes else np.zeros((0, 4))
    }

def generate_gt_points(gt_mask, num_points=3):
    """Generate GT points from ground truth mask (centroid + max distance + negative point)"""
    from scipy.ndimage import label, distance_transform_edt
    if gt_mask is None or gt_mask.sum() == 0:
        return np.zeros((0,2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    labeled, n = label(gt_mask)
    if n == 0:
        return np.zeros((0,2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    # componente principal
    sizes = [np.sum(labeled == i) for i in range(1, n+1)]
    comp = np.argmax(sizes) + 1
    main = (labeled == comp).astype(np.uint8)

    pts, lbs = [], []

    # 1) centroide
    ys, xs = np.where(main)
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    pts.append([cx, cy]); lbs.append(1)

    # 2) punto más interior
    if num_points >= 2:
        dist = distance_transform_edt(main)
        my, mx = np.unravel_index(np.argmax(dist), dist.shape)
        if abs(mx-cx) > 3 or abs(my-cy) > 3:
            pts.append([mx, my]); lbs.append(1)

    # 3) punto negativo fuera del borde
    if num_points >= 3:
        k = np.ones((3,3), np.uint8)
        outer = cv2.dilate(main, k, iterations=6)  # 5–10 px aprox
        neg_candidates = np.where((outer > 0) & (main == 0))
        if len(neg_candidates[0]) > 0:
            i = np.random.randint(len(neg_candidates[0]))
            ny, nx = int(neg_candidates[0][i]), int(neg_candidates[1][i])
            pts.append([nx, ny]); lbs.append(0)

    return np.array(pts, dtype=np.float32), np.array(lbs, dtype=np.int32)

def generate_30_gt_points(gt_mask):
    """Generate 30 GT points from ground truth mask (for comparison with previous results)"""
    if gt_mask is None or gt_mask.sum() == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    
    # Find connected components
    from scipy.ndimage import label
    labeled_mask, num_components = label(gt_mask)
    
    if num_components == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    
    # Get the largest component
    component_sizes = []
    for i in range(1, num_components + 1):
        component_sizes.append((labeled_mask == i).sum())
    
    largest_component_idx = np.argmax(component_sizes) + 1
    main_mask = (labeled_mask == largest_component_idx)
    
    # Sample 30 points from the mask
    y_coords, x_coords = np.where(main_mask)
    
    if len(y_coords) < 30:
        # If mask is smaller than 30 pixels, repeat some points
        points = []
        labels = []
        for i in range(30):
            idx = i % len(y_coords)
            points.append([x_coords[idx], y_coords[idx]])
            labels.append(1)
        return np.array(points), np.array(labels)
    
    # Randomly sample 30 points
    indices = np.random.choice(len(y_coords), 30, replace=False)
    points = []
    labels = []
    
    for idx in indices:
        points.append([x_coords[idx], y_coords[idx]])
        labels.append(1)  # All positive points
    
    return np.array(points), np.array(labels)

def compute_mask_iou(mask1, mask2):
    """Compute IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-8)

def apply_nms(masks, scores, iou_threshold=NMS_IOU_THRESHOLD):
    """Apply Non-Maximum Suppression to remove overlapping masks"""
    if len(masks) <= 1:
        return masks, scores
    
    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []
    
    for idx in sorted_indices:
        keep = True
        for kept_idx in keep_indices:
            iou = compute_mask_iou(masks[idx], masks[kept_idx])
            if iou > iou_threshold:
                keep = False
                break
        
        if keep:
            keep_indices.append(idx)
    
    return [masks[i] for i in keep_indices], [scores[i] for i in keep_indices]

def predict_with_auto_prompts(predictor, image, auto_prompts, confidence_threshold=CONFIDENCE_THRESHOLD, 
                             point_batch_size=POINT_BATCH_SIZE, box_batch_size=BOX_BATCH_SIZE):
    """Predict using automatic prompts with proper individual processing and NMS"""
    height, width = image.shape[:2]
    
    all_masks = []
    all_scores = []
    
    # Predict with grid points in very small batches (1-4 points per call)
    if len(auto_prompts['points']) > 0:
        try:
            points = auto_prompts['points']
            labels = auto_prompts['point_labels']
            
            # Process points in very small batches to avoid semantic mixing
            for i in range(0, len(points), point_batch_size):
                batch_points = points[i:i + point_batch_size]
                batch_labels = labels[i:i + point_batch_size]
                
                point_masks, point_scores, _ = predictor.predict(
                    point_coords=batch_points,
                    point_labels=batch_labels,
                    multimask_output=True
                )
                
                # Each batch call returns masks for the entire batch
                # We need to handle this correctly
                if len(batch_points) == 1:
                    # Single point: process normally
                    for j, score in enumerate(point_scores):
                        if score > confidence_threshold:
                            # Ensure binary mask
                            mask = (point_masks[j].astype(np.float32) > 0.5).astype(bool)
                            all_masks.append(mask)
                            all_scores.append(score)
                else:
                    # Multiple points: each mask corresponds to the entire batch
                    # We need to split them properly - this is complex with SAM2
                    # For now, treat as single prediction and use best score
                    best_idx = np.argmax(point_scores)
                    if point_scores[best_idx] > confidence_threshold:
                        mask = (point_masks[best_idx].astype(np.float32) > 0.5).astype(bool)
                        all_masks.append(mask)
                        all_scores.append(point_scores[best_idx])
                        
        except Exception as e:
            print(f"Error with point prompts: {e}")
    
    # Predict with auto boxes individually (one per call)
    if len(auto_prompts['boxes']) > 0:
        try:
            boxes = auto_prompts['boxes']
            
            # Process boxes individually to avoid semantic mixing
            for box in boxes:
                box_masks, box_scores, _ = predictor.predict(
                    box=box,
                    multimask_output=True
                )
                
                # Filter by confidence and add to collection
                for j, score in enumerate(box_scores):
                    if score > confidence_threshold:
                        # Ensure binary mask
                        mask = (box_masks[j].astype(np.float32) > 0.5).astype(bool)
                        all_masks.append(mask)
                        all_scores.append(score)
                        
        except Exception as e:
            print(f"Error with box prompts: {e}")
    
    if not all_masks:
        return np.zeros((height, width), dtype=bool), 0
    
    # Apply top-K filtering before NMS to control memory
    original_candidates = len(all_masks)
    if len(all_masks) > TOP_K_FILTER:
        # Sort by score and keep top K
        sorted_indices = np.argsort(all_scores)[::-1][:TOP_K_FILTER]
        all_masks = [all_masks[i] for i in sorted_indices]
        all_scores = [all_scores[i] for i in sorted_indices]
        print(f"Applied top-{TOP_K_FILTER} filtering: kept {len(all_masks)} masks from {original_candidates} candidates")
    
    # Apply NMS to remove overlapping masks
    filtered_masks, filtered_scores = apply_nms(all_masks, all_scores)
    
    if not filtered_masks:
        return np.zeros((height, width), dtype=bool), 0
    
    # Combine masks using logical OR (more robust than averaging)
    combined_mask = np.zeros((height, width), dtype=bool)
    for mask in filtered_masks:
        # Ensure binary before OR operation
        binary_mask = mask.astype(bool)
        combined_mask = np.logical_or(combined_mask, binary_mask)
    
    return combined_mask, len(filtered_masks)

def predict_with_gt_points(predictor, image, gt_points, gt_labels, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Predict using GT points (1-3 or 30 points)"""
    height, width = image.shape[:2]
    
    if len(gt_points) == 0:
        return np.zeros((height, width), dtype=bool), 0
    
    try:
        # For GT points, we can use all points at once since they're well-distributed
        point_masks, point_scores, _ = predictor.predict(
            point_coords=gt_points,
            point_labels=gt_labels,
            multimask_output=True
        )
        
        # Select the best mask based on score
        best_idx = np.argmax(point_scores)
        best_score = point_scores[best_idx]
        
        if best_score > confidence_threshold:
            pred_mask = (point_masks[best_idx].astype(np.float32) > 0.5).astype(bool)
            return pred_mask, 1
        else:
            return np.zeros((height, width), dtype=bool), 0
            
    except Exception as e:
        print(f"Error with GT point prompts: {e}")
        return np.zeros((height, width), dtype=bool), 0

def load_image_and_mask(img_path, ann_path):
    """Load image and corresponding binary mask (same preprocessing as training)"""
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load annotation
    mask = get_binary_gt_mask(ann_path, (image.shape[0], image.shape[1]))
    
    # Apply same resizing as training
    r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
    image_resized = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
    mask_resized = cv2.resize(mask, (image_resized.shape[1], image_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return image_resized, mask_resized

def calculate_metrics(pred_mask, gt_mask):
    """Calculate all evaluation metrics"""
    # Ensure binary masks
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Intersection and Union
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    # IoU
    iou = intersection / (union + 1e-8)
    
    # Pixel-level metrics
    tp = intersection
    fp = np.sum(pred_binary) - intersection
    fn = np.sum(gt_binary) - intersection
    
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
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    for th in thresholds:
        # Apply threshold to probability map
        pred_thresholded = (pred_prob > th).astype(np.uint8)
        
        # Calculate IoU at this threshold
        intersection = np.sum(pred_thresholded * gt_binary)
        union = np.sum(pred_thresholded) + np.sum(gt_binary) - intersection
        iou_th = intersection / (union + 1e-8)
        
        results[f'iou_{int(th*100)}'] = float(iou_th)
    
    return results

def evaluate_lora_sam2(predictor, test_data, threshold=0.5, save_examples=False, output_dir=None, 
                      evaluation_mode="auto_prompt", num_gt_points=3):
    """Evaluate LoRA SAM2 model on test data using different evaluation modes"""
    predictor.model.eval()
    
    all_metrics = []
    all_iou_thresholds = []
    inference_times = []
    
    # For AUPRC calculation (sample to save memory)
    sample_size = min(200, len(test_data))
    sample_indices = random.sample(range(len(test_data)), sample_size)
    sample_preds = []
    sample_targets = []
    
    print(f"Evaluating LoRA SAM2 on {len(test_data)} test images...")
    
    # Set evaluation mode description
    if evaluation_mode == "auto_prompt":
        print(f"Evaluation mode: SAM2-AutoPrompt (no GT, no external detector)")
    elif evaluation_mode == "gt_points":
        print(f"Evaluation mode: SAM2 with {num_gt_points} GT points (centroid + max distance + negative)")
    elif evaluation_mode == "30_gt_points":
        print(f"Evaluation mode: SAM2 with 30 GT points (for comparison with previous results)")
    
    with torch.no_grad():
        for idx, (img_path, ann_path) in enumerate(tqdm(test_data, desc="Evaluating")):
            try:
                # Load image and mask
                image, mask = load_image_and_mask(img_path, ann_path)
                
                # Generate prompts based on evaluation mode
                if evaluation_mode == "auto_prompt":
                    # Generate automatic prompts from image (no GT)
                    auto_prompts = generate_auto_prompts(image)
                    gt_points, gt_labels = None, None
                elif evaluation_mode == "gt_points":
                    # Generate 1-3 GT points
                    gt_points, gt_labels = generate_gt_points(mask, num_gt_points)
                    auto_prompts = None
                elif evaluation_mode == "30_gt_points":
                    # Generate 30 GT points
                    gt_points, gt_labels = generate_30_gt_points(mask)
                    auto_prompts = None
                
                # Measure inference time
                start_time = time.time()
                
                # Set image and predict based on mode
                predictor.set_image(image)
                
                # Predict based on evaluation mode
                if evaluation_mode == "auto_prompt":
                    pred_mask, num_masks = predict_with_auto_prompts(predictor, image, auto_prompts)
                else:
                    pred_mask, num_masks = predict_with_gt_points(predictor, image, gt_points, gt_labels)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Convert to probability map if needed
                if pred_mask.dtype == bool:
                    pred_prob = pred_mask.astype(np.float32)
                else:
                    pred_prob = pred_mask.astype(np.float32)
                
                # Ensure prediction matches ground truth size
                if pred_prob.shape != mask.shape:
                    pred_prob = cv2.resize(pred_prob, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # Calculate metrics
                metrics = calculate_metrics(pred_prob, mask)
                iou_thresholds = calculate_iou_at_thresholds(pred_prob, mask)
                
                # Combine all metrics
                combined_metrics = {**metrics, **iou_thresholds}
                all_metrics.append(combined_metrics)
                all_iou_thresholds.append(iou_thresholds)
                
                # Store sample for AUPRC
                if idx in sample_indices:
                    sample_preds.append(pred_prob.flatten())
                    sample_targets.append(mask.flatten())
                
                # Save examples if requested
                if save_examples and output_dir and idx < 20:
                    save_prediction_example(pred_prob, mask, img_path, output_dir, idx)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Calculate AUPRC from sample
        if sample_preds and sample_targets:
            all_preds_flat = np.concatenate(sample_preds)
            all_targets_flat = np.concatenate(sample_targets)
            auprc = average_precision_score(all_targets_flat, all_preds_flat)
            avg_metrics['auprc'] = auprc
        else:
            avg_metrics['auprc'] = 0.0
        
        # Add performance metrics
        avg_metrics['avg_inference_time'] = np.mean(inference_times)
        avg_metrics['total_inference_time'] = np.sum(inference_times)
        
        return avg_metrics, all_metrics
    else:
        return None, []

def save_prediction_example(pred, target, img_path, output_dir, idx):
    """Save prediction example for visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to match prediction
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img_resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
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
    filename = f"lora_sam2_pred_{idx:04d}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def save_results_csv(results, model_info, output_dir="new_src/evaluation/evaluation_results/sam2_lora", 
                    evaluation_mode="auto_prompt", num_gt_points=3):
    """Save evaluation results to CSV following TFM format"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    # Prepare prompt type based on evaluation mode
    if evaluation_mode == "auto_prompt":
        prompt_type = "sam2_autoprompt"
    elif evaluation_mode == "gt_points":
        prompt_type = f"gt_points_{num_gt_points}"
    elif evaluation_mode == "30_gt_points":
        prompt_type = "gt_points_30"
    
    # Prepare model name and variant based on model label
    model_label = model_info['model_label']
    if model_label == 'SAM2-Base':
        variant = "base_original_meta"
        exp_prefix = "exp_base_sam2"
    elif model_label == 'SAM2-FT':
        variant = "fine_tuned_no_lora"
        exp_prefix = "exp_ft_sam2"
    elif model_label == 'SAM2-Base+LoRA':
        variant = f"base_lora_r{model_info['rank']}_a{model_info['alpha']}_lr{model_info['lr']}"
        exp_prefix = "exp_base_lora_sam2"
    elif model_label == 'SAM2-FT+LoRA':
        variant = f"ft_lora_r{model_info['rank']}_a{model_info['alpha']}_lr{model_info['lr']}"
        exp_prefix = "exp_ft_lora_sam2"
    
    filename = f"{exp_prefix}_{prompt_type}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for CSV
    data = {
        'exp_id': f"{exp_prefix}_{prompt_type}_{timestamp}",
        'model': model_label,
        'subset_size': 'full_dataset',
        'variant': variant,
        'prompt_type': prompt_type,
        'img_size': '1024xAuto',
        'batch_size': 1,
        'steps': model_info.get('steps', 'N/A'),
        'lr': model_info['lr'],
        'wd': model_info.get('weight_decay', 'N/A'),
        'seed': 42,
        'val_mIoU': 'N/A',
        'val_Dice': 'N/A',
        'test_mIoU': results['iou'],
        'test_Dice': results['dice'],
        'IoU@50': results.get('iou_50', 'N/A'),
        'IoU@75': results.get('iou_75', 'N/A'),
        'IoU@90': results.get('iou_90', 'N/A'),
        'IoU@95': results.get('iou_95', 'N/A'),
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1': results['f1'],
        'AUPRC': results.get('auprc', 'N/A'),
        'avg_inference_time': results.get('avg_inference_time', 'N/A'),
        'backbone_ckpt': model_info['backbone_ckpt'],
        'lora_ckpt': model_info['lora_ckpt'],
        'timestamp': timestamp
    }
    
    # Create DataFrame and save
    df = pd.DataFrame([data])
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM2 with optional LoRA on Severstal test split')
    
    # Model configuration
    parser.add_argument('--backbone_ckpt', type=str, required=True,
                       help='Path to backbone checkpoint (Meta .pt or fine-tuned .torch)')
    parser.add_argument('--config', type=str,
                       default='libs/sam2base/sam2/configs/sam2/sam2_hiera_l.yaml',
                       help='Path to SAM2 config file')
    parser.add_argument('--lora_checkpoint', type=str, default=None,
                       help='Path to LoRA checkpoint (optional)')
    
    # Data paths
    parser.add_argument('--test_dir', type=str,
                       default='datasets/Data/splits/test_split',
                       help='Path to test data directory')
    
    # Evaluation settings
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary threshold for predictions')
    parser.add_argument('--save_examples', action='store_true',
                       help='Save prediction examples')
    parser.add_argument('--output_dir', type=str, default='new_src/evaluation/evaluation_results/sam2_lora',
                       help='Output directory for results')
    parser.add_argument('--evaluation_mode', type=str, 
                       choices=['auto_prompt', 'gt_points', '30_gt_points'],
                       default='auto_prompt',
                       help='Evaluation mode: auto_prompt (no GT), gt_points (1-3 GT points), or 30_gt_points (30 GT points)')
    parser.add_argument('--num_gt_points', type=int, default=3,
                       help='Number of GT points to use when evaluation_mode is gt_points (1-3)')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.evaluation_mode == 'gt_points' and (args.num_gt_points < 1 or args.num_gt_points > 3):
        print("Error: num_gt_points must be between 1 and 3 when evaluation_mode is gt_points")
        sys.exit(1)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"SAM2 Evaluation Script (with optional LoRA)")
    print("=" * 50)
    print(f"Backbone checkpoint: {args.backbone_ckpt}")
    if args.lora_checkpoint:
        print(f"LoRA checkpoint: {args.lora_checkpoint}")
    else:
        print("LoRA checkpoint: None (using backbone only)")
    print(f"Config: {args.config}")
    print(f"Test data: {args.test_dir}")
    print(f"Evaluation mode: {args.evaluation_mode}")
    if args.evaluation_mode == "gt_points":
        print(f"Number of GT points: {args.num_gt_points}")
    print(f"Threshold: {args.threshold}")
    print("-" * 50)
    
    # Get project root directory (go up from new_src/evaluation/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Convert relative paths to absolute paths from project root
    if not os.path.isabs(args.backbone_ckpt):
        args.backbone_ckpt = os.path.join(project_root, args.backbone_ckpt)
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    if not os.path.isabs(args.test_dir):
        args.test_dir = os.path.join(project_root, args.test_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    if args.lora_checkpoint and not os.path.isabs(args.lora_checkpoint):
        args.lora_checkpoint = os.path.join(project_root, args.lora_checkpoint)
    
    # Check if files exist
    if not os.path.exists(args.backbone_ckpt):
        raise ValueError(f"Backbone checkpoint not found: {args.backbone_ckpt}")
    if args.lora_checkpoint and not os.path.exists(args.lora_checkpoint):
        raise ValueError(f"LoRA checkpoint not found: {args.lora_checkpoint}")
    if not os.path.exists(args.config):
        raise ValueError(f"Config file not found: {args.config}")
    
    # Load test data
    print("Loading test data...")
    test_img_dir = os.path.join(args.test_dir, "img")
    test_ann_dir = os.path.join(args.test_dir, "ann")
    
    if not os.path.exists(test_img_dir) or not os.path.exists(test_ann_dir):
        raise ValueError(f"Test data structure not found. Expected: {test_img_dir} and {test_ann_dir}")
    
    # Get all image files
    image_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} test images")
    
    test_data = []
    for img_file in image_files:
        img_path = os.path.join(test_img_dir, img_file)
        ann_file = img_file + '.json'
        ann_path = os.path.join(test_ann_dir, ann_file)
        
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
    
    # Load model with optional LoRA
    predictor = load_backbone_with_optional_lora(
        backbone_ckpt=args.backbone_ckpt,
        config_path=args.config,
        lora_ckpt=args.lora_checkpoint,
        device=device
    )
    
    # Extract model info for CSV
    ckpt_backbone = args.backbone_ckpt
    ckpt_lora = args.lora_checkpoint or 'None'
    
    # Determine model label based on backbone and LoRA
    backbone_name = os.path.basename(ckpt_backbone)
    if '.pt' in backbone_name and 'sam2_hiera' in backbone_name:
        # Meta base model
        if ckpt_lora != 'None':
            model_label = 'SAM2-Base+LoRA'
        else:
            model_label = 'SAM2-Base'
    else:
        # Fine-tuned model
        if ckpt_lora != 'None':
            model_label = 'SAM2-FT+LoRA'
        else:
            model_label = 'SAM2-FT'
    
    model_info = {
        'model_label': model_label,
        'backbone_ckpt': ckpt_backbone,
        'lora_ckpt': ckpt_lora,
        'rank': 8,  # Default values for LoRA, N/A for base
        'alpha': 16,
        'lr': 'unknown',
        'steps': 'unknown',
        'weight_decay': 'unknown'
    }
    
    # Extract info from LoRA checkpoint path (if available)
    if ckpt_lora != 'None':
        lora_name = os.path.basename(ckpt_lora)
        
        if 'r8' in lora_name or 'r16' in lora_name:
            if 'r8' in lora_name:
                model_info['rank'] = 8
            elif 'r16' in lora_name:
                model_info['rank'] = 16
        
        if 'lr' in lora_name:
            # Extract learning rate from filename
            import re
            lr_match = re.search(r'lr([0-9e\-\.]+)', lora_name)
            if lr_match:
                model_info['lr'] = lr_match.group(1)
    else:  # no LoRA
        model_info['rank'] = 'N/A'
        model_info['alpha'] = 'N/A'
        model_info['lr'] = 'N/A'
        model_info['steps'] = 'N/A'
        model_info['weight_decay'] = 'N/A'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate model
    print(f"\nEvaluating {model_info['model_label']}...")
    results, detailed_results = evaluate_lora_sam2(
        predictor=predictor,
        test_data=test_data,
        threshold=args.threshold,
        save_examples=args.save_examples,
        output_dir=args.output_dir if args.save_examples else None,
        evaluation_mode=args.evaluation_mode,
        num_gt_points=args.num_gt_points
    )
    
    if results is None:
        print("Evaluation failed!")
        return
    
    # Print results
    print(f"\nEvaluation Results for {model_info['model_label']}:")
    print("=" * 50)
    print(f"Model: {model_info['model_label']}")
    print(f"Backbone: {os.path.basename(model_info['backbone_ckpt'])}")
    if model_info['lora_ckpt'] != 'None':
        print(f"LoRA: {os.path.basename(model_info['lora_ckpt'])}")
    print(f"Evaluation mode: {args.evaluation_mode}")
    if args.evaluation_mode == "gt_points":
        print(f"Number of GT points: {args.num_gt_points}")
    print(f"Basic Metrics (threshold={args.threshold}):")
    print(f"  mIoU: {results['iou']:.4f}")
    print(f"  mDice: {results['dice']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    print(f"\nIoU at Different Thresholds:")
    if 'iou_50' in results:
        print(f"  IoU@50: {results['iou_50']:.4f}")
    if 'iou_75' in results:
        print(f"  IoU@75: {results['iou_75']:.4f}")
    if 'iou_90' in results:
        print(f"  IoU@90: {results['iou_90']:.4f}")
    if 'iou_95' in results:
        print(f"  IoU@95: {results['iou_95']:.4f}")
    
    print(f"\nExtra Metrics:")
    if 'auprc' in results:
        print(f"  AUPRC: {results['auprc']:.4f}")
    
    print(f"\nPerformance:")
    if 'avg_inference_time' in results:
        print(f"  Average inference time: {results['avg_inference_time']:.4f}s")
    if 'total_inference_time' in results:
        print(f"  Total inference time: {results['total_inference_time']:.2f}s")
    
    # Save results to CSV
    csv_path = save_results_csv(results, model_info, "new_src/evaluation/evaluation_results/sam2_lora", 
                               args.evaluation_mode, args.num_gt_points)
    
    # Save detailed results as JSON
    json_path = os.path.join(args.output_dir, f"detailed_results_lora_sam2.json")
    results_json = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_json[key] = value.tolist()
        else:
            results_json[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {csv_path}")
    print(f"Detailed results saved to: {json_path}")
    if args.save_examples:
        print(f"Prediction examples saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
