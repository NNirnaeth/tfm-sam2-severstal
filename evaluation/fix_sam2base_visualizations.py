#!/usr/bin/env python3
"""
Script corregido para generar visualizaciones correctas de SAM2 base models.
Arregla el problema de la función get_binary_gt_mask que no maneja correctamente
las coordenadas de origen de los bitmaps.
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

# Add paths
sys.path.append('../../libs/sam2base')
sys.path.append('../../new_src/utils')
from metrics import SegmentationMetrics

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Constants for hyperparameters
POINT_BATCH_SIZE = 4
BOX_BATCH_SIZE = 8
GRID_SPACING = 128
BOX_SCALES = [128, 256]
CONFIDENCE_THRESHOLD = 0.3
NMS_IOU_THRESHOLD = 0.65
TOP_K_FILTER = 200

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_binary_gt_mask_fixed(ann_path, target_shape):
    """
    CORREGIDA: Load and combine all defect objects into a single binary mask from PNG bitmap
    Usa correctamente las coordenadas de origen (origin) para posicionar los bitmaps
    """
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        if 'objects' not in data or not data['objects']:
            return np.zeros(target_shape, dtype=bool)
        
        # Get target dimensions
        H, W = target_shape
        
        # Create empty mask
        full_mask = np.zeros((H, W), dtype=bool)
        
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
            
            # Handle different PNG modes correctly
            if png_image.mode == 'RGBA':
                # Use alpha channel for transparency
                patch = np.array(png_image.split()[-1])  # Alpha channel
            elif png_image.mode == 'RGB':
                # Convert RGB to grayscale
                patch = np.array(png_image.convert('L'))
            else:
                # Already grayscale
                patch = np.array(png_image)
            
            # Convert to binary mask
            patch_binary = (patch > 0).astype(np.uint8)
            
            # Get origin coordinates from bitmap (origin = [x, y])
            if 'origin' in bmp:
                ox, oy = map(int, bmp['origin'])
            else:
                ox, oy = 0, 0
            
            ph, pw = patch_binary.shape
            
            # Calculate placement coordinates (clamped to image boundaries)
            x1 = max(0, ox)
            y1 = max(0, oy)
            x2 = min(W, ox + pw)
            y2 = min(H, oy + ph)
            
            # Place the bitmap at the correct position
            if x2 > x1 and y2 > y1:
                # Calculate the portion of the patch to use
                patch_h = y2 - y1
                patch_w = x2 - x1
                
                # Extract the relevant portion of the patch
                patch_portion = patch_binary[:patch_h, :patch_w]
                
                # Place in the full mask using OR operation
                full_mask[y1:y2, x1:x2] = np.logical_or(
                    full_mask[y1:y2, x1:x2],
                    patch_portion.astype(bool)
                )
        
        return full_mask
        
    except Exception as e:
        print(f"Error loading GT from {ann_path}: {e}")
        return np.zeros(target_shape, dtype=bool)

def generate_auto_prompts(image, grid_spacing=GRID_SPACING, box_scales=BOX_SCALES):
    """Generate automatic prompts from image without GT information"""
    height, width = image.shape[:2]
    
    # Strategy 1: Grid-based point sampling
    grid_points = []
    grid_labels = []
    
    for y in range(grid_spacing, height - grid_spacing, grid_spacing):
        for x in range(grid_spacing, width - grid_spacing, grid_spacing):
            grid_points.append([x, y])
            grid_labels.append(1)  # All points are positive prompts
    
    # Strategy 2: Multi-scale box sweeping
    auto_boxes = []
    for scale in box_scales:
        stride = scale
        for y in range(0, height - scale, stride):
            for x in range(0, width - scale, stride):
                if x + scale <= width and y + scale <= height:
                    auto_boxes.append([x, y, x + scale, y + scale])
    
    return {
        'points': np.array(grid_points) if grid_points else np.zeros((0, 2)),
        'point_labels': np.array(grid_labels) if grid_labels else np.zeros((0,)),
        'boxes': np.array(auto_boxes) if auto_boxes else np.zeros((0, 4))
    }

def predict_with_auto_prompts(predictor, image, auto_prompts, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Predict using automatic prompts with proper individual processing and NMS"""
    height, width = image.shape[:2]
    
    all_masks = []
    all_scores = []
    
    # Predict with grid points
    if len(auto_prompts['points']) > 0:
        try:
            points = auto_prompts['points']
            labels = auto_prompts['point_labels']
            
            # Process points in small batches
            for i in range(0, len(points), POINT_BATCH_SIZE):
                batch_points = points[i:i + POINT_BATCH_SIZE]
                batch_labels = labels[i:i + POINT_BATCH_SIZE]
                
                predictor.set_image(image)
                masks, scores, _ = predictor.predict(
                    point_coords=batch_points,
                    point_labels=batch_labels,
                    multimask_output=True
                )
                
                # Filter by confidence
                for mask, score in zip(masks, scores):
                    if score > confidence_threshold:
                        all_masks.append(mask)
                        all_scores.append(score)
                        
        except Exception as e:
            print(f"Error with point prompts: {e}")
    
    # Predict with boxes
    if len(auto_prompts['boxes']) > 0:
        try:
            boxes = auto_prompts['boxes']
            
            # Process boxes in small batches
            for i in range(0, len(boxes), BOX_BATCH_SIZE):
                batch_boxes = boxes[i:i + BOX_BATCH_SIZE]
                
                predictor.set_image(image)
                masks, scores, _ = predictor.predict(
                    box=batch_boxes[0],  # SAM2 expects single box
                    multimask_output=True
                )
                
                # Filter by confidence
                for mask, score in zip(masks, scores):
                    if score > confidence_threshold:
                        all_masks.append(mask)
                        all_scores.append(score)
                        
        except Exception as e:
            print(f"Error with box prompts: {e}")
    
    if not all_masks:
        return np.zeros((height, width), dtype=bool), 0
    
    # Apply NMS
    filtered_masks, filtered_scores = apply_nms(all_masks, all_scores)
    
    # Combine masks
    if filtered_masks:
        combined_mask = np.zeros((height, width), dtype=bool)
        for mask in filtered_masks:
            combined_mask = np.logical_or(combined_mask, mask)
        return combined_mask, len(filtered_masks)
    else:
        return np.zeros((height, width), dtype=bool), 0

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

def compute_mask_iou(mask1, mask2):
    """Compute IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-8)

def create_visualization_panel_fixed(image, gt_mask, pred_mask, img_name, save_dir):
    """Create CORRECTED TP/FP/FN visualization panel"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'CORRECTED Analysis: {img_name}', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground Truth - CORREGIDO: Usar overlay rojo en lugar de verde
    axes[0, 1].imshow(image)
    # Create red overlay for GT
    gt_overlay = np.zeros_like(image)
    gt_overlay[:, :, 0] = gt_mask * 255  # Red channel
    axes[0, 1].imshow(gt_overlay, alpha=0.6)
    axes[0, 1].set_title('Ground Truth (Red Overlay)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Prediction - CORREGIDO: Usar overlay azul
    axes[1, 0].imshow(image)
    # Create blue overlay for prediction
    pred_overlay = np.zeros_like(image)
    pred_overlay[:, :, 2] = pred_mask * 255  # Blue channel
    axes[1, 0].imshow(pred_overlay, alpha=0.6)
    axes[1, 0].set_title('Prediction (Blue Overlay)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # TP/FP/FN Analysis
    tp = np.logical_and(pred_mask, gt_mask)
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
    
    analysis_img = np.zeros_like(image)
    analysis_img[tp] = [0, 255, 0]    # Green for TP
    analysis_img[fp] = [255, 0, 0]    # Red for FP
    analysis_img[fn] = [255, 255, 0]  # Yellow for FN
    
    axes[1, 1].imshow(analysis_img)
    axes[1, 1].set_title('TP/FP/FN Analysis\nGreen=TP, Red=FP, Yellow=FN', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add legend
    legend_elements = [
        patches.Patch(color='green', label='True Positive (TP)'),
        patches.Patch(color='red', label='False Positive (FP)'),
        patches.Patch(color='yellow', label='False Negative (FN)')
    ]
    axes[1, 1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Save visualization
    viz_path = os.path.join(save_dir, f"{img_name}_FIXED_analysis.png")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

def evaluate_sam2_base_model_fixed(config_file, model_name, results_dir, project_root, num_viz_samples=10):
    """Evaluate a base SAM2 model with CORRECTED visualization"""
    
    print(f"\nEvaluating {model_name} (Base Model - CORRECTED)...")
    
    # Create model-specific results directory
    model_results_dir = os.path.join(results_dir, f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_FIXED")
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Create visualizations directory
    viz_dir = os.path.join(model_results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load model - NO checkpoint, using base model as-is
    try:
        print("Loading base SAM2 model...")
        # Change to sam2base directory for proper config resolution
        original_dir = os.getcwd()
        sam2base_dir = os.path.join(project_root, 'libs', 'sam2base')
        os.chdir(sam2base_dir)
        
        # Use relative path from sam2base directory
        config_rel_path = os.path.relpath(config_file, sam2base_dir)
        sam2_model = build_sam2(config_rel_path, ckpt_path=None, device="cpu")
        sam2_model.to("cuda")
        sam2_model.eval()
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        # Change back to original directory
        os.chdir(original_dir)
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model {config_file}: {e}")
        # Change back to original directory in case of error
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return None
    
    # Test split paths
    test_img_dir = "../../datasets/Data/splits/test_split/img"
    test_ann_dir = "../../datasets/Data/splits/test_split/ann"
    
    # Get test images
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    selected_images = random.sample(test_images, min(num_viz_samples, len(test_images)))
    
    print(f"Selected {len(selected_images)} images for visualization")
    
    # Metrics
    all_ious = []
    all_dices = []
    
    # Process each image
    for i, img_file in enumerate(selected_images):
        print(f"Processing {i+1}/{len(selected_images)}: {img_file}")
        
        # Load image
        img_path = os.path.join(test_img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load image: {img_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Load ground truth with CORRECTED function
        ann_path = os.path.join(test_ann_dir, img_file.replace('.jpg', '.json'))
        gt_mask = get_binary_gt_mask_fixed(ann_path, (height, width))
        
        if gt_mask is None:
            print(f"Could not load GT for: {img_file}")
            continue
        
        # Generate automatic prompts
        auto_prompts = generate_auto_prompts(image)
        
        # Predict with SAM2
        pred_mask, num_masks = predict_with_auto_prompts(sam2_predictor, image, auto_prompts)
        
        # Calculate metrics
        if gt_mask.sum() > 0:  # Only if there are defects in GT
            iou = compute_mask_iou(pred_mask, gt_mask)
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
            
            all_ious.append(iou)
            all_dices.append(dice)
            
            print(f"  IoU: {iou:.4f}, Dice: {dice:.4f}, Masks: {num_masks}")
        
        # Create CORRECTED visualization
        base_name = os.path.splitext(img_file)[0]
        create_visualization_panel_fixed(image, gt_mask, pred_mask, base_name, viz_dir)
    
    # Calculate overall metrics
    if all_ious:
        mean_iou = np.mean(all_ious)
        mean_dice = np.mean(all_dices)
        print(f"\nOverall Metrics:")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Processed {len(all_ious)} images with defects")
    
    print(f"\nCORRECTED visualizations saved to: {viz_dir}")
    return model_results_dir

def main():
    parser = argparse.ArgumentParser(description='Fix SAM2 base model visualizations')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to visualize')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/ptp/sam2/new_src/evaluation/evaluation_results/sam2_base',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    print("SAM2 Base Models - CORRECTED Visualization")
    print("=" * 60)
    print(f"Visualizing {args.num_images} images")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Model configurations - use absolute paths
    models = [
        {
            'name': 'SAM2 Large Base',
            'config': os.path.join(project_root, 'libs', 'sam2base', 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
        },
        {
            'name': 'SAM2 Small Base', 
            'config': os.path.join(project_root, 'libs', 'sam2base', 'sam2', 'configs', 'sam2', 'sam2_hiera_s.yaml')
        }
    ]
    
    # Evaluate each model
    for model in models:
        if os.path.exists(model['config']):
            print(f"\n{'='*60}")
            print(f"Evaluating {model['name']}")
            print(f"{'='*60}")
            
            result_dir = evaluate_sam2_base_model_fixed(
                model['config'], 
                model['name'], 
                args.output_dir, 
                project_root,
                args.num_images
            )
            
            if result_dir:
                print(f"Results saved to: {result_dir}")
        else:
            print(f"Config file not found: {model['config']}")
    
    print(f"\n{'='*60}")
    print("CORRECTED visualization completed!")
    print("Key fixes applied:")
    print("1. Fixed get_binary_gt_mask to properly handle bitmap origin coordinates")
    print("2. Changed GT visualization from green overlay to red overlay")
    print("3. Changed prediction visualization from blue cmap to blue overlay")
    print("4. Added proper error handling for bitmap positioning")

if __name__ == "__main__":
    main()
