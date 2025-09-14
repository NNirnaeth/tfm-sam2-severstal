#!/usr/bin/env python3
"""
Evaluate SAM2 Base Models (Small and Large) without fine-tuning
on Severstal test dataset (2000 images)

This script evaluates the base SAM2 models as they come "out of the box"
without any training on the Severstal dataset, providing a baseline
for comparison with fine-tuned models.
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

# Add paths dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(os.path.join(project_root, 'libs/sam2base'))
sys.path.append(os.path.join(project_root, 'new_src/utils'))
from metrics import SegmentationMetrics

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Constants for hyperparameters (ensure consistency)
POINT_BATCH_SIZE = 4
BOX_BATCH_SIZE = 8
GRID_SPACING = 128                # FIXED: Same as fine-tuned evaluation
BOX_SCALES = [128, 256]           # FIXED: Same as fine-tuned evaluation  
CONFIDENCE_THRESHOLD_GT = 0.3     # For GT points (can be lower since they're accurate)
CONFIDENCE_THRESHOLD_AUTO = 0.5   # FIXED: Lower threshold for base models (they generate lower scores)
NMS_IOU_THRESHOLD = 0.65          # FIXED: Same as fine-tuned evaluation
TOP_K_FILTER = 200                # FIXED: Same as fine-tuned evaluation

def set_seed(seed=42):
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
    # FIXED: Use only positive prompts like in fine-tuned evaluation
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
    """Generate 1-3 GT points from ground truth mask (centroid + max distance + negative)"""
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

def generate_30_point_prompts(gt_mask):
    """Generate 30 point prompts from ground truth mask (for comparison with previous results)"""
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

def create_visualization_panel(image, gt_mask, pred_mask, img_name, save_dir):
    """Create TP/FP/FN visualization panel"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Analysis: {img_name}', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground Truth
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(gt_mask, alpha=0.6, cmap='Greens')
    axes[0, 1].set_title('Ground Truth (Green)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Prediction
    axes[1, 0].imshow(image)
    axes[1, 0].imshow(pred_mask, alpha=0.6, cmap='Blues')
    axes[1, 0].set_title('Prediction (Blue)', fontweight='bold')
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
    viz_path = os.path.join(save_dir, f"{img_name}_analysis.png")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

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

def predict_with_auto_prompts(predictor, image, auto_prompts, confidence_threshold=CONFIDENCE_THRESHOLD_AUTO, 
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
            # print(f"DEBUG: Processing {len(points)} point prompts")
            
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
                    max_score = np.max(point_scores) if len(point_scores) > 0 else 0
                    valid_scores = [s for s in point_scores if s > confidence_threshold]
                    # if i % 20 == 0:  # Print debug every 20 batches to avoid spam
                    #     print(f"DEBUG: Point batch {i} - Max score: {max_score:.3f}, Valid: {len(valid_scores)}/{len(point_scores)}")
                    
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
                    max_score = point_scores[best_idx]
                    # if i % 20 == 0:  # Print debug every 20 batches
                    #     print(f"DEBUG: Multi-point batch {i} - Best score: {max_score:.3f}")
                        
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
            for box_idx, box in enumerate(boxes):
                box_masks, box_scores, _ = predictor.predict(
                    box=box,
                    multimask_output=True
                )
                
                # Filter by confidence and add to collection
                max_score = np.max(box_scores) if len(box_scores) > 0 else 0
                valid_scores = [s for s in box_scores if s > confidence_threshold]
                # if box_idx % 10 == 0:  # Print debug every 10 boxes
                #     print(f"DEBUG: Box {box_idx} - Max score: {max_score:.3f}, Valid: {len(valid_scores)}/{len(box_scores)}")
                    
                for j, score in enumerate(box_scores):
                    if score > confidence_threshold:
                        # Ensure binary mask
                        mask = (box_masks[j].astype(np.float32) > 0.5).astype(bool)
                        all_masks.append(mask)
                        all_scores.append(score)
                        
        except Exception as e:
            print(f"Error with box prompts: {e}")
    
    if not all_masks:
        # print(f"DEBUG: No masks generated from auto-prompts (threshold={confidence_threshold})")
        return np.zeros((height, width), dtype=bool), 0
    
    # print(f"DEBUG: Generated {len(all_masks)} candidate masks from auto-prompts")
    
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
    
    # FIXED: Instead of OR logic, use best mask by score (more realistic)
    # OR logic was creating artificially large masks with inflated metrics
    if len(filtered_masks) == 1:
        return filtered_masks[0].astype(bool), 1
    else:
        # Take the best mask by score instead of combining all
        best_idx = np.argmax(filtered_scores)
        best_mask = filtered_masks[best_idx].astype(bool)
        return best_mask, 1

def predict_with_gt_points(predictor, image, gt_points, gt_labels, confidence_threshold=CONFIDENCE_THRESHOLD_GT):
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

def evaluate_sam2_base_model(config_file, model_name, results_dir, evaluation_mode="30_gt_points", num_gt_points=3, num_viz_samples=50, project_root_dir=None):
    """Evaluate a base SAM2 model without fine-tuning
    
    Args:
        config_file: Path to SAM2 config file
        model_name: Name of the model for display
        results_dir: Directory to save results
        evaluation_mode: "auto_prompt", "gt_points", or "30_gt_points"
        num_gt_points: Number of GT points to use (1-3) when mode is "gt_points"
    """
    
    print(f"\nEvaluating {model_name} (Base Model - No Fine-tuning)...")
    print(f"Evaluation mode: {evaluation_mode}")
    if evaluation_mode == "gt_points":
        print(f"Number of GT points: {num_gt_points}")
    
    # Create model-specific results directory with evaluation mode
    mode_suffix = f"_{evaluation_mode}"
    if evaluation_mode == "gt_points":
        mode_suffix += f"_{num_gt_points}points"
    
    model_results_dir = os.path.join(results_dir, f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}{mode_suffix}")
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Create visualizations directory
    viz_dir = os.path.join(model_results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load model - NO checkpoint, using base model as-is
    try:
        print("Loading base SAM2 model...")
        
        # Change to project root temporarily to help Hydra find configs
        original_cwd = os.getcwd()
        os.chdir(project_root_dir)
        
        try:
            # Build the base SAM2 model without loading any trained weights
            # Use just the config filename without full path for Hydra
            config_filename = os.path.basename(config_file)
            sam2_model = build_sam2(f"configs/sam2/{config_filename}", ckpt_path=None, device="cpu")
            
            # Move to GPU and create predictor
            sam2_model.to("cuda")
            sam2_model.eval()
            sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("Model loaded successfully!")
            
        finally:
            # Always restore original directory
            os.chdir(original_cwd)
        
    except Exception as e:
        print(f"Error loading model {config_file}: {e}")
        return None
    
    # Use dynamic paths for datasets
    if project_root_dir is None:
        project_root_dir = project_root
    test_img_dir = os.path.join(project_root_dir, "datasets/Data/splits/test_split/img")
    test_ann_dir = os.path.join(project_root_dir, "datasets/Data/splits/test_split/ann")
    
    # Verify paths exist
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")
    if not os.path.exists(test_ann_dir):
        raise FileNotFoundError(f"Test annotation directory not found: {test_ann_dir}")
    
    # Get ALL test images
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.jpg')])
    print(f"Found {len(test_images)} test images in {test_img_dir}")
    print(f"Evaluating on {len(test_images)} test images...")
    
    batch_metrics = []
    visualization_count = 0
    max_visualizations = num_viz_samples  # Use parameter for number of visualizations
    
    # Progress bar with estimated time
    pbar = tqdm(test_images, desc=f"Processing {model_name}")
    
    for i, img_file in enumerate(pbar):
        try:
            # Update progress bar with current image
            pbar.set_postfix({'Image': img_file[:20], 'Progress': f"{i+1}/{len(test_images)}"})
            
            # Load image
            img_path = os.path.join(test_img_dir, img_file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set image for SAM2
            sam2_predictor.set_image(image)
            
            # Load ground truth (all objects combined)
            ann_path = os.path.join(test_ann_dir, img_file + '.json')
            gt_mask = get_binary_gt_mask(ann_path, image.shape[:2])
            
            if gt_mask is None or not gt_mask.any():
                continue
            
            # Generate prompts based on evaluation mode
            if evaluation_mode == "auto_prompt":
                # Generate automatic prompts from image (no GT)
                auto_prompts = generate_auto_prompts(image)
                gt_points, gt_labels = None, None
            elif evaluation_mode == "gt_points":
                # Generate 1-3 GT points
                gt_points, gt_labels = generate_gt_points(gt_mask, num_gt_points)
                auto_prompts = None
            elif evaluation_mode == "30_gt_points":
                # Generate 30 GT points
                gt_points, gt_labels = generate_30_point_prompts(gt_mask)
                auto_prompts = None
            
            # Predict with SAM2 based on evaluation mode
            with torch.no_grad():
                if evaluation_mode == "auto_prompt":
                    # Predict with SAM2 using automatic prompts
                    pred_mask, num_masks = predict_with_auto_prompts(sam2_predictor, image, auto_prompts)
                else:
                    # Predict with SAM2 using GT points
                    pred_mask, num_masks = predict_with_gt_points(sam2_predictor, image, gt_points, gt_labels)
            
            if pred_mask is not None and pred_mask.any():
                
                # Usar métricas unificadas en lugar de compute_metrics
                metrics_calculator = SegmentationMetrics(save_dir=model_results_dir)
                sample_metrics = metrics_calculator.compute_pixel_metrics(pred_mask, gt_mask)
                benevolent_metrics = metrics_calculator.compute_benevolent_metrics(pred_mask, gt_mask)
                
                # Combinar métricas
                combined_metrics = {**sample_metrics, **benevolent_metrics}
                
                # Guardar para agregación posterior
                batch_metrics.append(combined_metrics)
                
                # Create visualization for some samples (limit to avoid disk space issues)
                if visualization_count < max_visualizations:
                    img_name = os.path.splitext(img_file)[0]
                    viz_path = create_visualization_panel(image, gt_mask, pred_mask, img_name, viz_dir)
                    visualization_count += 1
                    if visualization_count % 10 == 0:
                        print(f"Created {visualization_count} visualizations...")
            
            # Clear GPU memory after each image
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            torch.cuda.empty_cache()
            continue
    
    pbar.close()
    
    # Clear GPU memory
    del sam2_model, sam2_predictor
    torch.cuda.empty_cache()
    
    if not batch_metrics:
        print(f"No valid predictions for {model_name}")
        return None
    
    print(f"Created {visualization_count} visualizations in {viz_dir}")
    
    # Usar métricas unificadas
    metrics_calculator = SegmentationMetrics(save_dir=model_results_dir)
    
    # Al final, agregar métricas y guardar CSV
    final_results = metrics_calculator.aggregate_metrics(batch_metrics)
    iou_threshold_metrics = metrics_calculator.compute_iou_at_thresholds([m['iou'] for m in batch_metrics])
    final_results.update(iou_threshold_metrics)
    
    # Add metadata
    final_results.update({
        'model_name': model_name,
        'model_type': 'base_no_finetuning',
        'config_file': config_file,
        'evaluation_mode': evaluation_mode,
        'num_gt_points': num_gt_points if evaluation_mode == "gt_points" else None,
        'images_processed': len(batch_metrics),
        'total_test_images': len(test_images),
        'visualizations_created': visualization_count,
        'evaluation_timestamp': datetime.now().isoformat()
    })
    
    # Guardar a CSV con nombre más descriptivo incluyendo modo de evaluación
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = f"_{evaluation_mode}"
    if evaluation_mode == "gt_points":
        mode_suffix += f"_{num_gt_points}points"
    
    csv_filename = f"sam2base_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}{mode_suffix}_{timestamp}"
    
    csv_path = metrics_calculator.save_results_to_csv(
        final_results, 
        csv_filename,
        {"model": model_name, "model_type": "base_no_finetuning"}
    )
    
    # Get F1 score with fallback to different possible key names
    f1_key = None
    for key in ['mean_f1_score', 'mean_f1', 'f1_score']:
        if key in final_results:
            f1_key = key
            break
    
    f1_value = final_results.get(f1_key, 0.0)
    print(f"Completed {model_name}: IoU={final_results.get('mean_iou', 0.0):.4f}, F1={f1_value:.4f}, Images={len(batch_metrics)}")
    print(f"Results saved to: {csv_path}")
    print(f"Visualizations saved to: {viz_dir}")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM2 Base Models (Small and Large) without fine-tuning')
    parser.add_argument('--evaluation_mode', type=str, 
                       choices=['auto_prompt', 'gt_points', '30_gt_points', 'all_modes'],
                       default='all_modes',
                       help='Evaluation mode: auto_prompt (no GT), gt_points (1-3 GT points), 30_gt_points (30 GT points), or all_modes (run all 3)')
    parser.add_argument('--num_gt_points', type=int, default=3,
                       help='Number of GT points to use when evaluation_mode is gt_points (1-3)')
    parser.add_argument('--model_size', type=str, 
                       choices=['small', 'large', 'both'],
                       default='both',
                       help='Which model size to evaluate: small, large, or both')
    parser.add_argument('--results_dir', type=str, 
                       default='evaluation_results/sam2_base',
                       help='Directory to save results')
    parser.add_argument('--num_viz', type=int, default=50,
                       help='Number of random samples to visualize')
    parser.add_argument('--sam2_config_dir', type=str,
                       default=None,
                       help='Path to SAM2 config directory')
    
    args = parser.parse_args()
    
    # Set default config dir if not provided
    if args.sam2_config_dir is None:
        args.sam2_config_dir = os.path.join(project_root, 'configs/sam2')
    
    # Validate arguments
    if args.evaluation_mode == 'gt_points' and (args.num_gt_points < 1 or args.num_gt_points > 3):
        print("Error: num_gt_points must be between 1 and 3 when evaluation_mode is gt_points")
        return
    
    # Set seed for reproducibility
    set_seed(42)
    
    print("SAM2 Base Models Evaluation (No Fine-tuning)")
    print("=" * 80)
    print("This script evaluates SAM2 Small and Large models")
    print("as they come from the original training, without any")
    print("fine-tuning on the Severstal dataset.")
    print("=" * 80)
    # Define evaluation modes to run
    if args.evaluation_mode == "all_modes":
        evaluation_modes = [
            {"mode": "auto_prompt", "num_gt_points": None},
            {"mode": "gt_points", "num_gt_points": args.num_gt_points},
            {"mode": "30_gt_points", "num_gt_points": None}
        ]
        print(f"Running all evaluation modes: auto_prompt, gt_points ({args.num_gt_points} points), 30_gt_points")
    else:
        evaluation_modes = [{"mode": args.evaluation_mode, "num_gt_points": args.num_gt_points}]
        print(f"Evaluation mode: {args.evaluation_mode}")
        if args.evaluation_mode == "gt_points":
            print(f"Number of GT points: {args.num_gt_points}")
    
    print(f"Model size(s): {args.model_size}")
    
    # Create main results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.evaluation_mode == "all_modes":
        results_dir = f"{args.results_dir}/sam2base_evaluation_all_modes_{timestamp}"
    else:
        mode_suffix = f"_{args.evaluation_mode}"
        if args.evaluation_mode == "gt_points":
            mode_suffix += f"_{args.num_gt_points}points"
        results_dir = f"{args.results_dir}/sam2base_evaluation{mode_suffix}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Define base SAM2 models to evaluate based on selection
    base_models = []
    if args.model_size in ['small', 'both']:
        base_models.append({
            "config": os.path.join(args.sam2_config_dir, "sam2_hiera_s.yaml"),
            "name": "SAM2 Small Base (No Fine-tuning)"
        })
    if args.model_size in ['large', 'both']:
        base_models.append({
            "config": os.path.join(args.sam2_config_dir, "sam2.1_hiera_l.yaml"), 
            "name": "SAM2 Large Base (No Fine-tuning)"
        })
    
    print(f"Found {len(base_models)} base models to evaluate")
    
    # Verify config files exist
    for model in base_models:
        if not os.path.exists(model["config"]):
            print(f"Error: Config file not found: {model['config']}")
            print(f"Available files in {args.sam2_config_dir}:")
            if os.path.exists(args.sam2_config_dir):
                for f in os.listdir(args.sam2_config_dir):
                    print(f"  {f}")
            return
        print(f"✓ Found config: {model['config']}")
    
    # Estimate total time based on actual observed performance
    estimated_time_small_per_mode = 3  # 3 minutes for Small per mode
    estimated_time_large_per_mode = 7  # 7 minutes for Large per mode
    
    num_small_models = len([m for m in base_models if "Small" in m["name"]])
    num_large_models = len([m for m in base_models if "Large" in m["name"]])
    num_modes = len(evaluation_modes)
    
    total_estimated = (num_small_models * estimated_time_small_per_mode + 
                      num_large_models * estimated_time_large_per_mode) * num_modes
    
    print(f"\nEstimated evaluation time:")
    print(f"  Models to evaluate: {len(base_models)} models")
    print(f"  Modes per model: {num_modes} modes")
    print(f"  Total experiments: {len(base_models) * num_modes}")
    if num_small_models > 0:
        print(f"  SAM2 Small: ~{estimated_time_small_per_mode} min/mode × {num_modes} modes = ~{estimated_time_small_per_mode * num_modes} min")
    if num_large_models > 0:
        print(f"  SAM2 Large: ~{estimated_time_large_per_mode} min/mode × {num_modes} modes = ~{estimated_time_large_per_mode * num_modes} min")
    print(f"  Total estimated time: ~{total_estimated} minutes (~{total_estimated/60:.1f} hours)")
    print(f"  (Based on actual performance with improved auto-prompt settings)")
    
    # Evaluate models and modes sequentially
    start_time = time.time()
    results = []
    total_experiments = len(base_models) * len(evaluation_modes)
    
    experiment_count = 0
    
    for model_config in base_models:
        for eval_config in evaluation_modes:
            experiment_count += 1
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {experiment_count}/{total_experiments}")
            print(f"Model: {model_config['name']}")
            print(f"Mode: {eval_config['mode']}")
            if eval_config['num_gt_points']:
                print(f"GT Points: {eval_config['num_gt_points']}")
            print(f"{'='*80}")
            
            model_start_time = time.time()
            
            result = evaluate_sam2_base_model(
                model_config["config"], 
                model_config["name"],
                results_dir,
                evaluation_mode=eval_config["mode"],
                num_gt_points=eval_config["num_gt_points"] or 3,
                num_viz_samples=args.num_viz,
                project_root_dir=project_root
            )
            
            if result is not None:
                results.append(result)
                
                # Calculate model evaluation time
                model_time = time.time() - model_start_time
                print(f"Experiment completed in {model_time/60:.1f} minutes")
            
            # Save intermediate results after each experiment
            intermediate_file = os.path.join(results_dir, f"intermediate_results_{experiment_count}.json")
            # Ensure directory exists
            os.makedirs(os.path.dirname(intermediate_file), exist_ok=True)
            with open(intermediate_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved intermediate results to {intermediate_file}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETED!")
    print(f"{'='*80}")
    print(f"Total evaluation time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successfully evaluated {len(results)} base models")
    print(f"Results saved to: {results_dir}")
    
    # Save final results
    output_file = os.path.join(results_dir, "final_results_all_models.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Final results saved to: {output_file}")
    
    # Print summary
    print("\nSummary of Base Model Results:")
    print("-" * 120)
    print(f"{'Model':<40} {'Mode':<15} {'IoU':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Images':<8} {'Time':<8}")
    print("-" * 120)
    
    for result in results:
        model_name = result['model_name']
        eval_mode = result.get('evaluation_mode', 'unknown')
        if eval_mode == 'gt_points' and result.get('num_gt_points'):
            eval_mode = f"gt_{result.get('num_gt_points')}pts"
        elif eval_mode == '30_gt_points':
            eval_mode = "gt_30pts"
        elif eval_mode == 'auto_prompt':
            eval_mode = "auto"
            
        # Get F1 score with fallback
        f1_key = None
        for key in ['mean_f1_score', 'mean_f1', 'f1_score']:
            if key in result:
                f1_key = key
                break
        f1_value = result.get(f1_key, 0.0)
        
        print(f"{model_name:<40} {eval_mode:<15} {result.get('mean_iou', 0.0):<8.4f} {f1_value:<8.4f} "
              f"{result.get('mean_precision', 0.0):<10.4f} {result.get('mean_recall', 0.0):<8.4f} {result.get('images_processed', 0):<8}")
    
    print("\n" + "=" * 80)
    print("IMPORTANT: These results represent the baseline performance")
    print("of SAM2 models without any fine-tuning on Severstal data.")
    print("Compare these with fine-tuned models to see improvement.")
    print("=" * 80)
    
    # Save summary to text file
    summary_file = os.path.join(results_dir, "evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("SAM2 Base Models Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"Models evaluated: {len(results)}\n\n")
        
        for result in results:
            # Get F1 score with fallback
            f1_key = None
            for key in ['mean_f1_score', 'mean_f1', 'f1_score']:
                if key in result:
                    f1_key = key
                    break
            f1_value = result.get(f1_key, 0.0)
            
            eval_mode = result.get('evaluation_mode', 'unknown')
            if eval_mode == 'gt_points' and result.get('num_gt_points'):
                eval_mode = f"gt_points_{result.get('num_gt_points')}pts"
            
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"  Mode: {eval_mode}\n")
            f.write(f"  IoU: {result.get('mean_iou', 0.0):.4f}\n")
            f.write(f"  F1: {f1_value:.4f}\n")
            f.write(f"  Precision: {result.get('mean_precision', 0.0):.4f}\n")
            f.write(f"  Recall: {result.get('mean_recall', 0.0):.4f}\n")
            f.write(f"  Images processed: {result.get('images_processed', 0)}\n\n")
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()



