#!/usr/bin/env python3
"""
Evaluate SAM2 Large model trained on full Severstal dataset
using the best checkpoint from training.

This script evaluates the fine-tuned SAM2 Large model on the complete test split
with three evaluation modes:
1. Auto-prompt: SAM2-AutoPrompt (no GT, no external detector) - for reference/experiment
2. GT 30 points: SAM2 with 30 GT points (for comparison with previous results)
3. GT 3 points: SAM2 with 1-3 GT points (centroid + max distance + negative point)

The script provides comprehensive metrics, visualizations, and timing analysis.
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

def get_binary_gt_mask(ann_path, target_shape):
    """
    Crea una máscara binaria del tamaño de la imagen colocando cada bitmap
    (decodificado del PNG) en su posición correcta usando 'origin' = [x, y].
    """
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)

        H, W = target_shape
        full = np.zeros((H, W), dtype=np.uint8)

        for obj in data.get('objects', []):
            bmp = obj.get('bitmap', {})
            if 'data' not in bmp or 'origin' not in bmp:
                continue

            # Decodifica PNG comprimido
            raw = zlib.decompress(base64.b64decode(bmp['data']))
            patch = np.array(Image.open(io.BytesIO(raw)))

            # Usa canal alfa si existe; si no, primer canal
            if patch.ndim == 3 and patch.shape[2] == 4:
                patch = patch[:, :, 3]
            elif patch.ndim == 3:
                patch = patch[:, :, 0]

            patch = (patch > 0).astype(np.uint8)

            ox, oy = map(int, bmp['origin'])  # origin = [x, y]
            ph, pw = patch.shape

            x1 = max(0, ox); y1 = max(0, oy)
            x2 = min(W, ox + pw); y2 = min(H, oy + ph)
            if x2 > x1 and y2 > y1:
                full[y1:y2, x1:x2] = np.maximum(
                    full[y1:y2, x1:x2], patch[:(y2 - y1), :(x2 - x1)]
                )

        return full.astype(bool)

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

def create_visualization_panel(image, gt_mask, pred_mask, img_name, save_dir):
    """Create comprehensive visualization panel with predictions vs GT"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Evaluation: {img_name}', fontsize=16, fontweight='bold')
    
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

def evaluate_sam2_large_full_dataset(checkpoint_path, results_dir, num_viz_samples=10, evaluation_mode="auto_prompt", num_gt_points=3):
    """Evaluate SAM2 Large model trained on full dataset
    
    Args:
        checkpoint_path: Path to SAM2 checkpoint
        results_dir: Directory to save results
        num_viz_samples: Number of samples to visualize
        evaluation_mode: "auto_prompt", "gt_points", or "30_gt_points"
        num_gt_points: Number of GT points to use (1-3) when mode is "gt_points"
    """
    
    print(f"\nEvaluating SAM2 Large (Fine-tuned on Full Dataset)...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluation mode: {evaluation_mode}")
    if evaluation_mode == "gt_points":
        print(f"Number of GT points: {num_gt_points}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create visualizations directory
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load model with checkpoint
    try:
        print("Loading fine-tuned SAM2 Large model...")
        # Change to sam2base directory for proper config resolution
        original_dir = os.getcwd()
        sam2base_dir = "/home/ptp/sam2/libs/sam2base"
        os.chdir(sam2base_dir)
        
        # Use relative path from sam2base directory - Large model config
        config_file = "configs/sam2/sam2_hiera_l.yaml"
        sam2_model = build_sam2(config_file, ckpt_path=checkpoint_path, device="cpu")
        
        # Move to GPU and create predictor
        sam2_model.to("cuda")
        sam2_model.eval()
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("Model loaded successfully!")
        
        # Restore original directory
        os.chdir(original_dir)
        
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        # Restore original directory in case of error
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return None
    
    # Test split paths - use absolute paths
    test_img_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/img"
    test_ann_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/ann"
    
    # Verify paths exist
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")
    if not os.path.exists(test_ann_dir):
        raise FileNotFoundError(f"Test annotation directory not found: {test_ann_dir}")
    
    # Get test image files
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(test_images)} test images")
    
    # Initialize metrics
    metrics_calculator = SegmentationMetrics(save_dir=results_dir)
    
    # Storage for evaluation
    all_predictions = []
    all_ground_truths = []
    all_ious = []
    inference_times = []
    
    # Random sample for visualization
    viz_indices = random.sample(range(len(test_images)), min(num_viz_samples, len(test_images)))
    
    print(f"\nStarting evaluation...")
    print(f"Will create visualizations for {len(viz_indices)} random samples")
    
    # Set evaluation mode description
    if evaluation_mode == "auto_prompt":
        print(f"Evaluation mode: SAM2-AutoPrompt (no GT, no external detector) - for reference/experiment")
    elif evaluation_mode == "gt_points":
        print(f"Evaluation mode: SAM2 with {num_gt_points} GT points (centroid + max distance + negative)")
    elif evaluation_mode == "30_gt_points":
        print(f"Evaluation mode: SAM2 with 30 GT points (for comparison with previous results)")
    
    # Evaluation loop
    for idx, img_file in enumerate(tqdm(test_images, desc="Evaluating")):
        img_path = os.path.join(test_img_dir, img_file)
        ann_path = os.path.join(test_ann_dir, img_file + '.json')
        
        # Skip if no annotation
        if not os.path.exists(ann_path):
            continue
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Set image for SAM2 predictor
        sam2_predictor.set_image(image)
        
        # Load ground truth
        gt_mask = get_binary_gt_mask(ann_path, (height, width))
        if gt_mask is None:
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
            gt_points, gt_labels = generate_30_gt_points(gt_mask)
            auto_prompts = None
        
        # Inference with timing
        start_time = time.time()
        
        try:
            if evaluation_mode == "auto_prompt":
                # Predict with SAM2 using automatic prompts
                pred_mask, num_masks = predict_with_auto_prompts(sam2_predictor, image, auto_prompts)
            else:
                # Predict with SAM2 using GT points
                pred_mask, num_masks = predict_with_gt_points(sam2_predictor, image, gt_points, gt_labels)
            
        except Exception as e:
            print(f"Error during inference for {img_file}: {e}")
            pred_mask = np.zeros_like(gt_mask)
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Store for metrics
        all_predictions.append(pred_mask)
        all_ground_truths.append(gt_mask)
        
        # Compute IoU for this sample
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / (union + 1e-8)
        all_ious.append(iou)
        
        # Create visualization for random samples
        if idx in viz_indices:
            img_name = os.path.splitext(img_file)[0]
            viz_path = create_visualization_panel(image, gt_mask, pred_mask, img_name, viz_dir)
            print(f"Visualization saved: {viz_path}")
    
    # Compute comprehensive metrics
    print(f"\nComputing metrics for {len(all_predictions)} samples...")
    
    # Evaluate batch
    batch_results = metrics_calculator.evaluate_batch(all_predictions, all_ground_truths)
    
    # Add inference timing metrics
    batch_results['mean_inference_time'] = float(np.mean(inference_times))
    batch_results['std_inference_time'] = float(np.std(inference_times))
    batch_results['total_inference_time'] = float(np.sum(inference_times))
    
    # Add IoU threshold metrics
    iou_threshold_metrics = metrics_calculator.compute_iou_at_thresholds(all_ious)
    batch_results.update(iou_threshold_metrics)
    
    # Prepare model info for CSV based on evaluation mode
    if evaluation_mode == "auto_prompt":
        prompt_type = "sam2_autoprompt"
        prompt_strategy = "sam2_autoprompt_no_gt_no_detector"
    elif evaluation_mode == "gt_points":
        prompt_type = f"gt_points_{num_gt_points}"
        prompt_strategy = f"gt_points_centroid_maxdist_negative_{num_gt_points}"
    elif evaluation_mode == "30_gt_points":
        prompt_type = "gt_points_30"
        prompt_strategy = "gt_points_30_random_sampling"
    
    model_info = {
        'model': 'sam2_large',
        'variant': 'fine_tuned_full_dataset',
        'prompt_type': prompt_type,
        'checkpoint': os.path.basename(checkpoint_path),
        'dataset': 'severstal_full',
        'split': 'test',
        'evaluation_mode': evaluation_mode
    }
    
    # Additional experiment info
    additional_info = {
        'num_samples': len(all_predictions),
        'image_size': f"{height}x{width}",
        'prompt_strategy': prompt_strategy,
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_gt_points': num_gt_points if evaluation_mode == "gt_points" else None
    }
    
    # Save results to CSV
    experiment_name = f"eval_sam2_large_full_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}"
    csv_path = metrics_calculator.save_results_to_csv(
        batch_results, experiment_name, model_info, additional_info
    )
    
    # Print comprehensive summary
    metrics_calculator.print_metrics_summary(batch_results, "SAM2 Large (Fine-tuned)")
    
    # Print timing information
    print(f"\n{'='*60}")
    print("TIMING INFORMATION")
    print(f"{'='*60}")
    print(f"Mean inference time: {batch_results['mean_inference_time']:.4f}s")
    print(f"Std inference time: {batch_results['std_inference_time']:.4f}s")
    print(f"Total inference time: {batch_results['total_inference_time']:.2f}s")
    print(f"Images per second: {1.0/batch_results['mean_inference_time']:.2f}")
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Visualizations saved to: {viz_dir}")
    
    return batch_results

def test_script_functionality(checkpoint_path, num_test_images=5, evaluation_mode="auto_prompt", num_gt_points=3):
    """Test script functionality with a small subset of images"""
    print(f"\n{'='*60}")
    print("TESTING SCRIPT FUNCTIONALITY")
    print(f"{'='*60}")
    print(f"Evaluation mode: {evaluation_mode}")
    if evaluation_mode == "gt_points":
        print(f"Number of GT points: {num_gt_points}")
    
    # Test paths - use absolute paths
    test_img_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/img"
    test_ann_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/ann"
    
    # Verify paths exist
    if not os.path.exists(test_img_dir):
        print(f" Test image directory not found: {test_img_dir}")
        return False
    if not os.path.exists(test_ann_dir):
        print(f" Test annotation directory not found: {test_ann_dir}")
        return False
    
    print(f" Test directories found")
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f" Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f" Checkpoint found: {os.path.basename(checkpoint_path)}")
    
    # Test model loading
    try:
        print("Testing model loading...")
        # Change to sam2base directory for proper config resolution
        original_dir = os.getcwd()
        sam2base_dir = "/home/ptp/sam2/libs/sam2base"
        os.chdir(sam2base_dir)
        
        # Use relative path from sam2base directory - Large model config
        config_file = "configs/sam2/sam2_hiera_l.yaml"
        sam2_model = build_sam2(config_file, ckpt_path=checkpoint_path, device="cpu")
        sam2_model.to("cuda")
        sam2_model.eval()
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        print(" Model loaded successfully!")
        
        # Restore original directory
        os.chdir(original_dir)
        
        # Test with a few images
        test_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))][:num_test_images]
        print(f"Testing with {len(test_images)} images...")
        
        success_count = 0
        for img_file in test_images:
            img_path = os.path.join(test_img_dir, img_file)
            ann_path = os.path.join(test_ann_dir, img_file + '.json')
            
            if not os.path.exists(ann_path):
                continue
            
            # Load image and annotation
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Set image for SAM2 predictor
            sam2_predictor.set_image(image)
            
            gt_mask = get_binary_gt_mask(ann_path, (height, width))
            if gt_mask is None:
                continue
            
            # Test prompt generation and inference based on mode
            try:
                start_time = time.time()
                
                if evaluation_mode == "auto_prompt":
                    # Generate automatic prompts
                    auto_prompts = generate_auto_prompts(image)
                    pred_mask, num_masks = predict_with_auto_prompts(sam2_predictor, image, auto_prompts)
                elif evaluation_mode == "gt_points":
                    # Generate GT points
                    gt_points, gt_labels = generate_gt_points(gt_mask, num_gt_points)
                    pred_mask, num_masks = predict_with_gt_points(sam2_predictor, image, gt_points, gt_labels)
                elif evaluation_mode == "30_gt_points":
                    # Generate 30 GT points
                    gt_points, gt_labels = generate_30_gt_points(gt_mask)
                    pred_mask, num_masks = predict_with_gt_points(sam2_predictor, image, gt_points, gt_labels)
                
                inference_time = time.time() - start_time
                
                # Test metrics computation
                metrics = SegmentationMetrics()
                sample_metrics = metrics.compute_pixel_metrics(pred_mask, gt_mask)
                
                print(f"   {img_file}: IoU={sample_metrics['iou']:.3f}, Time={inference_time:.3f}s")
                success_count += 1
                
            except Exception as e:
                print(f"   {img_file}: Error during inference - {e}")
        
        print(f"\nTest completed: {success_count}/{len(test_images)} images processed successfully")
        
        if success_count > 0:
            print(" Script functionality verified! Ready for full evaluation.")
            return True
        else:
            print(" Script failed to process any images. Check errors above.")
            return False
            
    except Exception as e:
        print(f" Error during testing: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM2 Large on full Severstal dataset')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_large_full_dataset_lr0.0001_20250829_1646/best_step11000_iou0.5124_dice0.6461.torch',
                       help='Path to checkpoint file')
    parser.add_argument('--results_dir', type=str, 
                       default='evaluation_results/sam2_full_dataset',
                       help='Directory to save results')
    parser.add_argument('--num_viz', type=int, default=10,
                       help='Number of random samples to visualize')
    parser.add_argument('--test_only', action='store_true',
                       help='Only test script functionality without full evaluation')
    parser.add_argument('--evaluation_mode', type=str, 
                       choices=['auto_prompt', 'gt_points', '30_gt_points'],
                       default='auto_prompt',
                       help='Evaluation mode: auto_prompt (no GT), gt_points (1-3 GT points), or 30_gt_points (30 GT points)')
    parser.add_argument('--num_gt_points', type=int, default=3,
                       help='Number of GT points to use when evaluation_mode is gt_points (1-3)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.evaluation_mode == 'gt_points' and (args.num_gt_points < 1 or args.num_gt_points > 3):
        print("Error: num_gt_points must be between 1 and 3 when evaluation_mode is gt_points")
        return
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Test mode
    if args.test_only:
        test_script_functionality(args.checkpoint, evaluation_mode=args.evaluation_mode, num_gt_points=args.num_gt_points)
        return
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Available checkpoints:")
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.torch'):
                    print(f"  {os.path.join(checkpoint_dir, f)}")
        return
    
    # Run evaluation
    results = evaluate_sam2_large_full_dataset(
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        num_viz_samples=args.num_viz,
        evaluation_mode=args.evaluation_mode,
        num_gt_points=args.num_gt_points
    )
    
    if results:
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {args.results_dir}")
    else:
        print(f"\nEvaluation failed!")

if __name__ == "__main__":
    main()
