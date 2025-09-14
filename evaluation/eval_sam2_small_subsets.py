#!/usr/bin/env python3
"""
Evaluate SAM2 Small models trained on different subset sizes (100, 200, 500, 1000, 2000, 4000)
using the best checkpoints from training.

This script evaluates all fine-tuned SAM2 Small models on the complete test split
with SAM2-AutoPrompt (no GT, no external detector), comprehensive metrics, and visualizations.
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
import glob

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

def create_visualization_panel(image, gt_mask, pred_mask, img_name, model_name, save_dir):
    """Create comprehensive visualization panel with predictions vs GT"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name}: {img_name}', fontsize=16, fontweight='bold')
    
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
    safe_model_name = model_name.replace(' ', '_').replace('/', '_')
    viz_path = os.path.join(save_dir, f"{safe_model_name}_{img_name}_analysis.png")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

def find_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint based on IoU in filename"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for best_* checkpoints
    best_checkpoints = glob.glob(os.path.join(checkpoint_dir, "best_*.torch"))
    
    if not best_checkpoints:
        # Fallback to any .torch file
        all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.torch"))
        if all_checkpoints:
            return all_checkpoints[0]
        return None
    
    # Sort by IoU value in filename (best_step*_iou*.torch)
    def extract_iou(filename):
        try:
            # Extract IoU value from filename like "best_step1000_iou0.1234.torch"
            parts = os.path.basename(filename).split('_iou')
            if len(parts) > 1:
                return float(parts[1].replace('.torch', ''))
            return 0.0
        except:
            return 0.0
    
    best_checkpoints.sort(key=extract_iou, reverse=True)
    return best_checkpoints[0]

def evaluate_single_model(model_info, test_images, test_ann_dir, results_dir, num_viz_samples=5):
    """Evaluate a single SAM2 Small model on the test set"""
    model_path, model_name, subset_size, lr = model_info
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")
    print(f"Subset size: {subset_size}")
    print(f"Learning rate: {lr}")
    
    # Create model-specific results directory
    model_results_dir = os.path.join(results_dir, f"subset_{subset_size}")
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Create visualizations directory
    viz_dir = os.path.join(model_results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load model with checkpoint
    try:
        print("Loading fine-tuned SAM2 Small model...")
        # Change to sam2base directory for proper config resolution
        original_dir = os.getcwd()
        sam2base_dir = "/home/ptp/sam2/libs/sam2base"
        os.chdir(sam2base_dir)
        
        # Use relative path from sam2base directory
        config_file = "configs/sam2/sam2_hiera_s.yaml"
        sam2_model = build_sam2(config_file, ckpt_path=model_path, device="cpu")
        
        # Move to GPU and create predictor
        sam2_model.to("cuda")
        sam2_model.eval()
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("Model loaded successfully!")
        
        # Restore original directory
        os.chdir(original_dir)
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        # Restore original directory in case of error
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return None
    
    # Initialize metrics
    metrics_calculator = SegmentationMetrics(save_dir=model_results_dir)
    
    # Storage for evaluation
    all_predictions = []
    all_ground_truths = []
    all_ious = []
    inference_times = []
    
    # Random sample for visualization
    viz_indices = random.sample(range(len(test_images)), min(num_viz_samples, len(test_images)))
    
    print(f"\nStarting evaluation on {len(test_images)} test images...")
    print(f"Will create visualizations for {len(viz_indices)} random samples")
    print(f"Evaluation mode: SAM2-AutoPrompt (no GT, no external detector)")
    
    # Evaluation loop
    for idx, img_file in enumerate(tqdm(test_images, desc=f"Evaluating {model_name}")):
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
        
        # Generate automatic prompts from image (no GT)
        auto_prompts = generate_auto_prompts(image)
        
        # Inference with timing - using auto-prompts
        start_time = time.time()
        
        try:
            # Predict with SAM2 using automatic prompts
            pred_mask, num_masks = predict_with_auto_prompts(sam2_predictor, image, auto_prompts)
            
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
            viz_path = create_visualization_panel(image, gt_mask, pred_mask, img_name, model_name, viz_dir)
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
    
    # Prepare model info for CSV
    model_info_csv = {
        'model': 'sam2_small',
        'variant': f'subset_{subset_size}',
        'prompt_type': 'sam2_autoprompt',
        'checkpoint': os.path.basename(model_path),
        'dataset': f'severstal_subset_{subset_size}',
        'split': 'test',
        'subset_size': subset_size,
        'learning_rate': lr
    }
    
    # Additional experiment info
    additional_info = {
        'num_samples': len(all_predictions),
        'image_size': f"{height}x{width}",
        'prompt_strategy': 'sam2_autoprompt_no_gt_no_detector',
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results to CSV
    experiment_name = f"eval_sam2_small_subset_{subset_size}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    csv_path = metrics_calculator.save_results_to_csv(
        batch_results, experiment_name, model_info_csv, additional_info
    )
    
    # Print comprehensive summary
    metrics_calculator.print_metrics_summary(batch_results, f"SAM2 Small (Subset {subset_size})")
    
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
    
    # Clean up GPU memory
    del sam2_model, sam2_predictor
    torch.cuda.empty_cache()
    
    return batch_results

def find_subset_models():
    """Find all trained subset models and their best checkpoints"""
    models_dir = "/home/ptp/sam2/new_src/training/training_results/sam2_subsets"
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return []
    
    subset_models = []
    
    # Look for subset directories - ONLY Small models
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        
        # Only process Small models
        if os.path.isdir(item_path) and 'subset_' in item and 'sam2_small' in item:
            # Extract subset size from directory name
            # Example: sam2_small_subset_100_lr1e4_20241201_1200
            parts = item.split('_')
            subset_size = None
            lr = None
            
            for i, part in enumerate(parts):
                if part == 'subset' and i + 1 < len(parts):
                    try:
                        subset_size = int(parts[i + 1])
                    except ValueError:
                        continue
                elif part.startswith('lr'):
                    try:
                        lr = part.replace('lr', '')
                        # Convert to readable format
                        if 'e' in lr:
                            lr = f"1e-{lr.split('e')[1]}"
                        else:
                            lr = f"0.{lr}"
                        # Clean up any double dots
                        lr = lr.replace('0.0.', '0.')
                    except ValueError:
                        continue
            
            if subset_size is not None and lr is not None:
                # Find best checkpoint
                best_ckpt = find_best_checkpoint(item_path)
                if best_ckpt:
                    model_name = f"SAM2 Small - Subset {subset_size} - LR {lr}"
                    subset_models.append({
                        'checkpoint_path': best_ckpt,
                        'model_name': model_name,
                        'subset_size': subset_size,
                        'learning_rate': lr,
                        'model_dir': item_path
                    })
                    print(f"Found model: {model_name}")
                    print(f"  Checkpoint: {os.path.basename(best_ckpt)}")
                    print(f"  Directory: {item}")
    
    # Sort by subset size
    subset_models.sort(key=lambda x: x['subset_size'])
    
    return subset_models

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM2 Small models on different subset sizes')
    parser.add_argument('--results_dir', type=str, 
                       default='/home/ptp/sam2/new_src/evaluation/evaluation_results/sam2_subsets',
                       help='Directory to save results')
    parser.add_argument('--num_viz', type=int, default=5,
                       help='Number of random samples to visualize per model')
    parser.add_argument('--subset_sizes', nargs="+", 
                       default=["100", "200", "500", "1000", "2000", "4000"],
                       help='Specific subset sizes to evaluate (default: all)')
    parser.add_argument('--test_only', action='store_true',
                       help='Only test script functionality without full evaluation')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Test split paths - use absolute paths
    global test_img_dir, test_ann_dir
    test_img_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/img"
    test_ann_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/ann"
    
    # Verify paths exist
    if not os.path.exists(test_img_dir):
        print(f"Error: Test image directory not found: {test_img_dir}")
        return
    if not os.path.exists(test_ann_dir):
        print(f"Error: Test annotation directory not found: {test_ann_dir}")
        return
    
    # Get test image files
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    print(f"Found {len(test_images)} test images")
    
    # Test mode
    if args.test_only:
        print("Testing script functionality...")
        subset_models = find_subset_models()
        if subset_models:
            print(f"Found {len(subset_models)} subset models")
            # Test with first model
            test_model = subset_models[0]
            print(f"Testing with: {test_model['model_name']}")
            test_result = evaluate_single_model(
                (test_model['checkpoint_path'], test_model['model_name'], 
                 test_model['subset_size'], test_model['learning_rate']),
                test_images[:10],  # Test with 10 images
                test_ann_dir,
                args.results_dir,
                num_viz_samples=2
            )
            if test_result:
                print(" Script functionality verified!")
            else:
                print(" Script failed!")
        return
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Find all subset models
    print("Searching for trained subset models...")
    subset_models = find_subset_models()
    
    if not subset_models:
        print("No subset models found!")
        return
    
    print(f"\nFound {len(subset_models)} subset models to evaluate")
    
    # Filter by requested subset sizes if specified
    if args.subset_sizes != ["100", "200", "500", "1000", "2000", "4000"]:
        requested_sizes = [int(s) for s in args.subset_sizes]
        subset_models = [m for m in subset_models if m['subset_size'] in requested_sizes]
        print(f"Filtered to {len(subset_models)} models for requested sizes: {args.subset_sizes}")
    
    # Evaluate each model
    all_results = {}
    start_time = time.time()
    
    for i, model_info in enumerate(subset_models):
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL {i+1}/{len(subset_models)}")
        print(f"{'='*80}")
        
        try:
            result = evaluate_single_model(
                (model_info['checkpoint_path'], model_info['model_name'], 
                 model_info['subset_size'], model_info['learning_rate']),
                test_images,
                test_ann_dir,
                args.results_dir,
                num_viz_samples=args.num_viz
            )
            
            if result:
                all_results[model_info['subset_size']] = {
                    'success': True,
                    'metrics': result,
                    'model_info': model_info
                }
            else:
                all_results[model_info['subset_size']] = {
                    'success': False,
                    'error': 'Evaluation failed'
                }
                
        except Exception as e:
            print(f"Error evaluating subset {model_info['subset_size']}: {e}")
            all_results[model_info['subset_size']] = {
                'success': False,
                'error': str(e)
            }
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Models evaluated: {len(subset_models)}")
    
    successful_results = {k: v for k, v in all_results.items() if v['success']}
    failed_results = {k: v for k, v in all_results.items() if not v['success']}
    
    print(f"Successful evaluations: {len(successful_results)}")
    print(f"Failed evaluations: {len(failed_results)}")
    
    if successful_results:
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"{'Subset':<8} {'mIoU':<8} {'Dice':<8} {'IoU@50':<8} {'IoU@75':<8} {'Time/img':<10}")
        print(f"{'='*80}")
        
        for subset_size in sorted(successful_results.keys()):
            result = successful_results[subset_size]['metrics']
            print(f"{subset_size:<8} {result.get('mean_iou', 0):<8.4f} "
                  f"{result.get('mean_dice', 0):<8.4f} "
                  f"{result.get('iou_at_50', 0):<8.2f}% "
                  f"{result.get('iou_at_75', 0):<8.2f}% "
                  f"{result.get('mean_inference_time', 0):<10.4f}s")
    
    if failed_results:
        print(f"\n{'='*80}")
        print(f"FAILED EVALUATIONS")
        print(f"{'='*80}")
        for subset_size, result in failed_results.items():
            print(f"Subset {subset_size}: {result['error']}")
    
    # Save overall results summary
    summary_file = os.path.join(args.results_dir, f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': total_time,
            'models_evaluated': len(subset_models),
            'successful_evaluations': len(successful_results),
            'failed_evaluations': len(failed_results),
            'results': all_results
        }, f, indent=2)
    
    print(f"\nOverall results summary saved to: {summary_file}")
    print(f"Individual results saved to: {args.results_dir}")

if __name__ == "__main__":
    main()
