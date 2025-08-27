#!/usr/bin/env python3
"""
Evaluate SAM2 Small model trained on full Severstal dataset
using the best checkpoint from training.

This script evaluates the fine-tuned SAM2 Small model on the complete test split
with consistent prompt generation, comprehensive metrics, and visualizations.
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
sys.path.append('../utils')  # To access metrics.py
from metrics import SegmentationMetrics

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

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

def generate_30_point_prompts(gt_mask):
    """Generate 30 point prompts from ground truth mask"""
    gt_coords = np.where(gt_mask)
    
    if len(gt_coords[0]) == 0:
        return None, None
    
    # Sample 30 points from the defect area
    num_points = min(30, len(gt_coords[0]))
    indices = np.linspace(0, len(gt_coords[0]) - 1, num_points, dtype=int)
    
    points = []
    point_labels = []
    
    for idx in indices:
        y = gt_coords[0][idx]
        x = gt_coords[1][idx]
        points.append([x, y])
        point_labels.append(1)  # All points are positive
    
    return np.array(points), np.array(point_labels)

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

def evaluate_sam2_small_full_dataset(checkpoint_path, results_dir, num_viz_samples=10):
    """Evaluate SAM2 Small model trained on full dataset"""
    
    print(f"\nEvaluating SAM2 Small (Fine-tuned on Full Dataset)...")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create visualizations directory
    viz_dir = os.path.join(results_dir, "visualizations")
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
        
        # Generate prompts (30 points from GT)
        points, point_labels = generate_30_point_prompts(gt_mask)
        if points is None:
            continue
        
        # Set image for SAM2 predictor
        sam2_predictor.set_image(image)
        
        # Inference with timing
        start_time = time.time()
        
        try:
            # Predict with SAM2
            masks, scores, logits = sam2_predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=False
            )
            
            # Get best mask
            pred_mask = masks[0] if len(masks) > 0 else np.zeros_like(gt_mask)
            
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
    
    # Prepare model info for CSV
    model_info = {
        'model': 'sam2_small',
        'variant': 'fine_tuned_full_dataset',
        'prompt_type': '30_points_from_gt',
        'checkpoint': os.path.basename(checkpoint_path),
        'dataset': 'severstal_full',
        'split': 'test'
    }
    
    # Additional experiment info
    additional_info = {
        'num_samples': len(all_predictions),
        'image_size': f"{height}x{width}",
        'prompt_strategy': '30_points_from_gt',
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results to CSV
    experiment_name = f"eval_sam2_small_full_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}"
    csv_path = metrics_calculator.save_results_to_csv(
        batch_results, experiment_name, model_info, additional_info
    )
    
    # Print comprehensive summary
    metrics_calculator.print_metrics_summary(batch_results, "SAM2 Small (Fine-tuned)")
    
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

def test_script_functionality(checkpoint_path, num_test_images=5):
    """Test script functionality with a small subset of images"""
    print(f"\n{'='*60}")
    print("TESTING SCRIPT FUNCTIONALITY")
    print(f"{'='*60}")
    
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
        
        # Use relative path from sam2base directory
        config_file = "configs/sam2/sam2_hiera_s.yaml"
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
            
            # Test prompt generation
            points, point_labels = generate_30_point_prompts(gt_mask)
            if points is None:
                continue
            
            # Set image for SAM2 predictor
            sam2_predictor.set_image(image)
            
            # Test inference
            try:
                start_time = time.time()
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    multimask_output=False
                )
                inference_time = time.time() - start_time
                
                pred_mask = masks[0] if len(masks) > 0 else np.zeros_like(gt_mask)
                
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
    parser = argparse.ArgumentParser(description='Evaluate SAM2 Small on full Severstal dataset')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/ptp/sam2/models/severstal_updated/sam2_small_full_dataset_lr0.0001_20250819_2209/best_step7000_iou2.2902.torch',
                       help='Path to checkpoint file')
    parser.add_argument('--results_dir', type=str, 
                       default='new_src/evaluation/evaluation_results/sam2_full_dataset',
                       help='Directory to save results')
    parser.add_argument('--num_viz', type=int, default=10,
                       help='Number of random samples to visualize')
    parser.add_argument('--test_only', action='store_true',
                       help='Only test script functionality without full evaluation')
    
    args = parser.parse_args()
    
    # Test mode
    if args.test_only:
        test_script_functionality(args.checkpoint)
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
    results = evaluate_sam2_small_full_dataset(
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        num_viz_samples=args.num_viz
    )
    
    if results:
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {args.results_dir}")
    else:
        print(f"\nEvaluation failed!")

if __name__ == "__main__":
    main()
