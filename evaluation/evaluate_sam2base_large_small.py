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

# Add paths
sys.path.append('../../libs/sam2base')
sys.path.append('../../new_src/utils')  # To access metrics.py
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

def evaluate_sam2_base_model(config_file, model_name, results_dir):
    """Evaluate a base SAM2 model without fine-tuning"""
    
    print(f"\nEvaluating {model_name} (Base Model - No Fine-tuning)...")
    
    # Create model-specific results directory
    model_results_dir = os.path.join(results_dir, model_name.lower().replace(' ', '_').replace('(', '').replace(')', ''))
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Create visualizations directory
    viz_dir = os.path.join(model_results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load model - NO checkpoint, using base model as-is
    try:
        print("Loading base SAM2 model...")
        # Build the base SAM2 model without loading any trained weights
        sam2_model = build_sam2(config_file, ckpt_path=None, device="cpu")
        
        # Move to GPU and create predictor
        sam2_model.to("cuda")
        sam2_model.eval()
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model {config_file}: {e}")
        return None
    
    # CORRECTED: Use our generated test split paths (relative to script location)
    test_img_dir = "../../datasets/Data/splits/test_split/img"
    test_ann_dir = "../../datasets/Data/splits/test_split/ann"
    
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
    max_visualizations = 50  # Limit visualizations to avoid disk space issues
    
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
            
            # Generate 30 point prompts
            points, point_labels = generate_30_point_prompts(gt_mask)
            
            if points is None:
                continue
            
            # Predict with SAM2 using 30 points
            with torch.no_grad():
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    multimask_output=False,
                    return_logits=True
                )
            
            if masks is not None and len(masks) > 0:
                pred_mask = masks[0]
                
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
        'images_processed': len(batch_metrics),
        'total_test_images': len(test_images),
        'visualizations_created': visualization_count,
        'evaluation_timestamp': datetime.now().isoformat()
    })
    
    # Guardar a CSV con nombre más descriptivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sam2base_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}"
    
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
    print("SAM2 Base Models Evaluation (No Fine-tuning)")
    print("=" * 80)
    print("This script evaluates SAM2 Small and Large models")
    print("as they come from the original training, without any")
    print("fine-tuning on the Severstal dataset.")
    print("=" * 80)
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"evaluation_results/sam2_base/sam2base_evaluation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Define base SAM2 models to evaluate
    base_models = [
        {
            "config": "configs/sam2/sam2_hiera_s.yaml",
            "name": "SAM2 Small Base (No Fine-tuning)"
        },
        {
            "config": "configs/sam2/sam2_hiera_l.yaml", 
            "name": "SAM2 Large Base (No Fine-tuning)"
        }
    ]
    
    print(f"Found {len(base_models)} base models to evaluate")
    
    # Estimate total time
    estimated_time_small = 2.5 * 60  # 2.5 hours in minutes
    estimated_time_large = 5.0 * 60  # 5 hours in minutes
    total_estimated = estimated_time_small + estimated_time_large
    
    print(f"\nEstimated evaluation time:")
    print(f"  SAM2 Small: ~{estimated_time_small/60:.1f} hours")
    print(f"  SAM2 Large: ~{estimated_time_large/60:.1f} hours")
    print(f"  Total: ~{total_estimated/60:.1f} hours")
    print(f"  (Actual time may vary based on GPU performance)")
    
    # Evaluate models sequentially
    start_time = time.time()
    results = []
    
    for i, model_config in enumerate(tqdm(base_models, desc="Evaluating base models")):
        print(f"\n{'='*60}")
        print(f"Starting evaluation of {model_config['name']} ({i+1}/{len(base_models)})")
        print(f"{'='*60}")
        
        model_start_time = time.time()
        
        result = evaluate_sam2_base_model(
            model_config["config"], 
            model_config["name"],
            results_dir
        )
        
        if result is not None:
            results.append(result)
            
            # Calculate model evaluation time
            model_time = time.time() - model_start_time
            print(f"Model evaluation completed in {model_time/60:.1f} minutes")
        
        # Save intermediate results after each model
        intermediate_file = os.path.join(results_dir, f"intermediate_results_{i+1}.json")
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
    print("-" * 100)
    print(f"{'Model':<50} {'IoU':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Images':<8} {'Time':<8}")
    print("-" * 100)
    
    for result in results:
        model_name = result['model_name']
        # Get F1 score with fallback
        f1_key = None
        for key in ['mean_f1_score', 'mean_f1', 'f1_score']:
            if key in result:
                f1_key = key
                break
        f1_value = result.get(f1_key, 0.0)
        
        print(f"{model_name:<50} {result.get('mean_iou', 0.0):<8.4f} {f1_value:<8.4f} "
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
            
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"  IoU: {result.get('mean_iou', 0.0):.4f}\n")
            f.write(f"  F1: {f1_value:.4f}\n")
            f.write(f"  Precision: {result.get('mean_precision', 0.0):.4f}\n")
            f.write(f"  Recall: {result.get('mean_recall', 0.0):.4f}\n")
            f.write(f"  Images processed: {result.get('images_processed', 0)}\n\n")
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()



