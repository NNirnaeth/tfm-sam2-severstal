#!/usr/bin/env python3
"""
Evaluate Detectron2 Faster R-CNN model on test dataset with comprehensive metrics.

This script evaluates a trained Detectron2 model and provides:
- Detection metrics (AP, AP50, AP75, etc.)
- Visualization of predictions vs ground truth
- Option to save prediction examples
- Comprehensive reporting for analysis

Usage:
    python eval_detectron2.py --model_ckpt path/to/model.pth --img_dir path/to/images --ann_file path/to/annotations.json
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_test_loader

def set_seed(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(cfg, model_ckpt_path, device):
    """Load trained model from checkpoint"""
    # Build model
    model = build_model(cfg)
    
    # Load checkpoint
    print(f"Loading checkpoint: {model_ckpt_path}")
    checkpoint = torch.load(model_ckpt_path, map_location=device, weights_only=False)
    
    # Try different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Set to evaluation mode and configure for inference
    model.eval()
    
    # Configure model for inference (disable training components)
    for module in model.modules():
        if hasattr(module, 'training'):
            module.training = False
    
    print(f"Model loaded successfully")
    
    return model

def register_test_dataset(img_dir, ann_file, dataset_name="test_dataset"):
    """Register test dataset for evaluation"""
    # Clear existing dataset if it exists
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)
    
    # Register new dataset
    register_coco_instances(dataset_name, {}, ann_file, img_dir)
    
    # Set metadata
    MetadataCatalog.get(dataset_name).set(thing_classes=["defect"])
    
    print(f"Registered dataset: {dataset_name}")
    print(f"  Images: {img_dir}")
    print(f"  Annotations: {ann_file}")
    
    return dataset_name

def bbox_to_mask(bbox, height, width):
    """Convert bounding box to binary mask"""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Clamp coordinates to image boundaries
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    
    mask = np.zeros((height, width), dtype=np.uint8)
    if w > 0 and h > 0:
        mask[y:y+h, x:x+w] = 1
    
    return mask

def calculate_dice_metrics(model, dataset_dicts, device, max_samples=None):
    """Calculate DICE metrics by converting detections to masks"""
    print("Calculating DICE metrics...")
    
    if max_samples and len(dataset_dicts) > max_samples:
        dataset_dicts = random.sample(dataset_dicts, max_samples)
    
    dice_scores = []
    inference_times = []
    
    # Ensure model is in eval mode
    model.eval()
    
    for idx, d in enumerate(tqdm(dataset_dicts, desc="Calculating DICE")):
        try:
            # Load image
            img = cv2.imread(d["file_name"])
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img_rgb.shape[:2]
            
            # Prepare input in Detectron2 format - CRITICAL: Don't include gt_instances
            inputs = [{
                "image": torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device),
                "height": height,
                "width": width
                # IMPORTANT: Do NOT include gt_instances for inference
            }]
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                # Direct model call should work if model is properly in eval mode
                outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get predictions
            if isinstance(outputs, list) and len(outputs) > 0:
                predictions = outputs[0]
            else:
                predictions = outputs
            
            # Create ground truth mask from annotations
            gt_mask = np.zeros((height, width), dtype=np.uint8)
            for ann in d['annotations']:
                bbox = ann['bbox']  # [x, y, w, h]
                ann_mask = bbox_to_mask(bbox, height, width)
                gt_mask = np.maximum(gt_mask, ann_mask)
            
            # Create prediction mask from detections
            pred_mask = np.zeros((height, width), dtype=np.uint8)
            if 'instances' in predictions and len(predictions['instances']) > 0:
                for i in range(len(predictions['instances'])):
                    bbox = predictions['instances'].pred_boxes.tensor[i].cpu().numpy()  # [x1, y1, x2, y2]
                    # Convert to [x, y, w, h] format
                    x1, y1, x2, y2 = bbox
                    bbox_xywh = [x1, y1, x2-x1, y2-y1]
                    ann_mask = bbox_to_mask(bbox_xywh, height, width)
                    pred_mask = np.maximum(pred_mask, ann_mask)
            
            # Calculate DICE score
            intersection = np.sum(pred_mask * gt_mask)
            union = np.sum(pred_mask) + np.sum(gt_mask)
            dice = (2 * intersection) / (union + 1e-6)
            dice_scores.append(dice)
            
            # Debug: Print some examples
            if idx < 5:
                print(f"Image {idx}: GT_pixels={np.sum(gt_mask)}, Pred_pixels={np.sum(pred_mask)}, "
                      f"Intersection={intersection}, DICE={dice:.4f}")
            
        except Exception as e:
            print(f"Error calculating DICE for image {d['file_name']}: {e}")
            continue
    
    return {
        'mean_dice': np.mean(dice_scores) if dice_scores else 0.0,
        'std_dice': np.std(dice_scores) if dice_scores else 0.0,
        'avg_inference_time': np.mean(inference_times) if inference_times else 0.0,
        'total_inference_time': np.sum(inference_times) if inference_times else 0.0,
        'num_images': len(dataset_dicts)
    }

def evaluate_model_detectron2(model, cfg, dataset_name, device, output_dir=None, save_visualizations=False, max_samples=None):
    """Evaluate Detectron2 model and return comprehensive metrics"""
    model.eval()
    
    # Get dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    if max_samples and len(dataset_dicts) > max_samples:
        print(f"Limiting evaluation to {max_samples} samples")
        dataset_dicts = random.sample(dataset_dicts, max_samples)
        # Update dataset in catalog
        DatasetCatalog.remove(dataset_name)
        DatasetCatalog.register(dataset_name, lambda: dataset_dicts)
    
    print(f"Evaluating on {len(dataset_dicts)} test images...")
    
    # Calculate detection metrics using Detectron2's built-in evaluator
    print("Calculating detection metrics...")
    
    # Create COCO evaluator
    evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
    
    # Build test loader
    mapper = DatasetMapper(cfg, is_train=False)
    test_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    # Run evaluation
    results = inference_on_dataset(model, test_loader, evaluator)
    
    # Extract key metrics and print raw results for debugging
    print("Raw evaluation results:")
    print(results)
    
    if 'bbox' in results:
        bbox_results = results['bbox']
        # Convert from percentage to decimal for consistency
        metrics = {
            'AP': bbox_results.get('AP', 0.0) / 100.0,
            'AP50': bbox_results.get('AP50', 0.0) / 100.0,
            'AP75': bbox_results.get('AP75', 0.0) / 100.0,
            'APs': bbox_results.get('APs', 0.0) / 100.0,
            'APm': bbox_results.get('APm', 0.0) / 100.0,
            'APl': bbox_results.get('APl', 0.0) / 100.0,
            'AR1': bbox_results.get('AR1', 0.0) / 100.0,
            'AR10': bbox_results.get('AR10', 0.0) / 100.0,
            'AR100': bbox_results.get('AR100', 0.0) / 100.0,
            'ARs': bbox_results.get('ARs', 0.0) / 100.0,
            'ARm': bbox_results.get('ARm', 0.0) / 100.0,
            'ARl': bbox_results.get('ARl', 0.0) / 100.0,
        }
    else:
        metrics = {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0}
    
    # Calculate DICE metrics
    dice_metrics = calculate_dice_metrics(model, dataset_dicts, device, max_samples)
    metrics.update(dice_metrics)
    
    # Save visualizations if requested
    if save_visualizations and output_dir:
        print("Generating visualizations...")
        save_sample_visualizations(model, dataset_dicts, metadata, output_dir, max_samples=20)
    
    return metrics, results

def save_sample_visualizations(model, dataset_dicts, metadata, output_dir, max_samples=20):
    """Save sample visualizations for a subset of images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select sample images
    sample_indices = random.sample(range(len(dataset_dicts)), min(max_samples, len(dataset_dicts)))
    
    for idx in tqdm(sample_indices, desc="Generating visualizations"):
        try:
            d = dataset_dicts[idx]
            
            # Load image
            img = cv2.imread(d["file_name"])
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Prepare input
            height, width = img_rgb.shape[:2]
            inputs = [{
                "image": torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(next(model.parameters()).device),
                "height": height,
                "width": width
            }]
            
            # Run inference
            with torch.no_grad():
                outputs = model(inputs)
            
            # Get predictions
            if isinstance(outputs, list) and len(outputs) > 0:
                predictions = outputs[0]
            else:
                predictions = outputs
            
            # Save visualization
            save_prediction_visualization(img_rgb, d, predictions, metadata, output_dir, idx)
            
        except Exception as e:
            print(f"Error generating visualization for image {idx}: {e}")
            continue

def save_prediction_visualization(img, ground_truth, predictions, metadata, output_dir, idx):
    """Save visualization comparing ground truth and predictions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Ground truth - manual drawing for consistency
    try:
        gt_img = img.copy()
        num_gt = 0
        
        if 'annotations' in ground_truth:
            for ann in ground_truth['annotations']:
                bbox = ann['bbox']  # [x, y, w, h] format
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Draw rectangle
                cv2.rectangle(gt_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Add label
                cv2.putText(gt_img, 'defect', (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                num_gt += 1
        
        axes[1].imshow(gt_img)
        axes[1].set_title('Ground Truth', fontsize=14)
        axes[1].axis('off')
        
    except Exception as e:
        print(f"Warning: Could not visualize ground truth for image {idx}: {e}")
        axes[1].imshow(img)
        axes[1].set_title('Ground Truth (Error)', fontsize=14)
        axes[1].axis('off')
        num_gt = 0
    
    # Predictions - manual drawing since Detectron2's visualizer has issues
    try:
        # Create a copy of the image for drawing predictions
        pred_img = img.copy()
        
        num_preds = 0
        if 'instances' in predictions and len(predictions['instances']) > 0:
            instances = predictions['instances']
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            
            # Draw bounding boxes manually
            for i, (box, score) in enumerate(zip(boxes, scores)):
                if score > 0.5:  # Only show confident predictions
                    x1, y1, x2, y2 = box.astype(int)
                    # Draw rectangle
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add score text
                    cv2.putText(pred_img, f'{score:.2f}', (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    num_preds += 1
        
        axes[2].imshow(pred_img)
        axes[2].set_title('Predictions', fontsize=14)
        axes[2].axis('off')
        
        fig.suptitle(f'Image {idx}: GT={num_gt} objects, Pred={num_preds} objects', fontsize=16)
        
    except Exception as e:
        print(f"Warning: Could not visualize predictions for image {idx}: {e}")
        axes[2].imshow(img)
        axes[2].set_title('Predictions (Error)', fontsize=14)
        axes[2].axis('off')
        fig.suptitle(f'Image {idx}: GT={num_gt} objects, Pred=Error', fontsize=16)
    
    # Save
    filename = f"detection_example_{idx:04d}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {filename}")

def save_detailed_visualization(img, ground_truth, predictions, metadata, output_dir, idx):
    """Save detailed visualization with bounding box information"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Ground truth with bbox info
    visualizer_gt = Visualizer(img, metadata=metadata, scale=1.0)
    vis_gt = visualizer_gt.draw_dataset_dict(ground_truth)
    axes[0, 1].imshow(vis_gt.get_image())
    axes[0, 1].set_title(f'Ground Truth ({len(ground_truth["annotations"])} objects)', fontsize=14)
    axes[0, 1].axis('off')
    
    # Predictions with bbox info
    visualizer_pred = Visualizer(img, metadata=metadata, scale=1.0)
    vis_pred = visualizer_pred.draw_instance_predictions(predictions)
    axes[1, 0].imshow(vis_pred.get_image())
    axes[1, 0].set_title(f'Predictions ({len(predictions["instances"])} objects)', fontsize=14)
    axes[1, 0].axis('off')
    
    # Combined overlay
    visualizer_combined = Visualizer(img, metadata=metadata, scale=1.0)
    # Draw GT in red
    vis_combined = visualizer_combined.draw_dataset_dict(ground_truth)
    # Draw predictions in blue
    vis_combined = visualizer_combined.draw_instance_predictions(predictions)
    axes[1, 1].imshow(vis_combined.get_image())
    axes[1, 1].set_title('GT (Red) + Predictions (Blue)', fontsize=14)
    axes[1, 1].axis('off')
    
    # Add detailed info
    info_text = f"Image {idx}\n"
    info_text += f"GT objects: {len(ground_truth['annotations'])}\n"
    info_text += f"Pred objects: {len(predictions['instances'])}\n"
    
    if len(predictions['instances']) > 0:
        scores = predictions['instances'].scores.cpu().numpy()
        info_text += f"Avg confidence: {scores.mean():.3f}\n"
        info_text += f"Max confidence: {scores.max():.3f}\n"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save
    filename = f"detailed_detection_{idx:04d}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def save_results_csv(metrics, output_dir, model_name):
    """Save evaluation results to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results row
    results_row = {
        'model': model_name,
        'AP': metrics['AP'],
        'AP50': metrics['AP50'],
        'AP75': metrics['AP75'],
        'APs': metrics['APs'],
        'APm': metrics['APm'],
        'APl': metrics['APl'],
        'AR1': metrics['AR1'],
        'AR10': metrics['AR10'],
        'AR100': metrics['AR100'],
        'ARs': metrics['ARs'],
        'ARm': metrics['ARm'],
        'ARl': metrics['ARl'],
        'mean_dice': metrics.get('mean_dice', 0.0),
        'std_dice': metrics.get('std_dice', 0.0),
        'avg_inference_time': metrics['avg_inference_time'],
        'total_inference_time': metrics['total_inference_time'],
        'num_images': metrics['num_images'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"detection_results_{model_name}.csv")
    df = pd.DataFrame([results_row])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return csv_path

def print_results(metrics):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*60)
    print("DETECTION EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nAverage Precision (AP) Metrics (0-1 scale):")
    print(f"  AP (IoU=0.50:0.95): {metrics['AP']:.4f} ({metrics['AP']*100:.1f}%)")
    print(f"  AP50 (IoU=0.50):    {metrics['AP50']:.4f} ({metrics['AP50']*100:.1f}%)")
    print(f"  AP75 (IoU=0.75):    {metrics['AP75']:.4f} ({metrics['AP75']*100:.1f}%)")
    
    print(f"\nAP by Object Size:")
    print(f"  APs (small):        {metrics['APs']:.4f} ({metrics['APs']*100:.1f}%)")
    print(f"  APm (medium):       {metrics['APm']:.4f} ({metrics['APm']*100:.1f}%)")
    print(f"  APl (large):        {metrics['APl']:.4f} ({metrics['APl']*100:.1f}%)")
    
    print(f"\nAverage Recall (AR) Metrics:")
    print(f"  AR@1:               {metrics['AR1']:.4f} ({metrics['AR1']*100:.1f}%)")
    print(f"  AR@10:              {metrics['AR10']:.4f} ({metrics['AR10']*100:.1f}%)")
    print(f"  AR@100:             {metrics['AR100']:.4f} ({metrics['AR100']*100:.1f}%)")
    
    print(f"\nAR by Object Size:")
    print(f"  ARs (small):        {metrics['ARs']:.4f} ({metrics['ARs']*100:.1f}%)")
    print(f"  ARm (medium):       {metrics['ARm']:.4f} ({metrics['ARm']*100:.1f}%)")
    print(f"  ARl (large):        {metrics['ARl']:.4f} ({metrics['ARl']*100:.1f}%)")
    
    print(f"\nDICE Metrics (Bounding Box to Mask):")
    print(f"  Mean DICE:          {metrics.get('mean_dice', 0.0):.4f}")
    print(f"  Std DICE:           {metrics.get('std_dice', 0.0):.4f}")
    
    print(f"\nPerformance:")
    print(f"  Number of images:   {metrics['num_images']}")
    print(f"  Avg inference time: {metrics['avg_inference_time']:.4f}s")
    print(f"  Total time:         {metrics['total_inference_time']:.2f}s")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Detectron2 model on test dataset')
    
    # Model configuration
    parser.add_argument('--model_ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to config file (if different from training)')
    
    # Data paths
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Path to test images directory')
    parser.add_argument('--ann_file', type=str, required=True,
                       help='Path to COCO format annotations JSON file')
    
    # Evaluation settings
    parser.add_argument('--score_thresh', type=float, default=0.5,
                       help='Score threshold for detections')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save prediction visualizations')
    parser.add_argument('--save_detailed', action='store_true',
                       help='Save detailed visualizations with bbox info')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./detection_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default='faster_rcnn',
                       help='Model name for output files')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Register test dataset
    dataset_name = register_test_dataset(args.img_dir, args.ann_file)
    
    # Load configuration
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    else:
        # Use default Faster R-CNN config
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Update configuration for evaluation - CRITICAL SETTINGS
    cfg.MODEL.WEIGHTS = args.model_ckpt
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 defect class
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.OUTPUT_DIR = args.output_dir
    
    # CRITICAL: Set model to evaluation mode in config
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
    
    # Load model
    model = load_model(cfg, args.model_ckpt, device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully. Total parameters: {total_params:,}")
    
    # Evaluate model
    print(f"\nEvaluating {args.model_name}...")
    metrics, detailed_results = evaluate_model_detectron2(
        model=model,
        cfg=cfg,
        dataset_name=dataset_name,
        device=device,
        output_dir=args.output_dir,
        save_visualizations=args.save_visualizations or args.save_detailed,
        max_samples=args.max_samples
    )
    
    # Print results
    print_results(metrics)
    
    # Save results
    csv_path = save_results_csv(metrics, args.output_dir, args.model_name)
    
    # Save detailed results as JSON
    json_path = os.path.join(args.output_dir, f"detailed_results_{args.model_name}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for key, value in detailed_results.items():
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
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {csv_path}")
    print(f"Detailed results saved to: {json_path}")
    
    if args.save_visualizations or args.save_detailed:
        print(f"Visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
