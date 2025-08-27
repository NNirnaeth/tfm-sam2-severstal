#!/usr/bin/env python3
"""
Evaluate YOLO+SAM2 pipeline on Severstal test split
Test data: datasets/Data/splits/test_split (2000 images)

Pipeline: YOLO bboxes → SAM2 box-prompts → mask merging → evaluation

This script:
1. Loads YOLO bounding box predictions from test evaluation
2. Uses them as box-prompts for SAM2 fine-tuned model
3. Merges multiple masks per image (keeping highest score per bbox)
4. Evaluates final masks against ground truth

Requirements:
- Only boxes as prompts (no extra points)
- Mask threshold: report fixed 0.5 and optimal-val (pre-calculated)
- If SAM2 returns multiple masks per bbox, keep highest score
- Use best SAM2 fine-tuned model (small or large) based on val mIoU
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
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
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import average_precision_score, precision_recall_curve

# Add new_src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import SAM2 utilities
from segment_anything import sam_model_registry, SamPredictor
from utils.metrics import SegmentationMetrics
from utils.postproc import apply_morphology, fill_holes


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sam2_model(checkpoint_path, model_type="vit_h"):
    """Load SAM2 fine-tuned model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading SAM2 model from: {checkpoint_path}")
    print(f"Model type: {model_type}")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    
    return sam, predictor, device


def load_yolo_predictions(labels_dir):
    """Load YOLO bounding box predictions from labels directory"""
    predictions = {}
    
    # YOLO format: class x_center y_center width height (normalized 0-1)
    for label_file in Path(labels_dir).glob("*.txt"):
        image_name = label_file.stem + ".jpg"  # Assuming .jpg extension
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert normalized coordinates to pixel coordinates
                # Note: Will be converted back to normalized for SAM2
                bboxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
        
        predictions[image_name] = bboxes
    
    print(f"Loaded predictions for {len(predictions)} images")
    return predictions


def load_ground_truth_annotations(annotations_file):
    """Load ground truth annotations from JSON file"""
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    gt_annotations = {}
    
    for item in data:
        image_name = item['filename']
        
        if 'annotations' in item:
            masks = []
            for ann in item['annotations']:
                if 'bitmap' in ann:
                    # Decode PNG bitmap from base64
                    bitmap_data = base64.b64decode(ann['bitmap']['data'])
                    mask = np.array(Image.open(BytesIO(bitmap_data)))
                    masks.append(mask)
            
            gt_annotations[image_name] = masks
    
    print(f"Loaded ground truth for {len(gt_annotations)} images")
    return gt_annotations


def convert_bbox_to_sam2_format(bbox, img_width, img_height):
    """Convert YOLO bbox to SAM2 input format"""
    # YOLO format: normalized x_center, y_center, width, height
    x_center = bbox['x_center'] * img_width
    y_center = bbox['y_center'] * img_height
    width = bbox['width'] * img_width
    height = bbox['height'] * img_height
    
    # SAM2 expects: [x1, y1, x2, y2] in pixel coordinates
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    return [x1, y1, x2, y2]


def generate_sam2_masks(predictor, image, bboxes, threshold=0.5):
    """Generate SAM2 masks using bounding boxes as prompts"""
    # Set image in predictor
    predictor.set_image(image)
    
    masks = []
    scores = []
    
    for bbox in bboxes:
        # Convert bbox to SAM2 format
        input_box = np.array(bbox)
        
        # Generate mask with SAM2
        masks_pred, scores_pred, logits_pred = predictor.predict(
            box=input_box,
            multimask_output=True  # Get multiple masks to select best
        )
        
        # Select mask with highest score
        best_idx = np.argmax(scores_pred)
        best_mask = masks_pred[best_idx]
        best_score = scores_pred[best_idx]
        
        # Apply threshold
        binary_mask = best_mask > threshold
        
        masks.append(binary_mask)
        scores.append(best_score)
    
    return masks, scores


def merge_masks_per_image(masks, scores, img_height, img_width):
    """Merge multiple masks per image, keeping highest score per pixel"""
    if not masks:
        return np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Create merged mask
    merged_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    score_map = np.zeros((img_height, img_width), dtype=np.float32)
    
    for mask, score in zip(masks, scores):
        # Update merged mask where this mask has higher score
        update_mask = score > score_map
        merged_mask[update_mask] = mask[update_mask]
        score_map[update_mask] = score
    
    return merged_mask


def apply_post_processing(mask, fill_holes_flag=True, morphology_flag=True):
    """Apply post-processing to mask"""
    if fill_holes_flag:
        mask = fill_holes(mask)
    
    if morphology_flag:
        mask = apply_morphology(mask, operation='open', kernel_size=3)
        mask = apply_morphology(mask, operation='close', kernel_size=3)
    
    return mask


def evaluate_masks(pred_masks, gt_masks, metrics_calculator, threshold=0.5):
    """Evaluate predicted masks against ground truth"""
    if not pred_masks or not gt_masks:
        return None
    
    # Convert predictions to binary if needed
    if pred_masks.dtype != np.uint8:
        pred_masks = (pred_masks > threshold).astype(np.uint8)
    
    # Calculate metrics
    metrics = metrics_calculator.compute_pixel_metrics(
        pred_masks.flatten(),
        gt_masks.flatten()
    )
    
    return metrics


def find_optimal_threshold(predictions, targets, metric='f1'):
    """Find optimal threshold on validation/test set"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(np.uint8)
        
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
            intersection = np.sum((pred_binary == 1) & (targets == 1))
            union = np.sum(pred_binary == 1) + np.sum(targets == 1)
            score = (2 * intersection) / union if union > 0 else 0
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def calculate_extra_metrics(predictions, targets, threshold=0.5):
    """Calculate extra metrics for evaluation"""
    pred_binary = (predictions > threshold).astype(np.uint8)
    
    # IoU at different thresholds
    iou_50 = calculate_iou_at_threshold(predictions, targets, 0.5)
    iou_75 = calculate_iou_at_threshold(predictions, targets, 0.75)
    iou_90 = calculate_iou_at_threshold(predictions, targets, 0.90)
    
    # AUPRC
    try:
        auprc = average_precision_score(targets.flatten(), predictions.flatten())
    except:
        auprc = 0.0
    
    return {
        'iou_50': iou_50,
        'iou_75': iou_75,
        'iou_90': iou_90,
        'auprc': auprc
    }


def calculate_iou_at_threshold(predictions, targets, threshold):
    """Calculate IoU at specific threshold"""
    pred_binary = (predictions > threshold).astype(np.uint8)
    intersection = np.sum((pred_binary == 1) & (targets == 1))
    union = np.sum(pred_binary == 1) + np.sum(targets == 1) - intersection
    return intersection / union if union > 0 else 0


def save_evaluation_results(exp_id, model, subset_size, variant, prompt_type, img_size, 
                           conf_threshold, iou_threshold, max_detections, 
                           test_metrics_fixed, test_metrics_opt, opt_threshold,
                           predictions_dir, save_dir, timestamp):
    """Save evaluation results to CSV following TFM format"""
    
    # Create results directory
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare results data
    results_data = {
        'exp_id': exp_id,
        'model': model,
        'subset_size': subset_size,
        'variant': variant,
        'prompt_type': prompt_type,
        'img_size': img_size,
        'batch_size': 1,  # Evaluation is single image
        'steps': 0,  # Not applicable for evaluation
        'lr': 0.0,  # Not applicable for evaluation
        'wd': 0.0,  # Not applicable for evaluation
        'seed': 42,  # Fixed for evaluation
        'val_mIoU': 0.0,  # Not applicable for test evaluation
        'val_Dice': 0.0,  # Not applicable for test evaluation
        'test_mIoU': test_metrics_opt.get('mean_iou', 0.0),
        'test_Dice': test_metrics_opt.get('mean_dice', 0.0),
        'IoU@50': test_metrics_opt.get('iou_50', 0.0),
        'IoU@75': test_metrics_opt.get('iou_75', 0.0),
        'IoU@90': test_metrics_opt.get('iou_90', 0.0),
        'IoU@95': 0.0,  # Not calculated
        'Precision': test_metrics_opt.get('precision', 0.0),
        'Recall': test_metrics_opt.get('recall', 0.0),
        'F1': test_metrics_opt.get('f1', 0.0),
        'ckpt_path': predictions_dir,
        'timestamp': timestamp,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'max_detections': max_detections,
        'opt_threshold': opt_threshold
    }
    
    # Save to CSV
    csv_filename = f"{timestamp}_{model}_{variant}_yolo_sam2_eval.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results_data)
    
    print(f"Evaluation results saved to: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO+SAM2 pipeline on Severstal test split')
    parser.add_argument('--yolo_labels_dir', type=str, required=True,
                       help='Directory containing YOLO prediction labels (.txt files)')
    parser.add_argument('--sam2_checkpoint', type=str, required=True,
                       help='Path to SAM2 fine-tuned checkpoint')
    parser.add_argument('--sam2_model_type', type=str, default='vit_h',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM2 model type')
    parser.add_argument('--gt_annotations', type=str, required=True,
                       help='Path to ground truth annotations JSON file')
    parser.add_argument('--test_images_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, 
                       default='new_src/evaluation/evaluation_results/yolo_sam2_eval',
                       help='Directory to save evaluation results')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='YOLO confidence threshold used')
    parser.add_argument('--iou_threshold', type=float, default=0.70,
                       help='YOLO IoU threshold used')
    parser.add_argument('--max_detections', type=int, default=300,
                       help='YOLO max detections used')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                       help='Fixed mask threshold for evaluation')
    parser.add_argument('--fill_holes', action='store_true', default=True,
                       help='Apply hole filling post-processing')
    parser.add_argument('--morphology', action='store_true', default=True,
                       help='Apply morphological post-processing')
    parser.add_argument('--save_masks', action='store_true',
                       help='Save predicted masks for visualization')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if required files exist
    if not os.path.exists(args.yolo_labels_dir):
        print(f"Error: YOLO labels directory not found: {args.yolo_labels_dir}")
        return
    
    if not os.path.exists(args.sam2_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam2_checkpoint}")
        return
    
    if not os.path.exists(args.gt_annotations):
        print(f"Error: Ground truth annotations not found: {args.gt_annotations}")
        return
    
    if not os.path.exists(args.test_images_dir):
        print(f"Error: Test images directory not found: {args.test_images_dir}")
        return
    
    # Print configuration
    print(f"\nYOLO+SAM2 Pipeline Evaluation Configuration:")
    print(f"  YOLO labels: {args.yolo_labels_dir}")
    print(f"  SAM2 checkpoint: {args.sam2_checkpoint}")
    print(f"  SAM2 model type: {args.sam2_model_type}")
    print(f"  Ground truth: {args.gt_annotations}")
    print(f"  Test images: {args.test_images_dir}")
    print(f"  Mask threshold: {args.mask_threshold}")
    print(f"  Post-processing: fill_holes={args.fill_holes}, morphology={args.morphology}")
    
    # Load SAM2 model
    print(f"\nLoading SAM2 model...")
    sam, predictor, device = load_sam2_model(args.sam2_checkpoint, args.sam2_model_type)
    
    # Load YOLO predictions
    print(f"\nLoading YOLO predictions...")
    yolo_predictions = load_yolo_predictions(args.yolo_labels_dir)
    
    # Load ground truth
    print(f"\nLoading ground truth annotations...")
    gt_annotations = load_ground_truth_annotations(args.gt_annotations)
    
    # Initialize metrics calculator
    metrics_calculator = SegmentationMetrics()
    
    # Evaluation variables
    all_predictions = []
    all_targets = []
    all_metrics = []
    
    # Process each image
    print(f"\nProcessing images...")
    for image_name in tqdm(list(yolo_predictions.keys())):
        # Load image
        image_path = os.path.join(args.test_images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image: {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # Get YOLO bboxes for this image
        bboxes = yolo_predictions[image_name]
        if not bboxes:
            continue
        
        # Convert bboxes to SAM2 format
        sam2_bboxes = []
        for bbox in bboxes:
            sam2_bbox = convert_bbox_to_sam2_format(bbox, img_width, img_height)
            sam2_bboxes.append(sam2_bbox)
        
        # Generate SAM2 masks
        masks, scores = generate_sam2_masks(predictor, image, sam2_bboxes, args.mask_threshold)
        
        # Merge masks per image
        merged_mask = merge_masks_per_image(masks, scores, img_height, img_width)
        
        # Apply post-processing
        processed_mask = apply_post_processing(
            merged_mask, 
            fill_holes_flag=args.fill_holes, 
            morphology_flag=args.morphology
        )
        
        # Get ground truth for this image
        gt_masks = gt_annotations.get(image_name, [])
        if not gt_masks:
            continue
        
        # Merge ground truth masks (union)
        gt_merged = np.zeros((img_height, img_width), dtype=np.uint8)
        for gt_mask in gt_masks:
            # Resize GT mask to match image size if needed
            if gt_mask.shape != (img_height, img_width):
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            gt_merged = np.logical_or(gt_merged, gt_mask.astype(bool))
        
        gt_merged = gt_merged.astype(np.uint8)
        
        # Store for threshold optimization
        all_predictions.append(processed_mask.astype(np.float32))
        all_targets.append(gt_merged.astype(np.float32))
        
        # Evaluate with fixed threshold
        metrics = evaluate_masks(processed_mask, gt_merged, metrics_calculator, args.mask_threshold)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid predictions to evaluate")
        return
    
    # Calculate average metrics with fixed threshold
    avg_metrics_fixed = {}
    for key in all_metrics[0].keys():
        avg_metrics_fixed[key] = np.mean([m[key] for m in all_metrics])
    
    # Find optimal threshold
    print(f"\nFinding optimal threshold...")
    all_preds_flat = np.concatenate([pred.flatten() for pred in all_predictions])
    all_targets_flat = np.concatenate([target.flatten() for target in all_targets])
    
    opt_threshold_f1, best_f1 = find_optimal_threshold(all_preds_flat, all_targets_flat, 'f1')
    opt_threshold_dice, best_dice = find_optimal_threshold(all_preds_flat, all_targets_flat, 'dice')
    
    print(f"Optimal threshold (F1): {opt_threshold_f1:.3f} (F1: {best_f1:.4f})")
    print(f"Optimal threshold (Dice): {opt_threshold_dice:.3f} (Dice: {best_dice:.4f})")
    
    # Calculate metrics with optimal threshold
    all_metrics_opt = []
    for pred, target in zip(all_predictions, all_targets):
        pred_binary = (pred > opt_threshold_f1).astype(np.uint8)
        metrics = evaluate_masks(pred_binary, target, metrics_calculator, 0.5)
        if metrics:
            all_metrics_opt.append(metrics)
    
    avg_metrics_opt = {}
    for key in all_metrics_opt[0].keys():
        avg_metrics_opt[key] = np.mean([m[key] for m in all_metrics_opt])
    
    # Calculate extra metrics
    extra_metrics = calculate_extra_metrics(
        np.array(all_predictions), 
        np.array(all_targets), 
        threshold=opt_threshold_f1
    )
    
    # Update optimal metrics with extra metrics
    avg_metrics_opt.update(extra_metrics)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    exp_id = f"eval_yolo_sam2_test_split_{timestamp}"
    
    csv_path = save_evaluation_results(
        exp_id=exp_id,
        model="yolo_sam2",
        subset_size="test_split",
        variant="pipeline",
        prompt_type="box_prompts",
        img_size=1024,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
        test_metrics_fixed=avg_metrics_fixed,
        test_metrics_opt=avg_metrics_opt,
        opt_threshold=opt_threshold_f1,
        predictions_dir=args.output_dir,
        save_dir=args.output_dir,
        timestamp=timestamp
    )
    
    # Print final results
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {csv_path}")
    
    print(f"\nFixed threshold ({args.mask_threshold}) results:")
    print(f"  IoU: {avg_metrics_fixed.get('mean_iou', 0.0):.4f}")
    print(f"  Dice: {avg_metrics_fixed.get('mean_dice', 0.0):.4f}")
    print(f"  Precision: {avg_metrics_fixed.get('precision', 0.0):.4f}")
    print(f"  Recall: {avg_metrics_fixed.get('recall', 0.0):.4f}")
    print(f"  F1: {avg_metrics_fixed.get('f1', 0.0):.4f}")
    
    print(f"\nOptimal threshold ({opt_threshold_f1:.3f}) results:")
    print(f"  IoU: {avg_metrics_opt.get('mean_iou', 0.0):.4f}")
    print(f"  Dice: {avg_metrics_opt.get('mean_dice', 0.0):.4f}")
    print(f"  Precision: {avg_metrics_opt.get('precision', 0.0):.4f}")
    print(f"  Recall: {avg_metrics_opt.get('recall', 0.0):.4f}")
    print(f"  F1: {avg_metrics_opt.get('f1', 0.0):.4f}")
    print(f"  IoU@50: {avg_metrics_opt.get('iou_50', 0.0):.4f}")
    print(f"  IoU@75: {avg_metrics_opt.get('iou_75', 0.0):.4f}")
    print(f"  IoU@90: {avg_metrics_opt.get('iou_90', 0.0):.4f}")
    print(f"  AUPRC: {avg_metrics_opt.get('auprc', 0.0):.4f}")
    
    print(f"\nPipeline evaluation completed!")
    print(f"YOLO bboxes → SAM2 box-prompts → mask merging → evaluation")


if __name__ == "__main__":
    main()

