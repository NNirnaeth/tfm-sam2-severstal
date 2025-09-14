#!/usr/bin/env python3
"""
Evaluate YOLO+SAM2 pipeline on tiled Severstal test split
Test data: tiled dataset (800x256 tiles)

Pipeline: YOLO bboxes → SAM2 box-prompts → mask merging → evaluation

Features:
- Optimized for tiled data processing
- Recall-friendly YOLO threshold (to avoid missing instances before SAM2)
- Box preprocessing and merging for SAM2 input
- Comprehensive segmentation metrics (IoU, Dice, IoU@50/75/90, AUPRC)
- Memory-efficient batch processing
- Both detection and segmentation performance tracking

This script evaluates the complete YOLO detection + SAM2 segmentation pipeline
on tiled dataset with proper tile-specific optimizations and metrics.
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
import sys
sys.path.append('libs/sam2base')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.metrics import SegmentationMetrics
import cv2

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_memory():
    """Clean up GPU and CPU memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_sam2_model(checkpoint_path, model_type="sam2_hiera_l"):
    """Load SAM2 fine-tuned model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SAM2 model from: {checkpoint_path}")
    print(f"Model type: {model_type}")
    
    config_file = f"configs/sam2/{model_type}.yaml"
    sam2 = build_sam2(config_file=config_file, ckpt_path=checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2)
    
    return sam2, predictor, device

def load_yolo_predictions(labels_dir, conf_threshold=0.15):
    """Load YOLO bounding box predictions with recall-friendly threshold"""
    predictions = {}
    
    for label_file in Path(labels_dir).glob("*.txt"):
        image_name = label_file.stem + ".jpg"
        
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
                confidence = float(parts[5]) if len(parts) >= 6 else 1.0
                
                # Use lower threshold for recall-friendly detection
                if confidence >= conf_threshold:
                    bboxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'conf': confidence
                    })
        
        predictions[image_name] = bboxes
    
    print(f"Loaded predictions for {len(predictions)} images (conf≥{conf_threshold})")
    return predictions

def load_ground_truth_annotations(ann_dir):
    """Load ground truth annotations from Supervisely format JSON files"""
    gt = {}
    for jf in Path(ann_dir).glob("*.txt.json"):
        try:
            data = json.load(open(jf))
            objs = []
            for o in data.get("objects", []):
                bmp = o.get("bitmap", {})
                if "data" not in bmp or "origin" not in bmp:
                    continue
                raw = zlib.decompress(base64.b64decode(bmp["data"]))
                patch = np.array(Image.open(BytesIO(raw)))
                if patch.ndim == 3 and patch.shape[2] == 4:
                    patch = patch[:, :, 3]  # Alpha channel
                elif patch.ndim == 3:
                    patch = patch[:, :, 0]  # First channel
                patch = (patch > 0).astype(np.uint8)
                ox, oy = map(int, bmp["origin"])  # origin = [x, y]
                objs.append((patch, (ox, oy)))
            
            # Extract image name from filename
            image_name = jf.stem.replace('.txt', '') + ".jpg"
            gt[image_name] = objs
        except Exception as e:
            print(f"Warning: Could not load {jf}: {e}")
            image_name = jf.stem.replace('.txt', '') + ".jpg"
            gt[image_name] = []
            continue
    
    print(f"Loaded ground truth for {len(gt)} images")
    return gt

def box_iou_xyxy(a, b):
    """Calculate IoU between two boxes in [x1,y1,x2,y2] format"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1+1) * max(0, y2-y1+1)
    area_a = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    area_b = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    union = area_a + area_b - inter
    return inter/union if union>0 else 0.0

def widen_slender_boxes(box, W, H, min_w_px=8, pad_px=6):
    """Widen very narrow boxes to give context to SAM2 (optimized for tiles)"""
    x1, y1, x2, y2 = map(int, box)
    if (x2-x1+1) < min_w_px:
        cx = (x1+x2)//2
        x1 = max(0, cx - (min_w_px//2) - pad_px)
        x2 = min(W-1, cx + (min_w_px//2) + pad_px)
    return [x1, y1, x2, y2]

def preprocess_yolo_boxes_for_sam2(yolo_boxes, W, H, iou_merge=0.85, topk=15,
                                   min_w_px=6, pad_px=4):
    """
    Preprocess YOLO boxes for SAM2 (optimized for tiled data):
    - Convert normalized coordinates to pixels
    - Merge overlapping boxes (high IoU)
    - Widen very narrow boxes
    - Keep top-k by confidence
    """
    # Convert normalized coordinates to pixels
    boxes = []
    for b in yolo_boxes:
        cx, cy, w, h = b["x_center"]*W, b["y_center"]*H, b["width"]*W, b["height"]*H
        x1 = max(0, int(round(cx - w/2)))
        y1 = max(0, int(round(cy - h/2)))
        x2 = min(W-1, int(round(cx + w/2)))
        y2 = min(H-1, int(round(cy + h/2)))
        if x2 <= x1 or y2 <= y1: 
            continue
        boxes.append({"xyxy": [x1, y1, x2, y2], "conf": b.get("conf", 1.0)})

    # Sort by confidence (highest first)
    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)
    
    # Merge overlapping boxes
    kept = []
    for b in boxes:
        merged = False
        for k in kept:
            if box_iou_xyxy(b["xyxy"], k["xyxy"]) >= iou_merge:
                # Merge by taking union
                x1 = min(b["xyxy"][0], k["xyxy"][0])
                y1 = min(b["xyxy"][1], k["xyxy"][1])
                x2 = max(b["xyxy"][2], k["xyxy"][2])
                y2 = max(b["xyxy"][3], k["xyxy"][3])
                k["xyxy"] = [x1, y1, x2, y2]
                k["conf"] = max(k["conf"], b["conf"])
                merged = True
                break
        if not merged:
            kept.append(b)

    # Widen narrow boxes and return top-k
    out = []
    for k in kept[:topk]:
        widened = widen_slender_boxes(k["xyxy"], W, H, min_w_px=min_w_px, pad_px=pad_px)
        out.append(widened)
    
    return out

def generate_sam2_masks(predictor, image, bboxes, threshold=0.5):
    """Generate SAM2 masks using bounding boxes as prompts"""
    predictor.set_image(image)
    
    masks = []
    scores = []
    probs = []
    
    for bbox in bboxes:
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
        best_logits = logits_pred[best_idx] if logits_pred is not None else None
        
        # Get probability map from logits
        if best_logits is not None:
            prob = torch.sigmoid(torch.from_numpy(best_logits)).numpy()
            if prob.shape != image.shape[:2]:
                prob = cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            prob = best_mask.astype(np.float32)
        
        masks.append(best_mask)
        scores.append(best_score)
        probs.append(prob)
    
    return masks, scores, probs

def merge_masks_per_image(masks, scores, probs, img_height, img_width):
    """Merge multiple masks per image, keeping highest probability per pixel"""
    if not masks:
        return np.zeros((img_height, img_width), dtype=np.uint8), np.zeros((img_height, img_width), dtype=np.float32)
    
    # Create merged probability map (max probability per pixel)
    merged_prob = np.zeros((img_height, img_width), dtype=np.float32)
    
    for prob in probs:
        if prob.shape != (img_height, img_width):
            prob = cv2.resize(prob, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        merged_prob = np.maximum(merged_prob, prob)
    
    # Create binary mask from probability threshold
    merged_mask = (merged_prob > 0.5).astype(np.uint8)
    
    return merged_mask, merged_prob

def build_gt_union(gt_objs, H, W):
    """Build ground truth union mask from patches with proper origin positioning"""
    union = np.zeros((H, W), np.uint8)
    for patch, (ox, oy) in gt_objs:
        ph, pw = patch.shape
        x1, y1 = max(0, ox), max(0, oy)
        x2, y2 = min(W, ox + pw), min(H, oy + ph)
        if x2 > x1 and y2 > y1:
            union[y1:y2, x1:x2] = np.maximum(
                union[y1:y2, x1:x2], patch[:(y2 - y1), :(x2 - x1)]
            )
    return union

def apply_post_processing(mask, fill_holes_flag=True, morphology_flag=True):
    """Apply post-processing to mask using OpenCV"""
    if fill_holes_flag:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if morphology_flag:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def evaluate_masks(pred_masks, gt_masks, metrics_calculator, threshold=0.5):
    """Evaluate predicted masks against ground truth"""
    if pred_masks is None or gt_masks is None:
        return None
    
    if pred_masks.dtype != np.uint8:
        pred_masks = (pred_masks > threshold).astype(np.uint8)
    
    try:
        metrics = metrics_calculator.compute_pixel_metrics(
            pred_masks.flatten(),
            gt_masks.flatten()
        )
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

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
                           conf_threshold, iou_threshold, test_metrics_opt, opt_threshold,
                           predictions_dir, save_dir, timestamp, all_predictions, all_targets,
                           all_metrics, image_names, fill_holes=False, morphology=False,
                           iou_merge=0.85, topk_boxes=15, min_w_px=6, pad_px=4):
    """Save evaluation results to CSV following TFM format"""
    
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_data = {
        'exp_id': exp_id,
        'model': model,
        'subset_size': subset_size,
        'variant': variant,
        'prompt_type': prompt_type,
        'img_size': img_size,
        'batch_size': 1,
        'steps': 0,
        'lr': 0.0,
        'wd': 0.0,
        'seed': 42,
        'val_mIoU': 0.0,
        'val_Dice': 0.0,
        'test_mIoU': test_metrics_opt.get('iou', 0.0),
        'test_Dice': test_metrics_opt.get('dice', 0.0),
        'IoU@50': test_metrics_opt.get('iou_50', 0.0),
        'IoU@75': test_metrics_opt.get('iou_75', 0.0),
        'IoU@90': test_metrics_opt.get('iou_90', 0.0),
        'IoU@95': 0.0,
        'Precision': test_metrics_opt.get('precision', 0.0),
        'Recall': test_metrics_opt.get('recall', 0.0),
        'F1': test_metrics_opt.get('f1', 0.0),
        'AUPRC': test_metrics_opt.get('auprc', 0.0),
        'ckpt_path': predictions_dir,
        'timestamp': timestamp,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'opt_threshold': opt_threshold,
        'postproc_fill_holes': fill_holes,
        'postproc_morphology': morphology,
        'preproc_iou_merge': iou_merge,
        'preproc_topk_boxes': topk_boxes,
        'preproc_min_w_px': min_w_px,
        'preproc_pad_px': pad_px,
        'tile_size': f"{img_size}x256"
    }
    
    csv_filename = f"{timestamp}_{model}_{variant}_yolo_sam2_tiled_eval.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results_data)
    
    print(f"Evaluation results saved to: {csv_path}")
    
    # Save per-image results
    per_image_csv = save_per_image_results(
        exp_id, "yolo_sam2_tiled", "pipeline", all_predictions, all_targets,
        all_metrics, image_names, results_dir, timestamp
    )
    
    return csv_path

def save_per_image_results(exp_id, model, variant, all_predictions, all_targets,
                          all_metrics, image_names, results_dir, timestamp):
    """Save per-image evaluation results to CSV"""
    if not all_metrics:
        return None
    
    per_image_data = []
    for i, metrics in enumerate(all_metrics):
        image_name = image_names[i] if i < len(image_names) else f"image_{i}"
        
        if i < len(all_predictions):
            pred_binary = (all_predictions[i] > 0.5).astype(np.uint8)
            target = all_targets[i].astype(np.uint8)
            
            tp = np.sum((pred_binary == 1) & (target == 1))
            fp = np.sum((pred_binary == 1) & (target == 0))
            fn = np.sum((pred_binary == 0) & (target == 1))
            
            per_image_data.append({
                'image_name': image_name,
                'image_id': i,
                'iou': metrics.get('iou', 0.0),
                'dice': metrics.get('dice', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1': metrics.get('f1', 0.0),
                'tp': tp,
                'fp': fp,
                'fn': fn
            })
    
    per_image_csv = f"{timestamp}_{model}_{variant}_per_image_results.csv"
    per_image_path = os.path.join(results_dir, per_image_csv)
    
    with open(per_image_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'image_id', 'iou', 'dice', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_data)
    
    print(f"Per-image results saved to: {per_image_path}")
    return per_image_path

def process_batch(batch_images, predictor, yolo_predictions, gt_annotations,
                 args, metrics_calculator, test_images_dir):
    """Process a batch of images and return results"""
    batch_predictions = []
    batch_targets = []
    batch_metrics = []
    batch_image_names = []
    
    for image_name in batch_images:
        # Load image
        image_path = os.path.join(test_images_dir, image_name)
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
        bboxes = yolo_predictions.get(image_name, [])
        
        if not bboxes:
            # No bboxes detected - create zero mask
            merged_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            merged_prob = np.zeros((img_height, img_width), dtype=np.float32)
        else:
            # Preprocess YOLO boxes for SAM2 (optimized for tiles)
            sam2_bboxes = preprocess_yolo_boxes_for_sam2(
                bboxes, img_width, img_height,
                iou_merge=args.iou_merge, topk=args.topk_boxes,
                min_w_px=args.min_w_px, pad_px=args.pad_px
            )
            
            if not sam2_bboxes:
                merged_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                merged_prob = np.zeros((img_height, img_width), dtype=np.float32)
            else:
                # Generate SAM2 masks
                masks, scores, probs = generate_sam2_masks(predictor, image, sam2_bboxes, args.mask_threshold)
                merged_mask, merged_prob = merge_masks_per_image(masks, scores, probs, img_height, img_width)
        
        # Apply post-processing
        processed_mask = apply_post_processing(
            merged_mask,
            fill_holes_flag=args.fill_holes,
            morphology_flag=args.morphology
        )
        
        # Get ground truth for this image
        gt_objs = gt_annotations.get(image_name, [])
        gt_merged = build_gt_union(gt_objs, img_height, img_width)
        
        # Store for threshold optimization
        batch_predictions.append(merged_prob)
        batch_targets.append(gt_merged.astype(np.float32))
        
        # Evaluate with fixed threshold
        metrics = evaluate_masks(processed_mask, gt_merged, metrics_calculator, args.mask_threshold)
        if metrics:
            batch_metrics.append(metrics)
        
        batch_image_names.append(image_name)
    
    return batch_predictions, batch_targets, batch_metrics, batch_image_names

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO+SAM2 pipeline on tiled Severstal test split')
    parser.add_argument('--yolo_labels_dir', type=str, required=True,
                       help='Directory containing YOLO prediction labels (.txt files)')
    parser.add_argument('--sam2_checkpoint', type=str, required=True,
                       help='Path to SAM2 fine-tuned checkpoint')
    parser.add_argument('--sam2_model_type', type=str, default='sam2_hiera_l',
                       choices=['sam2_hiera_l', 'sam2_hiera_b', 'sam2_hiera_t'],
                       help='SAM2 model type')
    parser.add_argument('--gt_annotations', type=str, required=True,
                       help='Path to ground truth annotations directory (Supervisely format JSON files)')
    parser.add_argument('--test_images_dir', type=str, required=True,
                       help='Directory containing test images (tiles)')
    parser.add_argument('--output_dir', type=str,
                       default='new_src/evaluation/evaluation_results/yolo_sam2_tiled',
                       help='Directory to save evaluation results')
    parser.add_argument('--conf_threshold', type=float, default=0.15,
                       help='YOLO confidence threshold (recall-friendly)')
    parser.add_argument('--iou_threshold', type=float, default=0.70,
                       help='YOLO IoU threshold used')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                       help='Fixed mask threshold for evaluation')
    parser.add_argument('--fill_holes', action='store_true', default=False,
                       help='Apply hole filling post-processing')
    parser.add_argument('--morphology', action='store_true', default=False,
                       help='Apply morphological post-processing')
    parser.add_argument('--iou_merge', type=float, default=0.85,
                       help='IoU threshold for merging duplicate boxes')
    parser.add_argument('--topk_boxes', type=int, default=15,
                       help='Maximum number of boxes per image after preprocessing')
    parser.add_argument('--min_w_px', type=int, default=6,
                       help='Minimum width in pixels for widening slender boxes')
    parser.add_argument('--pad_px', type=int, default=4,
                       help='Padding in pixels when widening slender boxes')
    parser.add_argument('--opt_threshold', type=float, default=None,
                       help='Pre-calculated optimal threshold from validation')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Number of images to process in each batch')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate inputs
    if not os.path.exists(args.yolo_labels_dir):
        print(f"Error: YOLO labels directory not found: {args.yolo_labels_dir}")
        return
    
    if not os.path.exists(args.sam2_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam2_checkpoint}")
        return
    
    if not os.path.exists(args.gt_annotations):
        print(f"Error: Ground truth annotations directory not found: {args.gt_annotations}")
        return
    
    if not os.path.exists(args.test_images_dir):
        print(f"Error: Test images directory not found: {args.test_images_dir}")
        return
    
    print(f"\nYOLO+SAM2 Tiled Pipeline Evaluation Configuration:")
    print(f"  YOLO labels: {args.yolo_labels_dir}")
    print(f"  SAM2 checkpoint: {args.sam2_checkpoint}")
    print(f"  SAM2 model type: {args.sam2_model_type}")
    print(f"  Ground truth: {args.gt_annotations}")
    print(f"  Test images: {args.test_images_dir}")
    print(f"  Mask threshold: {args.mask_threshold}")
    print(f"  Post-processing: fill_holes={args.fill_holes}, morphology={args.morphology}")
    print(f"  Box preprocessing: iou_merge={args.iou_merge}, topk={args.topk_boxes}")
    print(f"  Tile optimizations: min_w_px={args.min_w_px}, pad_px={args.pad_px}")
    
    # Load SAM2 model
    print(f"\nLoading SAM2 model...")
    sam, predictor, device = load_sam2_model(args.sam2_checkpoint, args.sam2_model_type)
    
    # Load YOLO predictions
    print(f"\nLoading YOLO predictions...")
    yolo_predictions = load_yolo_predictions(args.yolo_labels_dir, args.conf_threshold)
    
    # Load ground truth
    print(f"\nLoading ground truth annotations...")
    gt_annotations = load_ground_truth_annotations(args.gt_annotations)
    
    # Initialize metrics calculator
    metrics_calculator = SegmentationMetrics()
    
    # Process images in batches
    print(f"\nProcessing images in batches of {args.batch_size}...")
    all_image_names = list(yolo_predictions.keys())
    total_images = len(all_image_names)
    
    all_predictions = []
    all_targets = []
    all_metrics = []
    image_names = []
    
    for batch_start in tqdm(range(0, total_images, args.batch_size), desc="Processing batches"):
        batch_end = min(batch_start + args.batch_size, total_images)
        batch_images = all_image_names[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//args.batch_size + 1}: images {batch_start+1}-{batch_end} of {total_images}")
        
        batch_predictions, batch_targets, batch_metrics, batch_image_names = process_batch(
            batch_images, predictor, yolo_predictions, gt_annotations, args, metrics_calculator, args.test_images_dir
        )
        
        all_predictions.extend(batch_predictions)
        all_targets.extend(batch_targets)
        all_metrics.extend(batch_metrics)
        image_names.extend(batch_image_names)
        
        cleanup_memory()
        
        del batch_predictions, batch_targets, batch_metrics, batch_image_names
    
    if not all_metrics:
        print("No valid predictions to evaluate")
        return
    
    # Calculate average metrics
    print(f"\nCalculating average metrics...")
    avg_metrics_fixed = {}
    for key in all_metrics[0].keys():
        avg_metrics_fixed[key] = np.mean([m[key] for m in all_metrics])
    
    # Use pre-calculated threshold or fixed threshold
    if args.opt_threshold is not None:
        opt_threshold_f1 = args.opt_threshold
        print(f"Using pre-calculated optimal threshold: {opt_threshold_f1:.3f}")
    else:
        opt_threshold_f1 = 0.5
        print(f"Using fixed threshold: {opt_threshold_f1:.3f}")
    
    # Calculate metrics with optimal threshold
    if opt_threshold_f1 == args.mask_threshold:
        avg_metrics_opt = avg_metrics_fixed.copy()
    else:
        print("Recalculating metrics with different threshold...")
        all_metrics_opt = []
        
        chunk_size = 100
        for i in range(0, len(all_predictions), chunk_size):
            chunk_preds = all_predictions[i:i+chunk_size]
            chunk_targets = all_targets[i:i+chunk_size]
            
            for pred, target in zip(chunk_preds, chunk_targets):
                pred_binary = (pred > opt_threshold_f1).astype(np.uint8)
                metrics = evaluate_masks(pred_binary, target, metrics_calculator, 0.5)
                if metrics:
                    all_metrics_opt.append(metrics)
            
            del chunk_preds, chunk_targets
            cleanup_memory()
        
        avg_metrics_opt = {}
        for key in all_metrics_opt[0].keys():
            avg_metrics_opt[key] = np.mean([m[key] for m in all_metrics_opt])
    
    # Calculate extra metrics
    print("Calculating extra metrics...")
    iou_50_scores = []
    iou_75_scores = []
    iou_90_scores = []
    auprc_scores = []
    
    chunk_size = 100
    for i in range(0, len(all_predictions), chunk_size):
        chunk_preds = all_predictions[i:i+chunk_size]
        chunk_targets = all_targets[i:i+chunk_size]
        
        for pred, target in zip(chunk_preds, chunk_targets):
            iou_50_scores.append(calculate_iou_at_threshold(pred, target, 0.5))
            iou_75_scores.append(calculate_iou_at_threshold(pred, target, 0.75))
            iou_90_scores.append(calculate_iou_at_threshold(pred, target, 0.90))
            
            try:
                auprc = average_precision_score(target.flatten(), pred.flatten())
                auprc_scores.append(auprc)
            except:
                auprc_scores.append(0.0)
        
        del chunk_preds, chunk_targets
        cleanup_memory()
    
    extra_metrics = {
        'iou_50': np.mean(iou_50_scores),
        'iou_75': np.mean(iou_75_scores),
        'iou_90': np.mean(iou_90_scores),
        'auprc': np.mean(auprc_scores)
    }
    
    avg_metrics_opt.update(extra_metrics)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    exp_id = f"eval_yolo_sam2_tiled_{timestamp}"
    
    csv_path = save_evaluation_results(
        exp_id=exp_id,
        model="yolo_sam2_tiled",
        subset_size="tiled_test",
        variant="pipeline",
        prompt_type="box_prompts",
        img_size=800,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        test_metrics_opt=avg_metrics_opt,
        opt_threshold=opt_threshold_f1,
        predictions_dir=args.output_dir,
        save_dir=args.output_dir,
        timestamp=timestamp,
        all_predictions=all_predictions,
        all_targets=all_targets,
        all_metrics=all_metrics,
        image_names=image_names,
        fill_holes=args.fill_holes,
        morphology=args.morphology,
        iou_merge=args.iou_merge,
        topk_boxes=args.topk_boxes,
        min_w_px=args.min_w_px,
        pad_px=args.pad_px
    )
    
    # Print final results
    print(f"\nTiled YOLO+SAM2 Pipeline Evaluation Results:")
    print(f"Results saved to: {csv_path}")
    
    print(f"\nOptimal threshold ({opt_threshold_f1:.3f}) results:")
    print(f"  IoU: {avg_metrics_opt.get('iou', 0.0):.4f}")
    print(f"  Dice: {avg_metrics_opt.get('dice', 0.0):.4f}")
    print(f"  Precision: {avg_metrics_opt.get('precision', 0.0):.4f}")
    print(f"  Recall: {avg_metrics_opt.get('recall', 0.0):.4f}")
    print(f"  F1: {avg_metrics_opt.get('f1', 0.0):.4f}")
    print(f"  IoU@50: {avg_metrics_opt.get('iou_50', 0.0):.4f}")
    print(f"  IoU@75: {avg_metrics_opt.get('iou_75', 0.0):.4f}")
    print(f"  IoU@90: {avg_metrics_opt.get('iou_90', 0.0):.4f}")
    print(f"  AUPRC: {avg_metrics_opt.get('auprc', 0.0):.4f}")
    
    print(f"\nTiled pipeline evaluation completed!")
    print(f"YOLO bboxes (tiles) → SAM2 box-prompts → mask merging → evaluation")

if __name__ == "__main__":
    main()




