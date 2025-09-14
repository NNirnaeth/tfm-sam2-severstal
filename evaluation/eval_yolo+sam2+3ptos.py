#!/usr/bin/env python3
"""
Evaluate YOLO+SAM2 pipeline with GT point prompts on Severstal test split

This script combines:
1. YOLO bounding box predictions (from trained model or .txt files)
2. 1-3 GT points as additional prompts (simulating human clicks)
3. SAM2-Large fine-tuned checkpoint for segmentation refinement

For each image:
- Load YOLO bboxes (normalized xywh → xyxy pixels)
- Generate 1-3 points from GT mask:
  * p1+: centroid of main GT mask
  * p2+ (optional): pixel with max distance to border
  * p3- (optional): negative point 5-10px outside border
- Call SAM2 with box + points prompts
- Handle fallback: if YOLO fails, run points-only variant
- Evaluate with standard metrics and save results

Requirements:
- YOLO detection weights/predictions
- SAM2-Large FT checkpoint (Best)
- Test images and GT (Supervisely JSON with bitmap.data + origin)
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
from sklearn.metrics import average_precision_score
from scipy.ndimage import distance_transform_edt

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append('/home/ptp/sam2/libs/sam2base')

# Import SAM2 and utilities
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.metrics import SegmentationMetrics
from ultralytics import YOLO


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


def load_yolo_model_or_predictions(yolo_path):
    """Load YOLO model or predictions from .txt files"""
    if yolo_path.endswith('.pt') or yolo_path.endswith('.pth'):
        # Load YOLO model
        print(f"Loading YOLO model from: {yolo_path}")
        model = YOLO(yolo_path)
        return model, None
    else:
        # Load predictions from directory of .txt files
        print(f"Loading YOLO predictions from: {yolo_path}")
        predictions = load_yolo_txt_predictions(yolo_path)
        return None, predictions


def load_yolo_txt_predictions(labels_dir):
    """Load YOLO bounding box predictions from labels directory"""
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
                
                bboxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'conf': confidence
                })
        
        predictions[image_name] = bboxes
    
    print(f"Loaded predictions for {len(predictions)} images")
    return predictions


def run_yolo_detection(yolo_model, image, conf_threshold=0.15, iou_threshold=0.7):
    """Run YOLO detection on image and return normalized bboxes"""
    results = yolo_model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    
    bboxes = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get normalized coordinates
                x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                bboxes.append({
                    'class_id': int(cls),
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height),
                    'conf': float(conf)
                })
    
    return bboxes


def load_ground_truth_annotations(ann_dir):
    """Load ground truth annotations from Supervisely format JSON files"""
    gt = {}
    # Handle both *.jpg.json and *.json patterns
    json_paths = list(Path(ann_dir).glob("*.jpg.json")) + list(Path(ann_dir).glob("*.json"))
    
    for jf in json_paths:
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
            
            # Extract image name: remove .json extension, keep .jpg
            image_name = jf.name.replace(".json", "")
            gt[image_name] = objs
        except Exception as e:
            print(f"Warning: Could not load {jf}: {e}")
            image_name = jf.name.replace(".json", "")
            gt[image_name] = []
    
    print(f"Loaded ground truth for {len(gt)} images")
    return gt


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


def generate_gt_points(gt_mask, num_points=3):
    """
    Generate 1-3 points from ground truth mask:
    - p1+: centroid of main GT mask
    - p2+ (optional): pixel with max distance to border  
    - p3- (optional): negative point 5-10px outside border
    """
    if np.sum(gt_mask) == 0:
        return None, None
    
    points = []
    point_labels = []
    
    # Find all connected components
    from scipy.ndimage import label as scipy_label
    labeled_mask, num_labels = scipy_label(gt_mask)
    
    if num_labels == 0:
        return None, None
    
    # Find largest component (main defect)
    component_sizes = []
    for i in range(1, num_labels + 1):
        component_sizes.append(np.sum(labeled_mask == i))
    
    main_component = np.argmax(component_sizes) + 1
    main_mask = (labeled_mask == main_component).astype(np.uint8)
    
    # Point 1: Centroid of main component
    coords = np.where(main_mask > 0)
    if len(coords[0]) > 0:
        centroid_y = int(np.mean(coords[0]))
        centroid_x = int(np.mean(coords[1]))
        points.append([centroid_x, centroid_y])
        point_labels.append(1)
    
    # Point 2: Pixel with maximum distance to border (if requested)
    if num_points >= 2 and len(coords[0]) > 0:
        # Calculate distance transform
        dist_transform = distance_transform_edt(main_mask)
        max_dist_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        max_dist_y, max_dist_x = max_dist_idx
        
        # Ensure it's different from centroid
        if abs(max_dist_x - centroid_x) > 3 or abs(max_dist_y - centroid_y) > 3:
            points.append([max_dist_x, max_dist_y])
            point_labels.append(1)
    
    # Point 3: Negative point outside border (if requested)
    if num_points >= 3 and len(coords[0]) > 0:
        # Find border pixels
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(main_mask, kernel, iterations=1)
        border = dilated - main_mask
        
        # Dilate border to get area outside defect
        outer_border = cv2.dilate(border, kernel, iterations=random.randint(5, 10))
        
        # Find valid negative points
        negative_candidates = np.where((outer_border > 0) & (gt_mask == 0))
        if len(negative_candidates[0]) > 0:
            # Randomly select one
            idx = random.randint(0, len(negative_candidates[0]) - 1)
            neg_y, neg_x = negative_candidates[0][idx], negative_candidates[1][idx]
            points.append([neg_x, neg_y])
            point_labels.append(0)
    
    if not points:
        return None, None
    
    return np.array(points), np.array(point_labels)


def convert_bbox_to_sam2_format(bbox, img_width, img_height):
    """Convert YOLO bbox to SAM2 input format [x1, y1, x2, y2] with explicit dtype"""
    x_center = bbox['x_center'] * img_width
    y_center = bbox['y_center'] * img_height
    width = bbox['width'] * img_width
    height = bbox['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def generate_sam2_masks_with_box_and_points(predictor, image, bboxes, gt_points, gt_point_labels):
    """Generate SAM2 masks using both bounding boxes and point prompts"""
    predictor.set_image(image)
    
    # Ensure proper types for SAM2
    gt_points = gt_points.astype(np.float32)  # (K,2) in pixels
    gt_point_labels = gt_point_labels.astype(np.int32)
    
    all_masks = []
    all_scores = []
    
    for bbox in bboxes:
        # Convert bbox to SAM2 format with explicit dtype
        input_box = convert_bbox_to_sam2_format(bbox, image.shape[1], image.shape[0])
        
        # Combine box with GT points
        masks_pred, scores_pred, logits_pred = predictor.predict(
            box=input_box,
            point_coords=gt_points,
            point_labels=gt_point_labels,
            multimask_output=True
        )
        
        # Select mask with highest score
        best_idx = np.argmax(scores_pred)
        best_mask = masks_pred[best_idx]
        best_score = scores_pred[best_idx]
        
        all_masks.append(best_mask)
        all_scores.append(best_score)
    
    return all_masks, all_scores


def generate_sam2_masks_points_only(predictor, image, gt_points, gt_point_labels):
    """Generate SAM2 masks using only point prompts (fallback when YOLO fails)"""
    predictor.set_image(image)
    
    # Ensure proper types for SAM2
    gt_points = gt_points.astype(np.float32)  # (K,2) in pixels
    gt_point_labels = gt_point_labels.astype(np.int32)
    
    masks_pred, scores_pred, logits_pred = predictor.predict(
        point_coords=gt_points,
        point_labels=gt_point_labels,
        multimask_output=True
    )
    
    # Select mask with highest score
    best_idx = np.argmax(scores_pred)
    best_mask = masks_pred[best_idx]
    best_score = scores_pred[best_idx]
    
    return [best_mask], [best_score]


def merge_masks_per_image(masks, scores, img_height, img_width):
    """Merge multiple masks per image, keeping maximum per pixel"""
    if not masks:
        return np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Create merged mask (maximum per pixel)
    merged_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for mask in masks:
        # Ensure mask has correct dimensions
        if mask.shape != (img_height, img_width):
            mask = cv2.resize(mask.astype(np.uint8), (img_width, img_height), 
                            interpolation=cv2.INTER_NEAREST)
        merged_mask = np.maximum(merged_mask, mask.astype(np.uint8))
    
    return merged_mask


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


def evaluate_masks(pred_mask, gt_mask, metrics_calculator):
    """Evaluate predicted masks against ground truth"""
    if pred_mask is None or gt_mask is None:
        return None
    
    try:
        metrics = metrics_calculator.compute_pixel_metrics(
            pred_mask.flatten(),
            gt_mask.flatten()
        )
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None


def process_single_image(image_name, image_path, yolo_model, yolo_predictions, 
                        gt_annotations, predictor, args, metrics_calculator):
    """Process a single image and return results"""
    # Load image
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]
    
    # Get ground truth for this image
    gt_objs = gt_annotations.get(image_name, [])
    gt_merged = build_gt_union(gt_objs, img_height, img_width)
    
    # Generate GT points for prompting (1-3 points)
    gt_points, gt_point_labels = generate_gt_points(gt_merged, num_points=args.num_gt_points)
    
    if gt_points is None:
        # No GT defect → empty mask and trivial metrics
        results = {
            'image_name': image_name,
            'method': 'no_gt',
            'num_gt_points': 0,
            'num_yolo_boxes': 0,
            'used_fallback': False,
            'iou': 1.0,  # Perfect IoU for empty prediction vs empty GT
            'dice': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0
        }
        return results
    
    # Get YOLO bboxes for this image
    if yolo_model is not None:
        # Run YOLO detection
        bboxes = run_yolo_detection(yolo_model, image, 
                                   conf_threshold=args.conf_threshold,
                                   iou_threshold=args.iou_threshold)
    else:
        # Use pre-computed predictions
        bboxes = yolo_predictions.get(image_name, [])
    
    # Initialize results
    results = {
        'image_name': image_name,
        'num_gt_points': len(gt_points),
        'num_yolo_boxes': len(bboxes),
        'used_fallback': False
    }
    
    # Filter bboxes by confidence and limit to top-K (avoid noise when giving points)
    bboxes = [b for b in bboxes if b['conf'] >= args.conf_threshold]
    bboxes = sorted(bboxes, key=lambda x: x['conf'], reverse=True)[:20]  # Top 20 boxes
    
    # Case 1: YOLO detected boxes + GT points
    if bboxes:
        # Convert bboxes to SAM2 format
        sam2_bboxes = [convert_bbox_to_sam2_format(bbox, img_width, img_height) 
                      for bbox in bboxes]
        
        if sam2_bboxes:
            # Generate masks with box + points prompts
            masks, scores = generate_sam2_masks_with_box_and_points(
                predictor, image, bboxes, gt_points, gt_point_labels)
            
            # Merge masks
            merged_mask = merge_masks_per_image(masks, scores, img_height, img_width)
            results['method'] = 'box_points'
        else:
            # No valid boxes after filtering
            masks, scores = generate_sam2_masks_points_only(
                predictor, image, gt_points, gt_point_labels)
            merged_mask = merge_masks_per_image(masks, scores, img_height, img_width)
            results['method'] = 'points_only'
            results['used_fallback'] = True
    else:
        # Case 2: No YOLO detection - use points-only fallback
        masks, scores = generate_sam2_masks_points_only(
            predictor, image, gt_points, gt_point_labels)
        merged_mask = merge_masks_per_image(masks, scores, img_height, img_width)
        results['method'] = 'points_only'
        results['used_fallback'] = True
    
    # Apply post-processing
    processed_mask = apply_post_processing(
        merged_mask,
        fill_holes_flag=args.fill_holes,
        morphology_flag=args.morphology
    )
    
    # Evaluate
    metrics = evaluate_masks(processed_mask, gt_merged, metrics_calculator)
    if metrics:
        results.update(metrics)
    
    return results


def save_results_to_csv(all_results, args, timestamp):
    """Save evaluation results to CSV files"""
    # Create results directory
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Aggregate results by method
    box_points_results = [r for r in all_results if r.get('method') == 'box_points']
    points_only_results = [r for r in all_results if r.get('method') == 'points_only']
    no_gt_results = [r for r in all_results if r.get('method') == 'no_gt']
    fallback_results = [r for r in all_results if r.get('used_fallback', False)]
    
    # Calculate average metrics
    def calc_avg_metrics(results):
        if not results:
            return {}
        metrics_keys = ['iou', 'dice', 'precision', 'recall', 'f1']
        avg_metrics = {}
        for key in metrics_keys:
            values = [r[key] for r in results if key in r]
            avg_metrics[key] = np.mean(values) if values else 0.0
        return avg_metrics
    
    box_points_avg = calc_avg_metrics(box_points_results)
    points_only_avg = calc_avg_metrics(points_only_results)
    overall_avg = calc_avg_metrics(all_results)
    
    # Save global summary
    exp_id = f"eval_yolo_sam2_{args.num_gt_points}ptos_{timestamp}"
    summary_data = {
        'exp_id': exp_id,
        'model': 'yolo_sam2_gt_points',
        'subset_size': 'test_split',
        'variant': f'{args.num_gt_points}_gt_points',
        'prompt_type': 'box_gt_points',
        'img_size': 1024,
        'num_gt_points': args.num_gt_points,
        'conf_threshold': args.conf_threshold,
        'iou_threshold': args.iou_threshold,
        'total_images': len(all_results),
        'box_points_images': len(box_points_results),
        'points_only_images': len(points_only_results),
        'no_gt_images': len(no_gt_results),
        'fallback_rate': len(fallback_results) / len(all_results) if all_results else 0.0,
        'overall_iou': overall_avg.get('iou', 0.0),
        'overall_dice': overall_avg.get('dice', 0.0),
        'overall_precision': overall_avg.get('precision', 0.0),
        'overall_recall': overall_avg.get('recall', 0.0),
        'overall_f1': overall_avg.get('f1', 0.0),
        'box_points_iou': box_points_avg.get('iou', 0.0),
        'box_points_dice': box_points_avg.get('dice', 0.0),
        'points_only_iou': points_only_avg.get('iou', 0.0),
        'points_only_dice': points_only_avg.get('dice', 0.0),
        'timestamp': timestamp,
        'sam2_checkpoint': args.sam2_checkpoint,
        'yolo_path': args.yolo_path
    }
    
    # Save summary CSV
    summary_csv = os.path.join(results_dir, f"{timestamp}_yolo_sam2_{args.num_gt_points}ptos_summary.csv")
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        writer.writeheader()
        writer.writerow(summary_data)
    
    # Save per-image CSV
    per_image_csv = os.path.join(results_dir, f"{timestamp}_yolo_sam2_{args.num_gt_points}ptos_per_image.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(per_image_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"Results saved:")
    print(f"  Summary: {summary_csv}")
    print(f"  Per-image: {per_image_csv}")
    
    return summary_csv, per_image_csv


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO+SAM2 with GT point prompts')
    parser.add_argument('--yolo_path', type=str, required=True,
                       help='Path to YOLO model (.pt) or predictions directory (.txt files)')
    parser.add_argument('--sam2_checkpoint', type=str, required=True,
                       help='Path to SAM2 fine-tuned checkpoint')
    parser.add_argument('--sam2_model_type', type=str, default='sam2_hiera_l',
                       choices=['sam2_hiera_l', 'sam2_hiera_b', 'sam2_hiera_t'],
                       help='SAM2 model type')
    parser.add_argument('--gt_annotations', type=str, required=True,
                       help='Path to ground truth annotations directory (Supervisely format)')
    parser.add_argument('--test_images_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, 
                       default='new_src/evaluation/evaluation_results/yolo_sam2_gt_points',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_gt_points', type=int, default=2, choices=[1, 2, 3],
                       help='Number of GT points to generate (1-3, default 2: center + max distance)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='YOLO confidence threshold (0.25 works better with points)')
    parser.add_argument('--iou_threshold', type=float, default=0.70,
                       help='YOLO IoU threshold for NMS')
    parser.add_argument('--fill_holes', action='store_true', default=False,
                       help='Apply hole filling post-processing')
    parser.add_argument('--morphology', action='store_true', default=False,
                       help='Apply morphological post-processing')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Number of images to process in each batch')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n YOLO+SAM2+{args.num_gt_points}Points Evaluation Configuration:")
    print(f"  YOLO: {args.yolo_path}")
    print(f"  SAM2: {args.sam2_checkpoint} ({args.sam2_model_type})")
    print(f"  GT annotations: {args.gt_annotations}")
    print(f"  Test images: {args.test_images_dir}")
    print(f"  GT points: {args.num_gt_points}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  Post-processing: fill_holes={args.fill_holes}, morphology={args.morphology}")
    
    # Load models and data
    print(f"\n Loading models and data...")
    
    # Load SAM2
    sam2, predictor, device = load_sam2_model(args.sam2_checkpoint, args.sam2_model_type)
    
    # Load YOLO
    yolo_model, yolo_predictions = load_yolo_model_or_predictions(args.yolo_path)
    
    # Load ground truth
    gt_annotations = load_ground_truth_annotations(args.gt_annotations)
    
    # Initialize metrics calculator
    metrics_calculator = SegmentationMetrics()
    
    # Get list of images to process
    if yolo_model is not None:
        # Use all images in test directory
        image_names = [f for f in os.listdir(args.test_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        # Use images that have YOLO predictions
        image_names = list(yolo_predictions.keys())
    
    # Filter to images that have GT annotations
    image_names = [name for name in image_names if name in gt_annotations]
    
    print(f" Processing {len(image_names)} images...")
    
    # Process images
    all_results = []
    
    for i, image_name in enumerate(tqdm(image_names, desc="Processing images")):
        image_path = os.path.join(args.test_images_dir, image_name)
        
        result = process_single_image(
            image_name, image_path, yolo_model, yolo_predictions,
            gt_annotations, predictor, args, metrics_calculator
        )
        
        if result is not None:
            all_results.append(result)
        
        # Clean up memory periodically
        if (i + 1) % args.batch_size == 0:
            cleanup_memory()
    
    if not all_results:
        print(" No valid results to evaluate")
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    summary_csv, per_image_csv = save_results_to_csv(all_results, args, timestamp)
    
    # Print summary
    box_points_results = [r for r in all_results if r.get('method') == 'box_points']
    points_only_results = [r for r in all_results if r.get('method') == 'points_only']
    no_gt_results = [r for r in all_results if r.get('method') == 'no_gt']
    fallback_results = [r for r in all_results if r.get('used_fallback', False)]
    
    def calc_avg(results, key):
        values = [r[key] for r in results if key in r]
        return np.mean(values) if values else 0.0
    
    print(f"\n EVALUATION RESULTS:")
    print(f"=" * 60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Box+Points method: {len(box_points_results)} images")
    print(f"Points-only method: {len(points_only_results)} images")
    print(f"No GT defect: {len(no_gt_results)} images")
    print(f"Fallback rate: {len(fallback_results)/len(all_results)*100:.1f}%")
    
    print(f"\n OVERALL PERFORMANCE:")
    print(f"  IoU:       {calc_avg(all_results, 'iou'):.4f}")
    print(f"  Dice:      {calc_avg(all_results, 'dice'):.4f}")
    print(f"  Precision: {calc_avg(all_results, 'precision'):.4f}")
    print(f"  Recall:    {calc_avg(all_results, 'recall'):.4f}")
    print(f"  F1:        {calc_avg(all_results, 'f1'):.4f}")
    
    if box_points_results:
        print(f"\n BOX+POINTS PERFORMANCE:")
        print(f"  IoU:       {calc_avg(box_points_results, 'iou'):.4f}")
        print(f"  Dice:      {calc_avg(box_points_results, 'dice'):.4f}")
        print(f"  F1:        {calc_avg(box_points_results, 'f1'):.4f}")
    
    if points_only_results:
        print(f"\n POINTS-ONLY PERFORMANCE:")
        print(f"  IoU:       {calc_avg(points_only_results, 'iou'):.4f}")
        print(f"  Dice:      {calc_avg(points_only_results, 'dice'):.4f}")
        print(f"  F1:        {calc_avg(points_only_results, 'f1'):.4f}")
    
    print(f"\n Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
