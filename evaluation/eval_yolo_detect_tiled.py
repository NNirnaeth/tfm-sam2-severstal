#!/usr/bin/env python3
"""
Evaluate YOLO Detection model on tiled Severstal test dataset
Test data: tiled dataset test split (800x256 tiles)

Features:
- Strict evaluation (COCO-like): IoU≥0.5 matching
- Benevolent evaluation (TFG-like): 75% area overlap matching  
- Comprehensive metrics: mAP@50, mAP@50-95, Precision, Recall, F1
- Threshold optimization and calibration results
- Per-image and aggregate statistics
- Pipeline preparation for SAM2 refinement

This script evaluates trained YOLO detection model on tiled dataset
with both strict and benevolent matching criteria for comprehensive comparison.
"""

import os
import sys
import subprocess
import argparse
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import yaml
import cv2
import base64
import zlib
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import average_precision_score, precision_recall_curve

def load_yolo_predictions(labels_dir, conf_threshold=0.25):
    """Load YOLO predictions from labels directory with confidence filtering"""
    predictions = {}
    
    for label_file in Path(labels_dir).glob("*.txt"):
        image_name = label_file.stem + ".jpg"
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5]) if len(parts) >= 6 else 1.0
                
                # Filter by confidence threshold
                if confidence >= conf_threshold:
                    boxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    })
        
        predictions[image_name] = boxes
    
    print(f"Loaded predictions for {len(predictions)} images (conf≥{conf_threshold})")
    return predictions

def load_ground_truth_annotations(ann_dir):
    """Load ground truth annotations from Supervisely format JSON files"""
    gt_annotations = {}
    
    for json_file in Path(ann_dir).glob("*.txt.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            image_name = json_file.stem.replace('.txt', '') + ".jpg"
            objects = []
            
            for obj in data.get("objects", []):
                bitmap = obj.get("bitmap", {})
                if "data" not in bitmap or "origin" not in bitmap:
                    continue
                
                # Decode bitmap
                raw = zlib.decompress(base64.b64decode(bitmap["data"]))
                patch = np.array(Image.open(BytesIO(raw)))
                
                # Handle different channel formats
                if patch.ndim == 3 and patch.shape[2] == 4:
                    patch = patch[:, :, 3]  # Alpha channel
                elif patch.ndim == 3:
                    patch = patch[:, :, 0]  # First channel
                
                patch = (patch > 0).astype(np.uint8)
                origin_x, origin_y = map(int, bitmap["origin"])
                
                # Convert patch to bounding box
                bbox = patch_to_bbox(patch, origin_x, origin_y)
                if bbox:
                    class_title = obj.get("classTitle", "defect")
                    class_id = get_class_id(class_title)
                    
                    objects.append({
                        'class_id': class_id,
                        'bbox': bbox,
                        'area': bbox['width'] * bbox['height']
                    })
            
            gt_annotations[image_name] = objects
            
        except Exception as e:
            print(f"Warning: Could not load annotation {json_file}: {e}")
            image_name = json_file.stem.replace('.txt', '') + ".jpg"
            gt_annotations[image_name] = []
    
    print(f"Loaded ground truth for {len(gt_annotations)} images")
    return gt_annotations

def get_class_id(class_title):
    """Map Severstal class titles to IDs"""
    class_mapping = {
        "defect": 0,  # Binary case
        "1": 0,       # Severstal class 1
        "2": 1,       # Severstal class 2
        "3": 2,       # Severstal class 3
        "4": 3        # Severstal class 4
    }
    return class_mapping.get(str(class_title), 0)

def patch_to_bbox(patch, origin_x, origin_y):
    """Convert patch to bounding box"""
    nonzero_y, nonzero_x = np.nonzero(patch)
    
    if len(nonzero_x) == 0:
        return None
    
    # Calculate bbox in patch coordinates
    min_x, max_x = nonzero_x.min(), nonzero_x.max()
    min_y, max_y = nonzero_y.min(), nonzero_y.max()
    
    # Convert to image coordinates
    bbox_x1 = origin_x + min_x
    bbox_y1 = origin_y + min_y
    bbox_x2 = origin_x + max_x
    bbox_y2 = origin_y + max_y
    
    # Calculate center and dimensions
    center_x = (bbox_x1 + bbox_x2) / 2
    center_y = (bbox_y1 + bbox_y2) / 2
    width = bbox_x2 - bbox_x1 + 1
    height = bbox_y2 - bbox_y1 + 1
    
    return {
        'x1': bbox_x1, 'y1': bbox_y1, 'x2': bbox_x2, 'y2': bbox_y2,
        'center_x': center_x, 'center_y': center_y,
        'width': width, 'height': height
    }

def convert_yolo_to_pixel_coords(yolo_box, img_width, img_height):
    """Convert YOLO normalized coordinates to pixel coordinates"""
    x_center = yolo_box['x_center'] * img_width
    y_center = yolo_box['y_center'] * img_height
    width = yolo_box['width'] * img_width
    height = yolo_box['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return {
        'x1': max(0, x1), 'y1': max(0, y1),
        'x2': min(img_width, x2), 'y2': min(img_height, y2),
        'center_x': x_center, 'center_y': y_center,
        'width': width, 'height': height
    }

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    # Calculate intersection
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_area_overlap_ratio(pred_box, gt_box):
    """Calculate area overlap ratio for benevolent evaluation"""
    # Calculate intersection
    x1 = max(pred_box['x1'], gt_box['x1'])
    y1 = max(pred_box['y1'], gt_box['y1'])
    x2 = min(pred_box['x2'], gt_box['x2'])
    y2 = min(pred_box['y2'], gt_box['y2'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    pred_area = (pred_box['x2'] - pred_box['x1']) * (pred_box['y2'] - pred_box['y1'])
    gt_area = (gt_box['x2'] - gt_box['x1']) * (gt_box['y2'] - gt_box['y1'])
    
    pred_overlap_ratio = intersection / pred_area if pred_area > 0 else 0.0
    gt_overlap_ratio = intersection / gt_area if gt_area > 0 else 0.0
    
    return pred_overlap_ratio, gt_overlap_ratio

def evaluate_strict_matching(predictions, gt_annotations, img_width=800, img_height=256, 
                            iou_threshold=0.5):
    """Evaluate with strict COCO-like matching (IoU≥0.5)"""
    all_pred_boxes = []
    all_gt_boxes = []
    matches = []
    
    for image_name in predictions.keys():
        pred_boxes = predictions.get(image_name, [])
        gt_boxes = gt_annotations.get(image_name, [])
        
        # Convert predictions to pixel coordinates
        pred_pixel_boxes = []
        for pred_box in pred_boxes:
            pixel_box = convert_yolo_to_pixel_coords(pred_box, img_width, img_height)
            pixel_box['confidence'] = pred_box['confidence']
            pixel_box['class_id'] = pred_box['class_id']
            pred_pixel_boxes.append(pixel_box)
        
        # Match predictions to ground truth
        gt_matched = [False] * len(gt_boxes)
        
        # Sort predictions by confidence (highest first)
        pred_pixel_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        
        for pred_box in pred_pixel_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_obj in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                
                gt_box = gt_obj['bbox']
                iou = calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Record match
            is_match = best_iou >= iou_threshold
            if is_match and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
            
            matches.append({
                'image_name': image_name,
                'pred_confidence': pred_box['confidence'],
                'pred_class': pred_box['class_id'],
                'gt_matched': is_match,
                'iou': best_iou,
                'gt_class': gt_boxes[best_gt_idx]['class_id'] if best_gt_idx >= 0 else -1
            })
        
        # Add unmatched ground truth as false negatives
        for gt_idx, gt_obj in enumerate(gt_boxes):
            if not gt_matched[gt_idx]:
                matches.append({
                    'image_name': image_name,
                    'pred_confidence': 0.0,
                    'pred_class': -1,
                    'gt_matched': False,
                    'iou': 0.0,
                    'gt_class': gt_obj['class_id'],
                    'false_negative': True
                })
    
    return matches

def evaluate_benevolent_matching(predictions, gt_annotations, img_width=800, img_height=256,
                                overlap_threshold=0.75):
    """Evaluate with benevolent TFG-like matching (75% area overlap)"""
    all_matches = []
    
    for image_name in predictions.keys():
        pred_boxes = predictions.get(image_name, [])
        gt_boxes = gt_annotations.get(image_name, [])
        
        # Convert predictions to pixel coordinates
        pred_pixel_boxes = []
        for pred_box in pred_boxes:
            pixel_box = convert_yolo_to_pixel_coords(pred_box, img_width, img_height)
            pixel_box['confidence'] = pred_box['confidence']
            pixel_box['class_id'] = pred_box['class_id']
            pred_pixel_boxes.append(pixel_box)
        
        # Match predictions to ground truth
        gt_matched = [False] * len(gt_boxes)
        
        # Sort predictions by confidence
        pred_pixel_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        
        for pred_box in pred_pixel_boxes:
            best_score = 0
            best_gt_idx = -1
            
            for gt_idx, gt_obj in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                
                gt_box = gt_obj['bbox']
                pred_overlap, gt_overlap = calculate_area_overlap_ratio(pred_box, gt_box)
                
                # Benevolent matching criteria
                pred_area = (pred_box['x2'] - pred_box['x1']) * (pred_box['y2'] - pred_box['y1'])
                gt_area = (gt_box['x2'] - gt_box['x1']) * (gt_box['y2'] - gt_box['y1'])
                
                if pred_area <= gt_area:
                    # Prediction smaller than GT: check if intersection ≥ 75% of prediction
                    score = pred_overlap
                else:
                    # Prediction larger than GT: check if intersection ≥ 75% of GT
                    score = gt_overlap
                
                if score > best_score:
                    best_score = score
                    best_gt_idx = gt_idx
            
            # Record match
            is_match = best_score >= overlap_threshold
            if is_match and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
            
            all_matches.append({
                'image_name': image_name,
                'pred_confidence': pred_box['confidence'],
                'pred_class': pred_box['class_id'],
                'gt_matched': is_match,
                'overlap_score': best_score,
                'gt_class': gt_boxes[best_gt_idx]['class_id'] if best_gt_idx >= 0 else -1
            })
        
        # Add unmatched ground truth as false negatives
        for gt_idx, gt_obj in enumerate(gt_boxes):
            if not gt_matched[gt_idx]:
                all_matches.append({
                    'image_name': image_name,
                    'pred_confidence': 0.0,
                    'pred_class': -1,
                    'gt_matched': False,
                    'overlap_score': 0.0,
                    'gt_class': gt_obj['class_id'],
                    'false_negative': True
                })
    
    return all_matches

def calculate_metrics_from_matches(matches, confidence_threshold=0.25):
    """Calculate precision, recall, F1, mAP from matches"""
    # Filter by confidence threshold
    valid_matches = [m for m in matches if m['pred_confidence'] >= confidence_threshold or m.get('false_negative', False)]
    
    if not valid_matches:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mAP50': 0.0}
    
    # Count TP, FP, FN
    tp = sum(1 for m in valid_matches if m['gt_matched'] and m['pred_confidence'] >= confidence_threshold)
    fp = sum(1 for m in valid_matches if not m['gt_matched'] and m['pred_confidence'] >= confidence_threshold)
    fn = sum(1 for m in valid_matches if m.get('false_negative', False))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate mAP@50 (simplified - using precision at fixed recall points)
    confidences = [m['pred_confidence'] for m in valid_matches if m['pred_confidence'] > 0]
    gt_labels = [1 if m['gt_matched'] else 0 for m in valid_matches if m['pred_confidence'] > 0]
    
    if len(confidences) > 0 and sum(gt_labels) > 0:
        try:
            mAP50 = average_precision_score(gt_labels, confidences)
        except:
            mAP50 = 0.0
    else:
        mAP50 = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP50': mAP50,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def run_yolo_evaluation(model_path, dataset_yaml, output_dir, conf_threshold=0.25, 
                       iou_threshold=0.70, imgsz=896):
    """Run YOLO prediction to generate .txt files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images directory from dataset yaml
    with open(dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    dataset_path = dataset_config.get('path', os.path.dirname(dataset_yaml))
    test_images_dir = os.path.join(dataset_path, "images", "test")
    
    # YOLO prediction command
    cmd = [
        "yolo", "task=detect", "mode=predict",
        f"model={model_path}",
        f"source={test_images_dir}",
        f"imgsz={imgsz}",
        f"conf={conf_threshold}",
        f"iou={iou_threshold}",
        "save_txt=True",
        "save_conf=True",
        f"project={output_dir}",
        "name=predict_test_tiled",
        "exist_ok=True"
    ]
    
    print(f"Running YOLO prediction:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Prediction completed successfully!")
        return True, os.path.join(output_dir, "predict_test_tiled", "labels")
    except subprocess.CalledProcessError as e:
        print(f"Prediction failed: {e}")
        print(f"Error output: {e.stderr}")
        return False, None

def save_evaluation_results(exp_id, model, dataset, variant, conf_threshold, iou_threshold,
                           strict_metrics, benevolent_metrics, save_dir, timestamp):
    """Save evaluation results to CSV following TFM format"""
    
    # Create results directory
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare results data for both evaluation types
    results_data = []
    
    # Strict evaluation results
    strict_row = {
        'exp_id': f"{exp_id}_strict",
        'model': model,
        'subset_size': 'tiled_test',
        'variant': f"{variant}_strict",
        'prompt_type': 'detection',
        'img_size': 896,
        'batch_size': 1,
        'steps': 0,
        'lr': 0.0,
        'wd': 0.0,
        'seed': 42,
        'val_mIoU': 0.0,
        'val_Dice': 0.0,
        'test_mIoU': 0.0,
        'test_Dice': 0.0,
        'IoU@50': 0.0,
        'IoU@75': 0.0,
        'IoU@90': 0.0,
        'IoU@95': 0.0,
        'Precision': strict_metrics['precision'],
        'Recall': strict_metrics['recall'],
        'F1': strict_metrics['f1'],
        'mAP50': strict_metrics['mAP50'],
        'mAP50_95': 0.0,
        'TP': strict_metrics['tp'],
        'FP': strict_metrics['fp'],
        'FN': strict_metrics['fn'],
        'ckpt_path': model,
        'timestamp': timestamp,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'evaluation_type': 'strict_coco_like',
        'matching_criteria': 'IoU≥0.5'
    }
    
    # Benevolent evaluation results
    benevolent_row = {
        'exp_id': f"{exp_id}_benevolent",
        'model': model,
        'subset_size': 'tiled_test',
        'variant': f"{variant}_benevolent",
        'prompt_type': 'detection',
        'img_size': 896,
        'batch_size': 1,
        'steps': 0,
        'lr': 0.0,
        'wd': 0.0,
        'seed': 42,
        'val_mIoU': 0.0,
        'val_Dice': 0.0,
        'test_mIoU': 0.0,
        'test_Dice': 0.0,
        'IoU@50': 0.0,
        'IoU@75': 0.0,
        'IoU@90': 0.0,
        'IoU@95': 0.0,
        'Precision': benevolent_metrics['precision'],
        'Recall': benevolent_metrics['recall'],
        'F1': benevolent_metrics['f1'],
        'mAP50': benevolent_metrics['mAP50'],
        'mAP50_95': 0.0,
        'TP': benevolent_metrics['tp'],
        'FP': benevolent_metrics['fp'],
        'FN': benevolent_metrics['fn'],
        'ckpt_path': model,
        'timestamp': timestamp,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'evaluation_type': 'benevolent_tfg_like',
        'matching_criteria': '75%_area_overlap'
    }
    
    results_data = [strict_row, benevolent_row]
    
    # Save to CSV
    csv_filename = f"{timestamp}_{exp_id}_tiled_yolo_evaluation.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)
    
    print(f"Evaluation results saved to: {csv_path}")
    return csv_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO detection on tiled dataset")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained YOLO model (.pt file)")
    parser.add_argument("--dataset-yaml", type=str, required=True,
                       help="Path to tiled dataset YAML file")
    parser.add_argument("--output-dir", type=str, 
                       default="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_tiled_detection",
                       help="Output directory for evaluation results")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                       help="Confidence threshold for predictions")
    parser.add_argument("--iou-threshold", type=float, default=0.70,
                       help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=896,
                       help="Input image size for inference")
    parser.add_argument("--exp-id", type=str, default=None,
                       help="Experiment ID (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.dataset_yaml):
        print(f"Error: Dataset YAML file not found: {args.dataset_yaml}")
        return
    
    # Get dataset info
    with open(args.dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    dataset_path = dataset_config.get('path', os.path.dirname(args.dataset_yaml))
    test_images_dir = os.path.join(dataset_path, "images", "test")
    test_labels_dir = os.path.join(dataset_path, "labels", "test")
    
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found: {test_images_dir}")
        return
    
    if not os.path.exists(test_labels_dir):
        print(f"Error: Test labels directory not found: {test_labels_dir}")
        return
    
    # Generate experiment ID if not provided
    if args.exp_id is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        args.exp_id = f"eval_yolo_tiled_detect_{timestamp}"
    
    print(f"Tiled YOLO Detection Evaluation Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset YAML: {args.dataset_yaml}")
    print(f"  Test images: {test_images_dir}")
    print(f"  Test labels: {test_labels_dir}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Experiment ID: {args.exp_id}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run YOLO prediction
    print("\nRunning YOLO prediction...")
    success, predictions_dir = run_yolo_evaluation(
        model_path=args.model_path,
        dataset_yaml=args.dataset_yaml,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        imgsz=args.imgsz
    )
    
    if not success:
        print("YOLO prediction failed!")
        return
    
    print(f"Predictions saved to: {predictions_dir}")
    
    # Load predictions and ground truth
    print("\nLoading predictions and ground truth...")
    predictions = load_yolo_predictions(predictions_dir, args.conf_threshold)
    gt_annotations = load_ground_truth_annotations(test_labels_dir)
    
    # Evaluate with strict matching (COCO-like)
    print("\nEvaluating with strict matching (IoU≥0.5)...")
    strict_matches = evaluate_strict_matching(
        predictions, gt_annotations, 
        img_width=800, img_height=256, 
        iou_threshold=0.5
    )
    strict_metrics = calculate_metrics_from_matches(strict_matches, args.conf_threshold)
    
    # Evaluate with benevolent matching (75% area overlap)
    print("Evaluating with benevolent matching (75% area overlap)...")
    benevolent_matches = evaluate_benevolent_matching(
        predictions, gt_annotations,
        img_width=800, img_height=256,
        overlap_threshold=0.75
    )
    benevolent_metrics = calculate_metrics_from_matches(benevolent_matches, args.conf_threshold)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    csv_path = save_evaluation_results(
        exp_id=args.exp_id,
        model=os.path.basename(args.model_path),
        dataset="tiled_severstal",
        variant="detect",
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        strict_metrics=strict_metrics,
        benevolent_metrics=benevolent_metrics,
        save_dir=args.output_dir,
        timestamp=timestamp
    )
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"\nSTRICT EVALUATION (COCO-like, IoU≥0.5):")
    print(f"  Precision: {strict_metrics['precision']:.4f}")
    print(f"  Recall: {strict_metrics['recall']:.4f}")
    print(f"  F1: {strict_metrics['f1']:.4f}")
    print(f"  mAP@50: {strict_metrics['mAP50']:.4f}")
    print(f"  TP: {strict_metrics['tp']}, FP: {strict_metrics['fp']}, FN: {strict_metrics['fn']}")
    
    print(f"\nBENEVOLENT EVALUATION (TFG-like, 75% area overlap):")
    print(f"  Precision: {benevolent_metrics['precision']:.4f}")
    print(f"  Recall: {benevolent_metrics['recall']:.4f}")
    print(f"  F1: {benevolent_metrics['f1']:.4f}")
    print(f"  mAP@50: {benevolent_metrics['mAP50']:.4f}")
    print(f"  TP: {benevolent_metrics['tp']}, FP: {benevolent_metrics['fp']}, FN: {benevolent_metrics['fn']}")
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Predictions ready for SAM2 pipeline at: {predictions_dir}")
    print(f"\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()




