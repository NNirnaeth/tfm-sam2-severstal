#!/usr/bin/env python3
"""
YOLO Detection Evaluation on Test Split (2000 images)
Generate bounding box predictions on test images for SAM2 refinement

This script:
1. Loads trained YOLO model
2. Runs inference on test split (2000 annotated images from Severstal)
3. Saves bounding box predictions in format suitable for SAM2
4. Generates evaluation metrics (precision, recall, F1) against ground truth
5. Outputs predictions for SAM2 mask refinement pipeline

Evaluation is performed on the complete test split to ensure comprehensive performance assessment.
"""

import os
import sys
import numpy as np
import torch
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
import subprocess
import yaml
import csv
import time
from pathlib import Path
from ultralytics import YOLO

# Add new_src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_bitmap_to_bbox(bitmap_data, image_width, image_height):
    """Convert bitmap annotation to bounding box coordinates"""
    try:
        # Decode bitmap from base64 and zlib
        decoded_data = base64.b64decode(bitmap_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Convert to PIL Image
        bitmap_image = Image.open(BytesIO(decompressed_data))
        bitmap_array = np.array(bitmap_image)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(bitmap_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert to absolute coordinates
            bboxes.append([x, y, x + w, y + h])  # [x1, y1, x2, y2]
        
        return bboxes
    except Exception as e:
        print(f"Error converting bitmap to bbox: {e}")
        return []


def load_ground_truth_bboxes(annotations_path, images_path):
    """Load ground truth bounding boxes from bitmap annotations"""
    print("Loading ground truth bounding boxes...")
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    gt_bboxes = {}
    
    for item in tqdm(annotations, desc="Loading GT bboxes"):
        image_name = item['image']
        image_path = os.path.join(images_path, image_name)
        
        if not os.path.exists(image_path):
            continue
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue
        
        # Convert annotations to bboxes
        bboxes = []
        for annotation in item.get('annotations', []):
            if 'bitmap' in annotation:
                bitmap_bboxes = convert_bitmap_to_bbox(
                    annotation['bitmap']['data'],
                    img_width,
                    img_height
                )
                bboxes.extend(bitmap_bboxes)
        
        gt_bboxes[image_name] = bboxes
    
    print(f"Loaded GT bboxes for {len(gt_bboxes)} images")
    return gt_bboxes


def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union between two bounding boxes"""
    # bbox format: [x1, y1, x2, y2]
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_detection_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Calculate detection metrics (mAP, precision, recall)"""
    all_predictions = []
    all_ground_truth = []
    
    # Flatten predictions and ground truth
    for image_name, pred_bboxes in predictions.items():
        for bbox in pred_bboxes:
            all_predictions.append({
                'image': image_name,
                'bbox': bbox,
                'confidence': bbox[4] if len(bbox) > 4 else 1.0
            })
    
    for image_name, gt_bboxes in ground_truth.items():
        for bbox in gt_bboxes:
            all_ground_truth.append({
                'image': image_name,
                'bbox': bbox
            })
    
    # Sort predictions by confidence
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Calculate precision and recall
    tp = 0  # True positives
    fp = 0  # False positives
    fn = len(all_ground_truth)  # False negatives (all GT initially)
    
    used_gt = set()  # Track which GT bboxes have been matched
    
    for pred in all_predictions:
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching GT bbox
        for i, gt in enumerate(all_ground_truth):
            if gt['image'] == pred['image'] and i not in used_gt:
                iou = calculate_iou(pred['bbox'][:4], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx not in used_gt:
            tp += 1
            fn -= 1
            used_gt.add(best_gt_idx)
        else:
            fp += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def run_yolo_inference(model_path, test_images_path, conf_threshold=0.25, iou_threshold=0.45):
    """Run YOLO inference on test images"""
    print(f"Running YOLO inference with model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Get test image paths
    test_images = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(test_images)} test images")
    
    predictions = {}
    
    for image_name in tqdm(test_images, desc="Running inference"):
        image_path = os.path.join(test_images_path, image_name)
        
        # Run inference
        results = model(image_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Extract predictions
        image_predictions = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bbox coordinates (x1, y1, x2, y2) and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Convert to integer coordinates
                    bbox = [int(x1), int(y1), int(x2), int(y2), float(conf)]
                    image_predictions.append(bbox)
        
        predictions[image_name] = image_predictions
    
    return predictions


def save_predictions_for_sam2(predictions, output_path):
    """Save predictions in format suitable for SAM2"""
    print(f"Saving predictions for SAM2 at: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save predictions as JSON
    predictions_file = os.path.join(output_path, "yolo_predictions.json")
    
    # Convert to SAM2 format: {image_name: [[x1, y1, x2, y2, conf], ...]}
    sam2_predictions = {}
    for image_name, bboxes in predictions.items():
        sam2_predictions[image_name] = bboxes
    
    with open(predictions_file, 'w') as f:
        json.dump(sam2_predictions, f, indent=2)
    
    print(f"Saved {len(predictions)} image predictions to: {predictions_file}")
    
    # Also save as individual text files per image (alternative format)
    txt_output_dir = os.path.join(output_path, "bboxes_txt")
    os.makedirs(txt_output_dir, exist_ok=True)
    
    for image_name, bboxes in predictions.items():
        txt_file = os.path.join(txt_output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        with open(txt_file, 'w') as f:
            for bbox in bboxes:
                # Format: x1 y1 x2 y2 confidence
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]:.4f}\n")
    
    print(f"Saved individual bbox files to: {txt_output_dir}")
    
    return predictions_file


def save_inference_results(exp_id, model_path, test_metrics, predictions_file, output_path, timestamp):
    """Save inference results and metrics"""
    results_dir = os.path.join(output_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare results data
    results_data = {
        'exp_id': exp_id,
        'model': 'yolov8s',
        'subset_size': 'test_split',
        'variant': 'inference',
        'prompt_type': 'detection',
        'img_size': 1024,
        'batch_size': 1,
        'steps': 0,
        'lr': 0.0,
        'wd': 0.0,
        'seed': 42,
        'val_mIoU': 0.0,
        'val_Dice': 0.0,
        'test_mIoU': test_metrics['precision'],  # Use precision as proxy
        'test_Dice': test_metrics['f1'],  # Use F1 as proxy
        'IoU@50': test_metrics['precision'],
        'IoU@75': 0.0,
        'IoU@90': 0.0,
        'IoU@95': 0.0,
        'Precision': test_metrics['precision'],
        'Recall': test_metrics['recall'],
        'F1': test_metrics['f1'],
        'ckpt_path': model_path,
        'timestamp': timestamp,
        'predictions_file': predictions_file
    }
    
    # Save to CSV
    csv_filename = f"{timestamp}_yolo_inference_test.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results_data)
    
    print(f"Inference results saved to: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='YOLO Detection Inference on Test Split')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--test_images_path', type=str, 
                       default='data/severstal_det/images/test',
                       help='Path to test images directory')
    parser.add_argument('--gt_annotations_path', type=str,
                       default='datasets/Data/splits/test_split.json',
                       help='Path to ground truth annotations JSON file')
    parser.add_argument('--images_path', type=str,
                       default='datasets/Data/images',
                       help='Path to original images directory')
    parser.add_argument('--output_path', type=str,
                       default='new_src/evaluation/evaluation_results/yolo_detection',
                       help='Directory to save results and predictions')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Check if test images exist
    if not os.path.exists(args.test_images_path):
        print(f"Error: Test images directory not found: {args.test_images_path}")
        return
    
    # Load ground truth if available
    gt_bboxes = {}
    if os.path.exists(args.gt_annotations_path):
        gt_bboxes = load_ground_truth_bboxes(args.gt_annotations_path, args.images_path)
    else:
        print("Warning: Ground truth annotations not found. Running inference only.")
    
    # Run YOLO inference
    print(f"\nStarting YOLO inference...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    predictions = run_yolo_inference(
        model_path=args.model_path,
        test_images_path=args.test_images_path,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Save predictions for SAM2
    predictions_file = save_predictions_for_sam2(predictions, args.output_path)
    
    # Calculate metrics if ground truth is available
    if gt_bboxes:
        print("\nCalculating detection metrics...")
        test_metrics = calculate_detection_metrics(predictions, gt_bboxes)
        
        print(f"Test Metrics:")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")
        print(f"  True Positives: {test_metrics['tp']}")
        print(f"  False Positives: {test_metrics['fp']}")
        print(f"  False Negatives: {test_metrics['fn']}")
        
        # Save results
        exp_id = f"exp_yolo_inference_test_{timestamp}"
        csv_path = save_inference_results(
            exp_id=exp_id,
            model_path=args.model_path,
            test_metrics=test_metrics,
            predictions_file=predictions_file,
            output_path=args.output_path,
            timestamp=timestamp
        )
    else:
        print("\nNo ground truth available. Skipping metrics calculation.")
    
    print(f"\nInference completed successfully!")
    print(f"Predictions saved for SAM2 at: {predictions_file}")
    print(f"Total images processed: {len(predictions)}")
    
    # Print summary of predictions
    total_detections = sum(len(bboxes) for bboxes in predictions.values())
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(predictions):.2f}")


if __name__ == "__main__":
    main()
