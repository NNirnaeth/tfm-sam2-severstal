#!/usr/bin/env python3
"""
Debug script to analyze mask generation and evaluation issues
"""

import os
import sys
import numpy as np
import cv2
import json
import base64
import zlib
from PIL import Image
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt

# Add new_src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.metrics import SegmentationMetrics

def debug_single_image():
    """Debug a single image to understand the pipeline"""
    
    # Test paths (using the small test)
    test_dir = "/tmp/debug_masks"
    os.makedirs(test_dir, exist_ok=True)
    
    # Pick first image from our test
    yolo_labels_dir = "/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection/predict_test_corrected/labels"
    gt_annotations = "/home/ptp/sam2/datasets/Data/splits/test_split/ann"
    test_images_dir = "/home/ptp/sam2/datasets/yolo_detection_fixed/images/test_split"
    
    # Get first image
    label_files = list(Path(yolo_labels_dir).glob("*.txt"))[:1]
    
    for label_file in label_files:
        image_name = label_file.stem + ".jpg"
        print(f"Debugging image: {image_name}")
        
        # Load YOLO predictions
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        bboxes = []
        print(f"YOLO predictions ({len(lines)} detections):")
        for i, line in enumerate(lines[:5]):  # Show first 5
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5]) if len(parts) >= 6 else 1.0
                
                print(f"  {i+1}: cls={class_id}, cx={x_center:.3f}, cy={y_center:.3f}, w={width:.3f}, h={height:.3f}, conf={confidence:.3f}")
                
                bboxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'conf': confidence
                })
        
        # Load ground truth
        gt_file = Path(gt_annotations) / f"{image_name}.json"
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
            
            print(f"\nGround truth objects: {len(gt_data.get('objects', []))}")
            
            # Load image to get dimensions
            image_path = Path(test_images_dir) / image_name
            if image_path.exists():
                image = cv2.imread(str(image_path))
                img_height, img_width = image.shape[:2]
                print(f"Image dimensions: {img_width}x{img_height}")
                
                # Build GT mask
                gt_union = np.zeros((img_height, img_width), np.uint8)
                for obj in gt_data.get("objects", []):
                    bmp = obj.get("bitmap", {})
                    if "data" not in bmp or "origin" not in bmp:
                        continue
                    
                    # Decode bitmap
                    raw = zlib.decompress(base64.b64decode(bmp["data"]))
                    patch = np.array(Image.open(BytesIO(raw)))
                    if patch.ndim == 3 and patch.shape[2] == 4:
                        patch = patch[:, :, 3]  # Alpha channel
                    elif patch.ndim == 3:
                        patch = patch[:, :, 0]  # First channel
                    patch = (patch > 0).astype(np.uint8)
                    
                    ox, oy = map(int, bmp["origin"])  # origin = [x, y]
                    ph, pw = patch.shape
                    x1, y1 = max(0, ox), max(0, oy)
                    x2, y2 = min(img_width, ox + pw), min(img_height, oy + ph)
                    
                    if x2 > x1 and y2 > y1:
                        gt_union[y1:y2, x1:x2] = np.maximum(
                            gt_union[y1:y2, x1:x2], patch[:(y2 - y1), :(x2 - x1)]
                        )
                
                # Create a simple prediction mask (simulate SAM2 output)
                pred_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                # Convert first few YOLO boxes to pixel coordinates and create simple masks
                for bbox in bboxes[:3]:  # Use first 3 boxes
                    cx, cy, w, h = bbox["x_center"]*img_width, bbox["y_center"]*img_height, bbox["width"]*img_width, bbox["height"]*img_height
                    x1 = max(0, int(round(cx - w/2)))
                    y1 = max(0, int(round(cy - h/2)))
                    x2 = min(img_width-1, int(round(cx + w/2)))
                    y2 = min(img_height-1, int(round(cy + h/2)))
                    
                    # Fill rectangle (simple simulation)
                    pred_mask[y1:y2, x1:x2] = 1
                
                # Debug metrics calculation
                print(f"\nMask statistics:")
                print(f"GT mask - shape: {gt_union.shape}, unique values: {np.unique(gt_union)}, sum: {gt_union.sum()}")
                print(f"Pred mask - shape: {pred_mask.shape}, unique values: {np.unique(pred_mask)}, sum: {pred_mask.sum()}")
                
                # Calculate metrics
                metrics_calculator = SegmentationMetrics()
                metrics = metrics_calculator.compute_pixel_metrics(pred_mask.flatten(), gt_union.flatten())
                
                print(f"\nMetrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                
                # Save debug images
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f'Original Image\n{image_name}')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(gt_union, cmap='gray')
                plt.title(f'Ground Truth\nSum: {gt_union.sum()}')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask, cmap='gray')
                plt.title(f'Prediction (Simulated)\nSum: {pred_mask.sum()}')
                plt.axis('off')
                
                plt.tight_layout()
                debug_path = os.path.join(test_dir, f"debug_{image_name.replace('.jpg', '.png')}")
                plt.savefig(debug_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Debug image saved: {debug_path}")
                
                # Test different thresholds
                print(f"\nTesting different thresholds:")
                for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    # Simulate probability mask
                    prob_mask = pred_mask.astype(np.float32) * 0.8  # Simulate 80% confidence
                    thresh_mask = (prob_mask > threshold).astype(np.uint8)
                    
                    metrics_thresh = metrics_calculator.compute_pixel_metrics(thresh_mask.flatten(), gt_union.flatten())
                    print(f"  Threshold {threshold}: IoU={metrics_thresh['iou']:.4f}, Dice={metrics_thresh['dice']:.4f}")
                
                break  # Only debug first image
            else:
                print(f"Image not found: {image_path}")
        else:
            print(f"GT file not found: {gt_file}")

def check_mask_formats():
    """Check if masks are in correct format"""
    print("=== MASK FORMAT CHECK ===")
    
    # Create test masks
    test_gt = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    test_pred = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.uint8)
    
    print(f"Test GT: shape={test_gt.shape}, values={np.unique(test_gt)}")
    print(f"Test Pred: shape={test_pred.shape}, values={np.unique(test_pred)}")
    
    metrics_calculator = SegmentationMetrics()
    metrics = metrics_calculator.compute_pixel_metrics(test_pred.flatten(), test_gt.flatten())
    
    print(f"Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Manual calculation
    intersection = np.logical_and(test_pred, test_gt).sum()
    union = np.logical_or(test_pred, test_gt).sum()
    manual_iou = intersection / union if union > 0 else 0
    
    print(f"\nManual calculation:")
    print(f"  Intersection: {intersection}")
    print(f"  Union: {union}")
    print(f"  Manual IoU: {manual_iou:.4f}")

if __name__ == "__main__":
    print("🔍 DEBUGGING MASK PIPELINE")
    print("=" * 50)
    
    # Check basic mask format
    check_mask_formats()
    print()
    
    # Debug single image
    debug_single_image()
    
    print("\n✅ Debug complete!")
    print("Check /tmp/debug_masks/ for visualization")









