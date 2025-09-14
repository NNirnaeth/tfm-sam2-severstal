#!/usr/bin/env python3
"""
Script to run YOLO2SAM2 refined segmentation on Severstal dataset (continued)
"""

import os
import sys
import json
import numpy as np
sys.path.append('src')
from YOLO.yolov8sam2 import YOLO2SAM2Pipeline
from utils import decode_bitmap_to_mask, compute_iou

def main():
    # Configuration paths
    YOLO_MODEL = "runs/detect/yolov8n_severstal3/weights/best.pt"  # Use your trained model
    SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"
    SAM2_BASE = "models/sam2_base_models/sam2_hiera_large.pt"
    SAM2_CKPT = "models/severstal_updated/sam2_large_7ksteps_lr1e3_val500_best_step3000_iou0.18.torch"
    
    # Dataset paths
    VAL_IMG_DIR = "datasets/yolo_subsets/images/val"
    VAL_ANN_DIR = "datasets/severstal/test_split/ann"
    
    # Output directory
    OUTPUT_DIR = "results/yolo2sam2_severstal_test"
    
    print(" Initializing YOLO2SAM2 Pipeline...")
    
    # Initialize pipeline
    pipeline = YOLO2SAM2Pipeline(
        yolo_model_path=YOLO_MODEL,
        sam2_config=SAM2_CONFIG,
        sam2_base=SAM2_BASE,
        sam2_checkpoint=SAM2_CKPT,
        device="cuda:0"
    )
    
    print(" Pipeline initialized successfully!")
    
    # Test with just 10 images first
    print(f"\n Testing with first 10 images...")
    print(f"Images: {VAL_IMG_DIR}")
    print(f"Annotations: {VAL_ANN_DIR}")
    
    # Get list of images and take only first 10
    image_files = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith(('.jpg', '.png'))][:10]
    
    all_ious = []
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        ann_path = os.path.join(VAL_ANN_DIR, img_file + '.json')
        
        # Process image
        pred_mask, detections, scores = pipeline.process_image(img_path)
        
        if pred_mask is None:
            continue
        
        # Load ground truth if available
        gt_mask = None
        if os.path.exists(ann_path):
            print(f"  Found annotation: {ann_path}")
            try:
                with open(ann_path, 'r') as f:
                    gt_data = json.load(f)
                
                h, w = pred_mask.shape
                masks_gt = [decode_bitmap_to_mask(obj['bitmap']['data'])
                           for obj in gt_data["objects"] if obj.get("geometryType") == "bitmap"]
                origins = [obj["bitmap"]["origin"] for obj in gt_data["objects"] if obj.get("geometryType") == "bitmap"]
                
                print(f"  Found {len(masks_gt)} bitmap objects")
                
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                for m, (x0, y0) in zip(masks_gt, origins):
                    gt_mask[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = np.maximum(
                        gt_mask[y0:y0 + m.shape[0], x0:x0 + m.shape[1]], m)
            except Exception as e:
                print(f"  Error loading annotation: {e}")
                gt_mask = None
        else:
            print(f"  No annotation found: {ann_path}")
        
        # Calculate IoU if ground truth available
        iou = None
        if gt_mask is not None:
            iou = compute_iou(pred_mask > 0, gt_mask > 0)
            all_ious.append(iou)
        
        # Save results
        result = {
            'image': img_file,
            'detections': len(detections),
            'iou': iou,
            'mean_score': float(np.mean(scores)) if scores is not None and len(scores) > 0 else 0.0
        }
        results.append(result)
        
        iou_str = f"{iou:.4f}" if iou is not None else "N/A"
        print(f"Processed {img_file}: {len(detections)} detections, IoU: {iou_str}")
    
    # Print evaluation metrics
    if all_ious:
        mean_iou = np.mean(all_ious)
        print(f"\n--- Evaluation Results (10 images) ---")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Images processed: {len(results)}")
        print(f"Average detections per image: {np.mean([r['detections'] for r in results]):.2f}")
        
        for threshold in [0.5, 0.75, 0.9]:
            acc = np.mean([iou >= threshold for iou in all_ious])
            print(f"IoU@{int(threshold*100)}: {acc*100:.2f}%")
    
    # Save results
    results_file = os.path.join(OUTPUT_DIR, "test_results.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'ious': all_ious,
            'mean_iou': float(np.mean(all_ious)) if all_ious else 0.0
        }, f, indent=2)
    
    print(f"\n Results saved to: {results_file}")

if __name__ == "__main__":
    main() 