#!/usr/bin/env python3
"""
Evaluate YOLO+SAM2 pipeline on all Severstal subsets for comparison with SAM2-only
"""

import os
import sys
import json
import numpy as np
sys.path.append('src')
from YOLO.yolov8sam2 import YOLO2SAM2Pipeline
from utils import decode_bitmap_to_mask, compute_iou

def compute_metrics(pred_mask, gt_mask):
    """Compute precision, recall, F1-score, and benevolent metrics"""
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    # Calculate precision and recall
    tp = intersection
    fp = pred_binary.sum() - intersection
    fn = gt_binary.sum() - intersection
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate benevolent (IoU with relaxed threshold)
    # Benevolent considers a prediction correct if it has some overlap with ground truth
    benevolent = 1.0 if intersection > 0 else 0.0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'benevolent': benevolent
    }

def evaluate_subset(subset_name, pipeline):
    """Evaluate pipeline on a specific subset"""
    print(f"\n Evaluating {subset_name}...")
    
    # Define paths for this subset
    img_dir = f"datasets/yolo_subsets/sub_{subset_name}/images/val"
    ann_dir = f"datasets/yolo_subsets/sub_{subset_name}/labels/val"
    
    if not os.path.exists(img_dir):
        print(f" Image directory not found: {img_dir}")
        return None
    
    # Get list of images
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    print(f"   Found {len(image_files)} images")
    
    all_metrics = []
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, img_file + '.json')
        
        # Process image
        pred_mask, detections, scores = pipeline.process_image(img_path)
        
        if pred_mask is None:
            continue
        
        # Load ground truth if available
        gt_mask = None
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    gt_data = json.load(f)
                
                h, w = pred_mask.shape
                masks_gt = [decode_bitmap_to_mask(obj['bitmap']['data'])
                           for obj in gt_data["objects"] if obj.get("geometryType") == "bitmap"]
                origins = [obj["bitmap"]["origin"] for obj in gt_data["objects"] if obj.get("geometryType") == "bitmap"]
                
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                for m, (x0, y0) in zip(masks_gt, origins):
                    gt_mask[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = np.maximum(
                        gt_mask[y0:y0 + m.shape[0], x0:x0 + m.shape[1]], m)
            except Exception as e:
                print(f"  Error loading annotation for {img_file}: {e}")
                gt_mask = None
        
        # Calculate metrics if ground truth available
        metrics = None
        if gt_mask is not None:
            metrics = compute_metrics(pred_mask, gt_mask)
            all_metrics.append(metrics)
        
        # Save results
        result = {
            'image': img_file,
            'detections': len(detections),
            'metrics': metrics,
            'mean_score': float(np.mean(scores)) if scores is not None and len(scores) > 0 else 0.0
        }
        results.append(result)
    
    # Calculate aggregate metrics
    if all_metrics:
        # Calculate means
        mean_iou = np.mean([m['iou'] for m in all_metrics])
        mean_precision = np.mean([m['precision'] for m in all_metrics])
        mean_recall = np.mean([m['recall'] for m in all_metrics])
        mean_f1 = np.mean([m['f1_score'] for m in all_metrics])
        mean_benevolent = np.mean([m['benevolent'] for m in all_metrics])
        
        # Calculate IoU thresholds
        iou_50 = np.mean([m['iou'] >= 0.5 for m in all_metrics]) * 100
        iou_75 = np.mean([m['iou'] >= 0.75 for m in all_metrics]) * 100
        
        # Calculate F1 thresholds
        f1_50 = np.mean([m['f1_score'] >= 0.5 for m in all_metrics]) * 100
        f1_75 = np.mean([m['f1_score'] >= 0.75 for m in all_metrics]) * 100
        
        avg_detections = np.mean([r['detections'] for r in results])
        
        print(f"   Mean IoU: {mean_iou:.4f}")
        print(f"   Mean Precision: {mean_precision:.4f}")
        print(f"   Mean Recall: {mean_recall:.4f}")
        print(f"   Mean F1-Score: {mean_f1:.4f}")
        print(f"   Mean Benevolent: {mean_benevolent:.4f}")
        print(f"   IoU@50: {iou_50:.2f}%")
        print(f"   IoU@75: {iou_75:.2f}%")
        print(f"   F1@50: {f1_50:.2f}%")
        print(f"   F1@75: {f1_75:.2f}%")
        print(f"   Avg detections: {avg_detections:.2f}")
        
        return {
            'subset': subset_name,
            'mean_iou': float(mean_iou),
            'mean_precision': float(mean_precision),
            'mean_recall': float(mean_recall),
            'mean_f1_score': float(mean_f1),
            'mean_benevolent': float(mean_benevolent),
            'iou_at_50': float(iou_50),
            'iou_at_75': float(iou_75),
            'f1_at_50': float(f1_50),
            'f1_at_75': float(f1_75),
            'avg_detections': float(avg_detections),
            'images_processed': len(results),
            'images_with_gt': len(all_metrics)
        }
    else:
        print(f"   No valid metrics calculations")
        return None

def main():
    # Configuration
    YOLO_MODEL = "runs/detect/yolov8n_severstal3/weights/best.pt"
    SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"
    SAM2_BASE = "models/sam2_base_models/sam2_hiera_large.pt"
    SAM2_CKPT = "models/severstal_updated/sam2_large_7ksteps_lr1e3_val500_best_step3000_iou0.18.torch"
    
    print(" Initializing YOLO+SAM2 Pipeline...")
    
    # Initialize pipeline
    pipeline = YOLO2SAM2Pipeline(
        yolo_model_path=YOLO_MODEL,
        sam2_config=SAM2_CONFIG,
        sam2_base=SAM2_BASE,
        sam2_checkpoint=SAM2_CKPT,
        device="cuda:0"
    )
    
    print(" Pipeline initialized successfully!")
    
    # Evaluate all subsets
    subsets = ['25', '50', '100', '200', '500']
    results = {}
    
    for subset in subsets:
        result = evaluate_subset(subset, pipeline)
        if result:
            results[subset] = result
    
    # Save results
    output_dir = "results/yolo2sam2_subsets"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "subset_evaluation.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n Summary of YOLO+SAM2 evaluation:")
    print(f"{'Subset':<8} {'IoU':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Bene':<8} {'IoU@50':<8} {'F1@50':<8}")
    print("-" * 80)
    
    for subset in subsets:
        if subset in results:
            r = results[subset]
            print(f"{subset:<8} {r['mean_iou']:<8.4f} {r['mean_precision']:<8.4f} {r['mean_recall']:<8.4f} "
                  f"{r['mean_f1_score']:<8.4f} {r['mean_benevolent']:<8.4f} {r['iou_at_50']:<8.2f} {r['f1_at_50']:<8.2f}")
    
    print(f"\n Results saved to: {results_file}")
    
    # Comparison with SAM2-only (you can add your previous results here)
    print(f"\n For comparison with SAM2-only results:")
    print(f"   - Add your SAM2-only results to compare")
    print(f"   - Look for patterns in how each approach scales with data size")

if __name__ == "__main__":
    main() 