#!/usr/bin/env python3
"""
Script to run YOLO2SAM2 refined segmentation on Severstal dataset
"""

import os
import sys
sys.path.append('src')
from YOLO.yolov8sam2 import YOLO2SAM2Pipeline

def main():
    # Configuration paths
    YOLO_MODEL = "runs/detect/yolov8n_severstal3/weights/best.pt"  # Use your trained model
    SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"
    SAM2_BASE = "models/sam2_base_models/sam2_hiera_large.pt"
    SAM2_CKPT = "models/severstal_updated/sam2_large_7ksteps_lr1e3_val500_best_step3000_iou0.18.torch"
    
    # Dataset paths
    TRAIN_IMG_DIR = "datasets/yolo_subsets/images/train"
    TRAIN_ANN_DIR = "datasets/severstal/train_split/ann"
    VAL_IMG_DIR = "datasets/yolo_subsets/images/val"
    VAL_ANN_DIR = "datasets/severstal/test_split/ann"
    
    # Output directory
    OUTPUT_DIR = "results/yolo2sam2_severstal"
    
    print("🚀 Initializing YOLO2SAM2 Pipeline...")
    
    # Initialize pipeline
    pipeline = YOLO2SAM2Pipeline(
        yolo_model_path=YOLO_MODEL,
        sam2_config=SAM2_CONFIG,
        sam2_base=SAM2_BASE,
        sam2_checkpoint=SAM2_CKPT,
        device="cuda:0"
    )
    
    print("✅ Pipeline initialized successfully!")
    
    # Evaluate on validation set
    print(f"\n📊 Evaluating on validation set...")
    print(f"Images: {VAL_IMG_DIR}")
    print(f"Annotations: {VAL_ANN_DIR}")
    
    val_results, val_ious = pipeline.evaluate_on_dataset(
        image_dir=VAL_IMG_DIR,
        annotation_dir=VAL_ANN_DIR,
        output_dir=os.path.join(OUTPUT_DIR, "val_visualizations")
    )
    
    # Save results
    import json
    import numpy as np
    results_file = os.path.join(OUTPUT_DIR, "validation_results.json")
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        results_serializable = []
        for result in val_results:
            results_serializable.append({
                'image': result['image'],
                'detections': int(result['detections']),
                'iou': float(result['iou']) if result['iou'] is not None else None,
                'mean_score': float(result['mean_score'])
            })
        
        ious_serializable = [float(iou) for iou in val_ious] if val_ious else []
        
        json.dump({
            'results': results_serializable,
            'ious': ious_serializable,
            'mean_iou': float(np.mean(val_ious)) if val_ious else 0.0
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"📁 Visualizations saved to: {OUTPUT_DIR}/val_visualizations/")

if __name__ == "__main__":
    main() 