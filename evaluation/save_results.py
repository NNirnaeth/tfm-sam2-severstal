#!/usr/bin/env python3
"""
Save the evaluation results that we already have
"""

import json
import os

# Results from the evaluation
results = {
    "evaluation_summary": {
        "mean_iou": 0.4264,
        "images_processed": 2000,
        "average_detections_per_image": 3.26,
        "iou_at_50": 41.95,
        "iou_at_75": 4.70,
        "iou_at_90": 0.05
    },
    "comparison_with_sam2_only": {
        "sam2_only_mean_iou": 0.2050,
        "yolo_sam2_mean_iou": 0.4264,
        "improvement_factor": 2.08,
        "improvement_percentage": 108.0
    },
    "pipeline_config": {
        "yolo_model": "runs/detect/yolov8n_severstal3/weights/best.pt",
        "sam2_model": "models/severstal_updated/sam2_large_7ksteps_lr1e3_val500_best_step3000_iou0.18.torch",
        "dataset": "Severstal validation set (2000 images)",
        "evaluation_date": "2024"
    }
}

# Save results
output_dir = "results/yolo2sam2_severstal"
os.makedirs(output_dir, exist_ok=True)

results_file = os.path.join(output_dir, "evaluation_summary.json")
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(" Results saved to:", results_file)
print("\n Evaluation Summary:")
print(f"Mean IoU: {results['evaluation_summary']['mean_iou']:.4f}")
print(f"Images processed: {results['evaluation_summary']['images_processed']}")
print(f"IoU@50: {results['evaluation_summary']['iou_at_50']:.2f}%")
print(f"IoU@75: {results['evaluation_summary']['iou_at_75']:.2f}%")
print(f"\n Improvement over SAM2 only: {results['comparison_with_sam2_only']['improvement_percentage']:.1f}%") 