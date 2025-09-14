#!/usr/bin/env python3
"""
Evaluate YOLO-Detect on Severstal test split
Test data: datasets/Data/splits/test_split (2000 images)

Object detection: defect instances as bounding boxes
Output: YOLO format .txt files with bboxes for pipeline integration

Fixed hyperparameters for pipeline traceability:
- conf=0.25, iou=0.70, max_det=300
- agnostic_nms=False, imgsz=1024, rect=True
- save_txt=True, save_conf=True

This script evaluates the best trained YOLO model and saves predictions
in the format needed for the SAM2 refinement pipeline.
"""

import os
import sys
import argparse
import subprocess
import json
import csv
from datetime import datetime
from pathlib import Path
import yaml

# Add new_src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_yolo_evaluation(model_path, dataset_yaml, output_dir, conf=0.25, iou=0.70, 
                       max_det=300, agnostic_nms=False, imgsz=1024, rect=True):
    """Run YOLO evaluation with specified parameters"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # YOLO evaluation command
    cmd = [
        "yolo", "task=detect", "mode=predict",
        f"model={model_path}",
        f"source={os.path.join(os.path.dirname(dataset_yaml), 'images', 'test')}",
        f"imgsz={imgsz}",
        f"rect={rect}",
        f"conf={conf}",
        f"iou={iou}",
        f"max_det={max_det}",
        f"agnostic_nms={agnostic_nms}",
        "save_txt=True",
        "save_conf=True",
        f"project={output_dir}",
        "name=predict_test",
        "exist_ok=True"
    ]
    
    print(f"Running YOLO evaluation command:")
    print(" ".join(cmd))
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Evaluation completed successfully!")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr


def parse_evaluation_results(eval_dir):
    """Parse YOLO evaluation results - in predict mode, we don't have metrics CSV"""
    # In predict mode, YOLO doesn't generate metrics CSV
    # We'll create a basic metrics structure for pipeline compatibility
    print("Note: YOLO predict mode doesn't generate metrics CSV")
    print("Creating basic metrics structure for pipeline compatibility")
    
    # Return basic metrics structure
    results = {
        'metrics/mAP50(B)': 0.0,  # Will be filled later if available
        'metrics/mAP50-95(B)': 0.0,
        'metrics/mAP75(B)': 0.0,
        'metrics/precision(B)': 0.0,
        'metrics/recall(B)': 0.0
    }
    
    return results


def save_evaluation_results(exp_id, model, subset_size, variant, prompt_type, img_size, 
                           conf, iou, max_det, agnostic_nms, test_metrics, 
                           predictions_dir, save_dir, timestamp):
    """Save evaluation results to CSV following TFM format"""
    
    # Use save_dir directly (no subdirectory)
    results_dir = save_dir
    
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
        'test_mAP50': test_metrics.get('metrics/mAP50(B)', 0.0),
        'test_mAP50_95': test_metrics.get('metrics/mAP50-95(B)', 0.0),
        'IoU@50': test_metrics.get('metrics/mAP50(B)', 0.0),
        'IoU@75': test_metrics.get('metrics/mAP75(B)', 0.0),
        'IoU@90': 0.0,  # YOLO doesn't provide this
        'IoU@95': 0.0,  # YOLO doesn't provide this
        'Precision': test_metrics.get('metrics/precision(B)', 0.0),
        'Recall': test_metrics.get('metrics/recall(B)', 0.0),
        'F1': 0.0,  # Will calculate if possible
        'ckpt_path': predictions_dir,
        'timestamp': timestamp,
        'conf_threshold': conf,
        'iou_threshold': iou,
        'max_detections': max_det,
        'agnostic_nms': agnostic_nms
    }
    
    # Calculate F1 if precision and recall are available
    if results_data['Precision'] > 0 and results_data['Recall'] > 0:
        results_data['F1'] = 2 * (results_data['Precision'] * results_data['Recall']) / (results_data['Precision'] + results_data['Recall'])
    
    # Save to CSV
    csv_filename = f"{timestamp}_{model}_{variant}_detect_eval_{conf}_{iou}_{max_det}.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results_data)
    
    print(f"Evaluation results saved to: {csv_path}")
    return csv_path


def ensure_all_images_have_predictions(test_images_dir, predictions_dir):
    """Ensure all test images have corresponding .txt files (empty if no detections)"""
    print(f"Ensuring all test images have prediction files...")
    
    # Get all test images
    test_images = set()
    for ext in ['.jpg', '.jpeg', '.png']:
        test_images.update([f.replace(ext, '') for f in os.listdir(test_images_dir) if f.endswith(ext)])
    
    # Get existing predictions
    existing_preds = set()
    for f in os.listdir(predictions_dir):
        if f.endswith('.txt'):
            existing_preds.add(f.replace('.txt', ''))
    
    # Create empty .txt files for images without detections
    missing_preds = test_images - existing_preds
    if missing_preds:
        print(f"Creating {len(missing_preds)} empty prediction files for images without detections")
        for img_name in missing_preds:
            empty_txt_path = os.path.join(predictions_dir, f"{img_name}.txt")
            with open(empty_txt_path, 'w') as f:
                pass  # Empty file
    else:
        print("All test images already have prediction files")
    
    total_files = len(test_images)
    print(f"Total prediction files: {total_files} (including {len(missing_preds)} empty)")
    return total_files


def create_pipeline_metadata(predictions_dir, output_dir, conf, iou, max_det, agnostic_nms):
    """Create metadata file for pipeline integration"""
    
    metadata = {
        'model_type': 'yolo_detect',
        'predictions_dir': predictions_dir,
        'evaluation_params': {
            'conf_threshold': conf,
            'iou_threshold': iou,
            'max_detections': max_det,
            'agnostic_nms': agnostic_nms,
            'imgsz': 1024,
            'rect': True
        },
        'output_format': 'yolo_txt',
        'bbox_format': 'normalized_xywh',
        'class_mapping': {
            '0': 'defect'
        },
        'pipeline_stage': 'detection',
        'next_stage': 'sam2_refinement'
    }
    
    metadata_file = os.path.join(output_dir, "pipeline_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Pipeline metadata saved to: {metadata_file}")
    return metadata_file


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO-Detect on Severstal test split')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained YOLO model weights (.pt file)')
    parser.add_argument('--dataset_yaml', type=str, default='data/severstal_det.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--output_dir', type=str, 
                       default='new_src/evaluation/evaluation_results/yolo_detection',
                       help='Directory to save evaluation results')
    parser.add_argument('--conf', type=float, default=0.15,
                       help='Confidence threshold (lower = more recall for SAM2 prompts)')
    parser.add_argument('--iou', type=float, default=0.70,
                       help='IoU threshold for NMS')
    parser.add_argument('--max_det', type=int, default=300,
                       help='Maximum detections per image')
    parser.add_argument('--agnostic_nms', action='store_true',
                       help='Use agnostic NMS')
    parser.add_argument('--imgsz', type=int, default=1024,
                       help='Image size')
    parser.add_argument('--rect', action='store_true', default=True,
                       help='Use rectangular training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean previous predict_test directory if it exists
    predict_test_dir = os.path.join(args.output_dir, "predict_test")
    if os.path.exists(predict_test_dir):
        print(f"Cleaning previous predict_test directory: {predict_test_dir}")
        import shutil
        shutil.rmtree(predict_test_dir)
    
    # Check if YOLO is available
    try:
        subprocess.run(["yolo", "--version"], capture_output=True, check=True)
        print("YOLO is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: YOLO command not found. Please install ultralytics: pip install ultralytics")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Check if dataset YAML exists
    if not os.path.exists(args.dataset_yaml):
        print(f"Error: Dataset YAML file not found: {args.dataset_yaml}")
        return
    
    # Print evaluation configuration
    print(f"\nYOLO Detection Evaluation Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {args.dataset_yaml}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    print(f"  Max detections: {args.max_det}")
    print(f"  Agnostic NMS: {args.agnostic_nms}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Rectangular: {args.rect}")
    print(f"  Output directory: {args.output_dir}")
    
    # Run evaluation
    print(f"\nStarting YOLO evaluation...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    success, output = run_yolo_evaluation(
        model_path=args.model_path,
        dataset_yaml=args.dataset_yaml,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        imgsz=args.imgsz,
        rect=args.rect
    )
    
    if not success:
        print("Evaluation failed. Exiting.")
        return
    
    # Find the evaluation results directory
    eval_results_dir = os.path.join(args.output_dir, "predict_test")
    if not os.path.exists(eval_results_dir):
        print(f"Evaluation results directory not found: {eval_results_dir}")
        return
    
    # Parse results
    print(f"\nParsing evaluation results from: {eval_results_dir}")
    test_metrics = parse_evaluation_results(eval_results_dir)
    
    if test_metrics is None:
        print("Could not parse test metrics")
        return
    
    # Find predictions directory (labels)
    labels_dir = os.path.join(eval_results_dir, "labels")
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return
    
    # Count prediction files
    pred_files = list(Path(labels_dir).glob("*.txt"))
    print(f"Generated {len(pred_files)} prediction files in: {labels_dir}")
    
    # Ensure all test images have prediction files (empty if no detections)
    test_images_dir = os.path.join(os.path.dirname(args.dataset_yaml), 'images', 'test_split')
    total_predictions = ensure_all_images_have_predictions(test_images_dir, labels_dir)
    
    # Update the count for final reporting
    pred_files = list(Path(labels_dir).glob("*.txt"))
    
    # Save evaluation results
    exp_id = f"eval_yolo_detect_test_split_{timestamp}"
    
    # Determine model variant from path
    model_variant = "unknown"
    if "lr1e3" in args.model_path:
        model_variant = "lr1e3"
    elif "lr1e4" in args.model_path:
        model_variant = "lr1e4"
    
    csv_path = save_evaluation_results(
        exp_id=exp_id,
        model="yolov8s",
        subset_size="test_split",
        variant=model_variant,
        prompt_type="detection",
        img_size=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        test_metrics=test_metrics,
        predictions_dir=labels_dir,
        save_dir=args.output_dir,
        timestamp=timestamp
    )
    
    # Create pipeline metadata
    metadata_file = create_pipeline_metadata(
        predictions_dir=labels_dir,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms
    )
    
    # Print final results
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {csv_path}")
    print(f"Pipeline metadata: {metadata_file}")
    print(f"Predictions saved to: {labels_dir}")
    print(f"Total prediction files: {total_predictions} (including empty files for images without detections)")
    
    if test_metrics:
        print(f"\nTest set metrics:")
        for key, value in test_metrics.items():
            if 'mAP' in key or 'precision' in key or 'recall' in key:
                try:
                    print(f"  {key}: {float(value):.4f}")
                except:
                    print(f"  {key}: {value}")
    
    print(f"\nPipeline integration ready!")
    print(f"Bounding box predictions are saved in YOLO format (.txt files)")
    print(f"Each file contains: class x_center y_center width height (normalized 0-1)")
    print(f"Ready for SAM2 refinement stage")


if __name__ == "__main__":
    main()

