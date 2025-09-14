#!/usr/bin/env python3
"""
Evaluate YOLO Detection model on corrected Severstal test dataset
Test data: datasets/yolo_detection_fixed/images/test (full dataset)
       or: datasets/yolo_detection_fixed/subset_XXX/images/test (subset)

This script evaluates the trained YOLO detection model on the corrected dataset
and saves predictions in the format needed for the SAM2 refinement pipeline.
Supports both full dataset and subset evaluation.
"""

import os
import sys
import subprocess
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import yaml

def materialize_empty_txts(images_dir, labels_dir):
    """Create empty .txt files for images without detections (YOLO format)."""
    os.makedirs(labels_dir, exist_ok=True)
    imgs = {p.stem for p in Path(images_dir).glob("*.*")}
    lbls = {p.stem for p in Path(labels_dir).glob("*.txt")}
    missing = imgs - lbls
    for stem in missing:
        open(Path(labels_dir)/f"{stem}.txt", "w").close()
    return len(missing)

def run_yolo_test_metrics(model_path, dataset_yaml, conf=0.25, iou=0.70, imgsz=1024):
    """Run YOLO validation to get test metrics (mAP, precision, recall)"""
    
    cmd = [
        "yolo", "task=detect", "mode=val",
        f"model={model_path}",
        f"data={dataset_yaml}",
        "split=test",
        f"imgsz={imgsz}",
        f"conf={conf}",
        f"iou={iou}",
        "save_json=False"
    ]
    
    print(f"Running YOLO test metrics evaluation:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Test metrics evaluation completed successfully!")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Test metrics evaluation failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr

def run_yolo_evaluation(model_path, dataset_yaml, output_dir, conf=0.25, iou=0.70, 
                       max_det=300, agnostic_nms=False, imgsz=1024, subset_size=None):
    """Run YOLO evaluation with specified parameters"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique name based on subset size
    if subset_size is not None:
        exp_name = f"predict_test_corrected_subset_{subset_size}"
    else:
        exp_name = "predict_test_corrected_full"
    
    # YOLO evaluation command (removed rect parameter as it's not applicable for predict)
    cmd = [
        "yolo", "task=detect", "mode=predict",
        f"model={model_path}",
        f"source={os.path.join(os.path.dirname(dataset_yaml), 'images', 'test_split')}",
        f"imgsz={imgsz}",
        f"conf={conf}",
        f"iou={iou}",
        f"max_det={max_det}",
        f"agnostic_nms={agnostic_nms}",
        "save_txt=True",
        "save_conf=True",
        f"project={output_dir}",
        f"name={exp_name}",
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

def parse_yolo_results(results_dir):
    """Parse YOLO validation results from results.csv"""
    results_file = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_file):
        return {}
    
    try:
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                # Get the last line (final epoch results)
                last_line = lines[-1].strip().split(',')
                if len(last_line) >= 6:
                    return {
                        'mAP50': float(last_line[5]) if last_line[5] else 0.0,
                        'mAP50_95': float(last_line[6]) if len(last_line) > 6 and last_line[6] else 0.0,
                        'precision': float(last_line[7]) if len(last_line) > 7 and last_line[7] else 0.0,
                        'recall': float(last_line[8]) if len(last_line) > 8 and last_line[8] else 0.0
                    }
    except Exception as e:
        print(f"Warning: Could not parse results.csv: {e}")
    
    return {}

def save_pipeline_metadata(output_dir, model_path, predictions_dir, args):
    """Save pipeline metadata for experiment tracking"""
    meta = {
        "model_type": "yolov8s",
        "model_path": model_path,
        "predictions_dir": predictions_dir,
        "evaluation_params": {
            "conf": args.conf, 
            "iou": args.iou, 
            "max_det": args.max_det,
            "agnostic_nms": args.agnostic_nms, 
            "imgsz": args.imgsz
        },
        "output_format": "yolo_txt",
        "bbox_format": "normalized_xywh",
        "dataset": "yolo_detection_fixed",
        "test_split": "test_split"
    }
    
    metadata_file = os.path.join(output_dir, "pipeline_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Pipeline metadata saved to: {metadata_file}")
    return metadata_file

def save_evaluation_results(exp_id, model, dataset, variant, conf, iou, max_det, 
                           agnostic_nms, predictions_dir, save_dir, timestamp, 
                           detection_metrics=None, subset_size=None):
    """Save evaluation results to CSV following TFM format"""
    
    # Create results directory
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get detection metrics if available
    if detection_metrics is None:
        detection_metrics = {}
    
    # Determine subset size for results
    subset_str = str(subset_size) if subset_size is not None else 'full'
    
    # Prepare results data
    results_data = {
        'exp_id': exp_id,
        'model': model,
        'subset_size': subset_str,
        'variant': variant,
        'prompt_type': 'bbox',
        'img_size': 1024,
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
        'Precision': detection_metrics.get('precision', 0.0),
        'Recall': detection_metrics.get('recall', 0.0),
        'F1': 0.0,  # Will be calculated if precision and recall are available
        'mAP50': detection_metrics.get('mAP50', 0.0),
        'mAP50_95': detection_metrics.get('mAP50_95', 0.0),
        'ckpt_path': model,
        'timestamp': timestamp,
        'dataset': dataset,
        'conf_threshold': conf,
        'iou_threshold': iou,
        'max_detections': max_det,
        'agnostic_nms': agnostic_nms
    }
    
    # Calculate F1 if precision and recall are available
    if results_data['Precision'] > 0 and results_data['Recall'] > 0:
        results_data['F1'] = 2 * (results_data['Precision'] * results_data['Recall']) / (results_data['Precision'] + results_data['Recall'])
    
    # Save to CSV
    csv_file = os.path.join(results_dir, f"{exp_id}_yolo_detect_corrected.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_data.keys())
        writer.writeheader()
        writer.writerow(results_data)
    
    print(f"Results saved to: {csv_file}")
    return csv_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO detection on corrected dataset")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained YOLO model (.pt file)")
    parser.add_argument("--dataset-yaml", type=str, default=None,
                       help="Path to dataset YAML file (auto-detected if not provided)")
    parser.add_argument("--subset-size", type=int, default=None,
                       help="Subset size for evaluation (e.g., 500, 1000, 2000). Uses full dataset if not specified.")
    parser.add_argument("--output-dir", type=str, default="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection",
                       help="Output directory for predictions")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.70, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--agnostic-nms", action="store_true", help="Use agnostic NMS")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")

    parser.add_argument("--save-dir", type=str, default="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection",
                       help="Directory to save results CSV")
    parser.add_argument("--exp-id", type=str, default=None,
                       help="Experiment ID (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Auto-detect dataset YAML if not provided
    if args.dataset_yaml is None:
        if args.subset_size is not None:
            # Use subset dataset
            args.dataset_yaml = f"datasets/yolo_detection_fixed/subset_{args.subset_size}/yolo_detection_{args.subset_size}.yaml"
        else:
            # Use full dataset
            args.dataset_yaml = "datasets/yolo_detection_fixed/yolo_detection.yaml"
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Check if dataset YAML exists
    if not os.path.exists(args.dataset_yaml):
        print(f"Error: Dataset YAML file not found: {args.dataset_yaml}")
        sys.exit(1)
    
    # Check if test directory exists
    test_dir = os.path.join(os.path.dirname(args.dataset_yaml), "images", "test_split")
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    # Generate experiment ID if not provided
    if args.exp_id is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        if args.subset_size is not None:
            args.exp_id = f"exp_yolo_detect_corrected_subset_{args.subset_size}_bbox_1024_{timestamp}"
        else:
            args.exp_id = f"exp_yolo_detect_corrected_full_bbox_1024_{timestamp}"
    
    print(f"Model: {args.model_path}")
    print(f"Dataset YAML: {args.dataset_yaml}")
    print(f"Test directory: {test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Max detections: {args.max_det}")
    print(f"Agnostic NMS: {args.agnostic_nms}")
    print(f"Experiment ID: {args.exp_id}")
    if args.subset_size:
        print(f"Subset size: {args.subset_size}")
    else:
        print(f"Using full dataset")
    
    # Start evaluation
    print("\nStarting YOLO detection evaluation...")
    start_time = datetime.now()
    
    # First run test metrics evaluation to get mAP, precision, recall
    print("\n1. Running test metrics evaluation...")
    metrics_success, metrics_output = run_yolo_test_metrics(
        model_path=args.model_path,
        dataset_yaml=args.dataset_yaml,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz
    )
    
    # Parse detection metrics if available
    detection_metrics = {}
    if metrics_success:
        # Look for results.csv in the default YOLO output directory (current working dir)
        results_dir = os.path.join(os.getcwd(), "runs", "detect", "val")
        detection_metrics = parse_yolo_results(results_dir)
        
        # Also try the original path as fallback
        if not detection_metrics:
            results_dir = os.path.join(args.output_dir, "runs", "detect", "val")
            detection_metrics = parse_yolo_results(results_dir)
        
        # Try to get metrics from training results as fallback
        if not detection_metrics:
            training_results_file = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), "results.csv")
            if os.path.exists(training_results_file):
                print(f"Using training results from: {training_results_file}")
                # Parse the training results file directly
                try:
                    with open(training_results_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            # Get the last line (final epoch results)
                            last_line = lines[-1].strip().split(',')
                            if len(last_line) >= 6:
                                detection_metrics = {
                                    'mAP50': float(last_line[5]) if last_line[5] else 0.0,
                                    'mAP50_95': float(last_line[6]) if len(last_line) > 6 and last_line[6] else 0.0,
                                    'precision': float(last_line[3]) if len(last_line) > 3 and last_line[3] else 0.0,
                                    'recall': float(last_line[4]) if len(last_line) > 4 and last_line[4] else 0.0
                                }
                except Exception as e:
                    print(f"Error parsing training results: {e}")
                    detection_metrics = {}
        
        if detection_metrics:
            print(f"Detection metrics: {detection_metrics}")
        else:
            print("Warning: Could not parse detection metrics from any results.csv")
    
    # Then run prediction to generate .txt files
    print("\n2. Running prediction to generate .txt files...")
    success, output = run_yolo_evaluation(
        model_path=args.model_path,
        dataset_yaml=args.dataset_yaml,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        imgsz=args.imgsz,
        subset_size=args.subset_size
    )
    
    if success:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\nEvaluation completed successfully in {duration:.2f} seconds")
        
        # Find predictions directory
        if args.subset_size is not None:
            exp_name = f"predict_test_corrected_subset_{args.subset_size}"
        else:
            exp_name = "predict_test_corrected_full"
        
        predictions_dir = os.path.join(args.output_dir, exp_name, "labels")
        if os.path.exists(predictions_dir):
            print(f"Predictions saved to: {predictions_dir}")
            
            # Count prediction files
            pred_files = list(Path(predictions_dir).glob("*.txt"))
            print(f"Generated {len(pred_files)} prediction files")
            
            # Create empty .txt files for images without detections
            images_dir = os.path.join(os.path.dirname(args.dataset_yaml), "images", "test_split")
            created = materialize_empty_txts(images_dir, predictions_dir)
            print(f"Created {created} empty .txt files → total should be 2000.")
            
            # Save pipeline metadata
            save_pipeline_metadata(args.output_dir, args.model_path, predictions_dir, args)
            
            # Save results to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_file = save_evaluation_results(
                exp_id=args.exp_id,
                model=os.path.basename(args.model_path),
                dataset="yolo_detection_fixed",
                variant="corrected",
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                agnostic_nms=args.agnostic_nms,
                predictions_dir=predictions_dir,
                save_dir=args.save_dir,
                timestamp=timestamp,
                detection_metrics=detection_metrics,
                subset_size=args.subset_size
            )
            
            print(f"\nEvaluation results:")
            print(f"- Predictions: {predictions_dir}")
            print(f"- Results CSV: {csv_file}")
            print(f"- Pipeline metadata: {os.path.join(args.output_dir, 'pipeline_metadata.json')}")
            print(f"- Ready for SAM2 refinement pipeline")
        else:
            print("Warning: Predictions directory not found")
    else:
        print("\nEvaluation failed!")
        print("Output:", output)
        sys.exit(1)

if __name__ == "__main__":
    main()
