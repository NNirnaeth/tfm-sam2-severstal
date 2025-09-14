#!/usr/bin/env python3
"""
Run complete YOLO+SAM2 pipeline with corrected dataset
This script orchestrates the entire workflow:

1. Train YOLO detection on corrected dataset
2. Evaluate YOLO on test set
3. Run YOLO+SAM2 pipeline evaluation
4. Generate comprehensive results

Dataset: datasets/yolo_detection_fixed/
Output: runs/ + data/results/
"""

import os
import sys
import subprocess
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import time

def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        if e.stderr:
            print("Error output:", e.stderr)
        return False, e.stderr

def train_yolo_detection(dataset_yaml, epochs=100, imgsz=1024, batch=16, save_dir="runs/detect"):
    """Train YOLO detection model"""
    cmd = [
        "python", "new_src/training/train_yolo_detect_corrected.py",
        f"--dataset-yaml={dataset_yaml}",
        f"--epochs={epochs}",
        f"--imgsz={imgsz}",
        f"--batch={batch}",
        f"--save-dir={save_dir}"
    ]
    
    return run_command(cmd, "YOLO Detection Training")

def evaluate_yolo_detection(model_path, dataset_yaml, output_dir="runs/evaluate", save_dir="data/results"):
    """Evaluate YOLO detection model"""
    cmd = [
        "python", "new_src/evaluation/eval_yolo_detect_corrected.py",
        f"--model-path={model_path}",
        f"--dataset-yaml={dataset_yaml}",
        f"--output-dir={output_dir}",
        f"--save-dir={save_dir}"
    ]
    
    return run_command(cmd, "YOLO Detection Evaluation")

def evaluate_yolo_sam2_pipeline(sam2_model_path, yolo_predictions_dir, test_images_dir, 
                               test_annotations_dir, output_dir="runs/evaluate", save_dir="data/results"):
    """Evaluate YOLO+SAM2 pipeline"""
    cmd = [
        "python", "new_src/evaluation/eval_yolo+sam2_full_dataset.py",
        f"--sam2-model={sam2_model_path}",
        f"--yolo-predictions={yolo_predictions_dir}",
        f"--test-images={test_images_dir}",
        f"--test-annotations={test_annotations_dir}",
        f"--output-dir={output_dir}",
        f"--save-dir={save_dir}"
    ]
    
    return run_command(cmd, "YOLO+SAM2 Pipeline Evaluation")

def find_best_yolo_model(save_dir):
    """Find the best trained YOLO model"""
    best_model = os.path.join(save_dir, "yolo_detect_corrected", "weights", "best.pt")
    if os.path.exists(best_model):
        return best_model
    
    # Fallback to last model
    last_model = os.path.join(save_dir, "yolo_detect_corrected", "weights", "last.pt")
    if os.path.exists(last_model):
        print(f"Warning: Best model not found, using last model: {last_model}")
        return last_model
    
    return None

def find_yolo_predictions(output_dir):
    """Find YOLO prediction directory"""
    predictions_dir = os.path.join(output_dir, "predict_test_corrected", "labels")
    if os.path.exists(predictions_dir):
        return predictions_dir
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Run complete YOLO+SAM2 corrected pipeline")
    parser.add_argument("--dataset-yaml", type=str, default="datasets/yolo_detection_fixed/yolo_detection.yaml",
                       help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--sam2-model", type=str, required=True,
                       help="Path to SAM2 fine-tuned model")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip YOLO training (use existing model)")
    parser.add_argument("--skip-yolo-eval", action="store_true",
                       help="Skip YOLO evaluation (use existing predictions)")
    parser.add_argument("--train-dir", type=str, default="/home/ptp/sam2/new_src/training/training_results/yolo_detection",
                       help="Directory for YOLO training")
    parser.add_argument("--eval-dir", type=str, default="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection",
                       help="Directory for evaluation outputs")
    parser.add_argument("--results-dir", type=str, default="/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection",
                       help="Directory for results CSV files")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not os.path.exists(args.dataset_yaml):
        print(f"Error: Dataset YAML file not found: {args.dataset_yaml}")
        sys.exit(1)
    
    if not os.path.exists(args.sam2_model):
        print(f"Error: SAM2 model not found: {args.sam2_model}")
        sys.exit(1)
    
    # Create directories
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"Dataset YAML: {args.dataset_yaml}")
    print(f"SAM2 Model: {args.sam2_model}")
    print(f"Training directory: {args.train_dir}")
    print(f"Evaluation directory: {args.eval_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    
    pipeline_start = time.time()
    
    # Step 1: Train YOLO detection (if not skipped)
    yolo_model_path = None
    if not args.skip_training:
        success, _ = train_yolo_detection(
            dataset_yaml=args.dataset_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            save_dir=args.train_dir
        )
        
        if not success:
            print("❌ YOLO training failed. Pipeline stopped.")
            sys.exit(1)
        
        # Find trained model
        yolo_model_path = find_best_yolo_model(args.train_dir)
        if not yolo_model_path:
            print("❌ No YOLO model found after training. Pipeline stopped.")
            sys.exit(1)
        
        print(f"✅ YOLO model trained: {yolo_model_path}")
    else:
        print("⏭️ Skipping YOLO training")
        # Try to find existing model
        yolo_model_path = find_best_yolo_model(args.train_dir)
        if not yolo_model_path:
            print("❌ No existing YOLO model found. Cannot skip training.")
            sys.exit(1)
        print(f"✅ Using existing YOLO model: {yolo_model_path}")
    
    # Step 2: Evaluate YOLO detection (if not skipped)
    yolo_predictions_dir = None
    if not args.skip_yolo_eval:
        success, _ = evaluate_yolo_detection(
            model_path=yolo_model_path,
            dataset_yaml=args.dataset_yaml,
            output_dir=args.eval_dir,
            save_dir=args.results_dir
        )
        
        if not success:
            print("❌ YOLO evaluation failed. Pipeline stopped.")
            sys.exit(1)
        
        # Find predictions
        yolo_predictions_dir = find_yolo_predictions(args.eval_dir)
        if not yolo_predictions_dir:
            print("❌ No YOLO predictions found after evaluation. Pipeline stopped.")
            sys.exit(1)
        
        print(f"✅ YOLO predictions generated: {yolo_predictions_dir}")
    else:
        print("⏭️ Skipping YOLO evaluation")
        # Try to find existing predictions
        yolo_predictions_dir = find_yolo_predictions(args.eval_dir)
        if not yolo_predictions_dir:
            print("❌ No existing YOLO predictions found. Cannot skip evaluation.")
            sys.exit(1)
        print(f"✅ Using existing YOLO predictions: {yolo_predictions_dir}")
    
    # Step 3: Evaluate YOLO+SAM2 pipeline
    test_images_dir = os.path.join(os.path.dirname(args.dataset_yaml), "images", "test_split")
    test_annotations_dir = os.path.join(os.path.dirname(args.dataset_yaml), "labels", "test_split")
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        sys.exit(1)
    
    if not os.path.exists(test_annotations_dir):
        print(f"❌ Test annotations directory not found: {test_annotations_dir}")
        sys.exit(1)
    
    success, _ = evaluate_yolo_sam2_pipeline(
        sam2_model_path=args.sam2_model,
        yolo_predictions_dir=yolo_predictions_dir,
        test_images_dir=test_images_dir,
        test_annotations_dir=test_annotations_dir,
        output_dir=args.eval_dir,
        save_dir=args.results_dir
    )
    
    if not success:
        print("❌ YOLO+SAM2 pipeline evaluation failed.")
        sys.exit(1)
    
    # Pipeline completed successfully
    pipeline_end = time.time()
    pipeline_duration = pipeline_end - pipeline_start
    
    print(f"\n{'='*60}")
    print(f"🎉 YOLO+SAM2 Corrected Pipeline Completed Successfully!")
    print(f"{'='*60}")
    print(f"Total pipeline time: {pipeline_duration:.2f} seconds")
    print(f"YOLO model: {yolo_model_path}")
    print(f"YOLO predictions: {yolo_predictions_dir}")
    print(f"SAM2 model: {args.sam2_model}")
    print(f"Results saved to: {args.results_dir}")
    print(f"Evaluation outputs: {args.eval_dir}")
    print(f"\nNext steps:")
    print(f"1. Check results in: {args.results_dir}")
    print(f"2. Review evaluation outputs in: {args.eval_dir}")
    print(f"3. Analyze performance metrics")
    print(f"4. Generate visualizations if needed")

if __name__ == "__main__":
    main()
