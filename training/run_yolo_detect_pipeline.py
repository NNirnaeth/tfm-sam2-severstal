#!/usr/bin/env python3
"""
Complete YOLO Detection Pipeline for Severstal Dataset
This script demonstrates the complete workflow:

1. Prepare detection dataset from bitmap annotations
2. Train YOLO detection model
3. Run inference on test split
4. Generate bounding boxes for SAM2 refinement

Usage:
    python run_yolo_detect_pipeline.py --prepare_dataset --train --inference
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete YOLO Detection Pipeline')
    parser.add_argument('--prepare_dataset', action='store_true',
                       help='Prepare detection dataset from bitmap annotations')
    parser.add_argument('--train', action='store_true',
                       help='Train YOLO detection model')
    parser.add_argument('--inference', action='store_true',
                       help='Run inference on test split')
    parser.add_argument('--lr', type=float, default=1e-4,
                       choices=[1e-3, 1e-4],
                       help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model for inference (required if --inference)')
    
    args = parser.parse_args()
    
    if not any([args.prepare_dataset, args.train, args.inference]):
        print("Please specify at least one action: --prepare_dataset, --train, or --inference")
        return
    
    # Step 1: Prepare detection dataset
    if args.prepare_dataset:
        print("\n Step 1: Preparing detection dataset...")
        cmd = [
            "python", "new_src/experiments/prepare_yolo_detect_dataset.py"
        ]
        
        if not run_command(cmd, "Dataset preparation"):
            print(" Dataset preparation failed. Exiting.")
            return
    
    # Step 2: Train YOLO model
    if args.train:
        print("\n Step 2: Training YOLO detection model...")
        cmd = [
            "python", "new_src/training/train_yolo_detect_full_dataset.py",
            "--lr", str(args.lr),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--save_dir", "new_src/training/training_results/yolo_detection"
        ]
        
        if not run_command(cmd, "YOLO training"):
            print(" Training failed. Exiting.")
            return
        
        print("\n Training completed! Check the results directory for the trained model.")
        print("Note: The best model will be saved as 'best.pt' in the runs_tfm directory.")
    
    # Step 3: Run inference on test split
    if args.inference:
        if not args.model_path:
            print(" Error: --model_path is required for inference")
            print("Please provide the path to your trained YOLO model (.pt file)")
            return
        
        if not os.path.exists(args.model_path):
            print(f" Error: Model file not found: {args.model_path}")
            return
        
        print("\n Step 3: Running inference on test split...")
        cmd = [
            "python", "new_src/evaluation/eval_yolo_detect.py",
            "--model_path", args.model_path,
                            "--test_images_path", "datasets/yolo_detectionion/images/test_split",
            "--gt_annotations_path", "datasets/Data/splits/test_split",
            "--images_path", "datasets/Data/splits",
            "--output_path", "new_src/training/training_results/yolo_detection"
        ]
        
        if not run_command(cmd, "YOLO inference"):
            print(" Inference failed. Exiting.")
            return
        
        print("\n Inference completed!")
        print("Bounding box predictions have been saved for SAM2 refinement.")
        print("Check the output directory for the predictions JSON file.")
    
    print(f"\n Pipeline completed successfully!")
    
    if args.inference:
        print(f"\nNext steps:")
        print(f"1. Use the generated bounding boxes from: new_src/training/results/yolo_inference_test/yolo_predictions.json")
        print(f"2. Pass these bboxes to SAM2 for mask refinement")
        print(f"3. The bboxes are in format: {{image_name: [[x1, y1, x2, y2, conf], ...]}}")

if __name__ == "__main__":
    main()
