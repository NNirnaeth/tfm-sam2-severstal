#!/usr/bin/env python3
"""
Example usage of the YOLO training pipeline with proper origin handling

This example shows how to:
1. Convert annotations to YOLO format with proper origin handling
2. Train YOLO models on the converted datasets

Steps:
1. First run conversion: python convert_to_yolo_format.py --all-subsets --output-dir datasets/yolo_segmentation_subsets
2. Then run training: python train_yolov8_preconverted.py --all-subsets --both-lr
"""

import subprocess
import os

def run_conversion():
    """Convert all subsets to YOLO format"""
    print("Step 1: Converting annotations to YOLO format...")
    cmd = [
        "python", "new_src/training/convert_to_yolo_format.py",
        "--all-subsets",
        "--output-dir", "datasets/yolo_segmentation_subsets"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Conversion completed successfully")
    else:
        print("✗ Conversion failed")
        print(f"Error: {result.stderr}")
        return False
    return True

def run_training():
    """Train YOLO models on converted datasets"""
    print("Step 2: Training YOLO models...")
    cmd = [
        "python", "new_src/training/train_yolov8_preconverted.py",
        "--all-subsets",
        "--both-lr",
        "--converted-dir", "datasets/yolo_segmentation_subsets"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Training completed successfully")
    else:
        print("✗ Training failed")
        print(f"Error: {result.stderr}")
        return False
    return True

def main():
    print("YOLO Segmentation Training Pipeline")
    print("=" * 50)
    
    # Step 1: Convert annotations
    if not run_conversion():
        return
    
    # Step 2: Train models
    if not run_training():
        return
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()

