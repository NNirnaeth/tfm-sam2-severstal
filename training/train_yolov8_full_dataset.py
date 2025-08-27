#!/usr/bin/env python3
"""
Train YOLOv8 Segmentation models on full Severstal dataset
Training data: datasets/Data/splits/train_split (4200 images)
Validation data: datasets/Data/splits/val_split (466 images)

Binary segmentation: defect vs background (unified classes)
Image size: 1024x256 (maintaining aspect ratio)

This script executes YOLO training commands for:
- YOLO-Seg lr=5e-4 (primary)
- YOLO-Seg lr=1e-4 (comparison)

TFM Requirements:
- AdamW optimizer with weight_decay=1e-4
- Compare lr = 5e-4 vs 1e-4
- Single class segmentation
- Save best model by validation metrics
- Comprehensive logging and metrics
- Cosine learning rate scheduling
- Extended training (150 epochs, patience=30)
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
import tempfile
from datetime import datetime
import time
from pathlib import Path
import numpy as np
from PIL import Image
import base64
import zlib
import cv2
import io

def decode_bitmap_to_mask(bitmap_data):
    """Decode PNG bitmap data from Supervisely format"""
    try:
        # Decode base64 and decompress PNG
        decoded_data = base64.b64decode(bitmap_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Open as PIL Image and convert to numpy array
        mask = Image.open(io.BytesIO(decompressed_data))
        mask_array = np.array(mask)
        
        # Convert to binary mask (0 or 1)
        binary_mask = (mask_array > 0).astype(np.uint8)
        return binary_mask
    except Exception as e:
        print(f"Error decoding bitmap: {e}")
        return None

def mask_to_yolo_seg(mask, img_width, img_height):
    """Convert binary mask to YOLO segmentation format"""
    try:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_segments = []
        for contour in contours:
            # Simplify contour and convert to normalized coordinates
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Flatten and normalize coordinates
            coords = approx.reshape(-1, 2)
            normalized_coords = []
            
            for x, y in coords:
                norm_x = x / img_width
                norm_y = y / img_height
                normalized_coords.extend([norm_x, norm_y])
            
            yolo_segments.append(normalized_coords)
        
        return yolo_segments
    except Exception as e:
        print(f"Error converting mask to YOLO format: {e}")
        return []

def convert_annotations_to_yolo(input_dir, output_dir):
    """Convert .jpg.json annotations to YOLO format"""
    print(f"Converting annotations from {input_dir} to {output_dir}")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    
    # Process train split
    train_img_dir = os.path.join(input_dir, "train_split", "img")
    train_ann_dir = os.path.join(input_dir, "train_split", "ann")
    
    train_images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    train_converted = 0
    
    for img_file in train_images:
        print(f"Converting train annotation: {img_file}")
        img_path = os.path.join(train_img_dir, img_file)
        ann_file = img_file + '.json'
        ann_path = os.path.join(train_ann_dir, ann_file)
        
        if os.path.exists(ann_path):
            # Copy image
            dst_img = os.path.join(output_dir, "images", "train", img_file)
            shutil.copy2(img_path, dst_img)
            
            # Convert annotation
            try:
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                
                # Get image dimensions
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                # Process bitmap objects
                yolo_lines = []
                for obj in ann_data.get('objects', []):
                    if obj.get('geometryType') == 'bitmap':
                        bitmap_data = obj.get('bitmap', {}).get('data')
                        if bitmap_data:
                            mask = decode_bitmap_to_mask(bitmap_data)
                            if mask is not None:
                                segments = mask_to_yolo_seg(mask, img_width, img_height)
                                for segment in segments:
                                    # YOLO format: class_id + normalized coordinates
                                    line = f"0 {' '.join([f'{coord:.6f}' for coord in segment])}"
                                    yolo_lines.append(line)
                
                # Write YOLO label file
                if yolo_lines:
                    label_file = img_file.replace('.jpg', '.txt')
                    label_path = os.path.join(output_dir, "labels", "train", label_file)
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    train_converted += 1
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
    
    # Process validation split
    val_img_dir = os.path.join(input_dir, "val_split", "img")
    val_ann_dir = os.path.join(input_dir, "val_split", "ann")
    
    val_images = [f for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    val_converted = 0
    
    for img_file in val_images:
        print(f"Converting validation annotation: {img_file}")
        img_path = os.path.join(val_img_dir, img_file)
        ann_file = img_file + '.json'
        ann_path = os.path.join(val_ann_dir, ann_file)
        
        if os.path.exists(ann_path):
            # Copy image
            dst_img = os.path.join(output_dir, "images", "val", img_file)
            shutil.copy2(img_path, dst_img)
            
            # Convert annotation
            try:
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                
                # Get image dimensions
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                # Process bitmap objects
                yolo_lines = []
                for obj in ann_data.get('objects', []):
                    if obj.get('geometryType') == 'bitmap':
                        bitmap_data = obj.get('bitmap', {}).get('data')
                        if bitmap_data:
                            mask = decode_bitmap_to_mask(bitmap_data)
                            if mask is not None:
                                segments = mask_to_yolo_seg(mask, img_width, img_height)
                                for segment in segments:
                                    # YOLO format: class_id + normalized coordinates
                                    line = f"0 {' '.join([f'{coord:.6f}' for coord in segment])}"
                                    yolo_lines.append(line)
                
                # Write YOLO label file
                if yolo_lines:
                    label_file = img_file.replace('.jpg', '.txt')
                    label_path = os.path.join(output_dir, "labels", "val", label_file)
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    val_converted += 1
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
    
    # Create YAML configuration
    yaml_content = f"""path: {os.path.abspath(output_dir)}  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Val images (relative to 'path')

nc: 1  # Number of classes
names: ['defect']  # Class names
"""
    
    yaml_path = os.path.join(output_dir, 'yolo_segmentation.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nConversion completed!")
    print(f"Training: {len(train_images)} images, {train_converted} annotations")
    print(f"Validation: {len(val_images)} images, {val_converted} annotations")
    print(f"Output directory: {output_dir}")
    print(f"YAML config: {yaml_path}")
    
    return output_dir

def run_yolo_training(lr, project_dir="new_src/training/training_results/yolo_segmentation", model_size="s", dataset_yaml=None):
    """
    Execute YOLO training command with specified parameters
    
    Args:
        lr (float): Learning rate (1e-3 or 1e-4)
        project_dir (str): Project directory for saving results
        model_size (str): YOLO model size (s, m, l, x)
        dataset_yaml (str): Path to dataset YAML file
    """
    
    # Format learning rate for naming (consistent with existing models)
    if lr == 0.001:
        lr_str = "lr1e-3"
    elif lr == 0.0005:
        lr_str = "lr5e-4"
    else:
        lr_str = "lr1e-4"
    
    # Construct YOLO command
    cmd = [
        "yolo", "task=segment", "mode=train",
        f"model=yolov8{model_size}-seg.pt",
        f"data={dataset_yaml}",
        "epochs=150", "batch=8", "imgsz=1024", "rect=True", "seed=42",
        "patience=30", "single_cls=True",
        "optimizer=AdamW", "weight_decay=0.0001",
        f"lr0={lr}", "lrf=0.01", "cos_lr=True",
        "mosaic=0.0", "mixup=0.0", "degrees=10.0", "translate=0.10",
        "scale=0.10", "shear=0.0", "fliplr=0.5", "flipud=0.0",
        "hsv_h=0.0", "hsv_s=0.0", "hsv_v=0.0",
        f"project={project_dir}",
        f"name=yolov8{model_size}_seg_{lr_str}"
    ]
    
    print(f"Executing YOLO training with lr={lr}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        # Execute YOLO command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Training completed successfully for lr={lr}")
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed for lr={lr}")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation models")
    parser.add_argument("--lr", type=float, choices=[0.001, 0.0005, 0.0001], 
                       help="Learning rate to use (1e-3, 5e-4, or 1e-4)")
    parser.add_argument("--both", action="store_true", 
                       help="Train both learning rates sequentially")
    parser.add_argument("--project", type=str, default="new_src/training/training_results/yolo_segmentation",
                       help="Project directory for saving results")
    parser.add_argument("--model-size", type=str, default="s", choices=["s", "m", "l", "x"],
                       help="YOLO model size")
    parser.add_argument("--input-dir", type=str, default="datasets/Data/splits",
                       help="Input directory with .jpg.json annotations")
            parser.add_argument("--output-dir", type=str, default="/home/ptp/sam2/datasets/yolo_segmentation",
                       help="Output directory for YOLO format dataset")
    parser.add_argument("--skip-conversion", action="store_true",
                       help="Skip annotation conversion (use existing YOLO dataset)")
    
    args = parser.parse_args()
    
    # Set default behavior
    if not args.lr and not args.both:
        args.both = True
    
    # Use existing annotations by default
    if not args.skip_conversion:
        args.skip_conversion = True
    
    print("YOLOv8 Segmentation Training Script")
    print("=" * 50)
    print(f"Project directory: {args.project}")
    print(f"Model size: yolov8{args.model_size}-seg")
    print(f"Training mode: {'Both learning rates' if args.both else f'Learning rate {args.lr}'}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    start_time = time.time()
    
    # Convert annotations if needed
    dataset_yaml = None
    if not args.skip_conversion:
        print("\nConverting annotations to YOLO format...")
        # Dependencies already imported at the top
        pass
        
        dataset_yaml = convert_annotations_to_yolo(args.input_dir, args.output_dir)
        dataset_yaml = os.path.join(args.output_dir, 'yolo_segmentation.yaml')
    else:
        # Use existing YOLO dataset
        dataset_yaml = os.path.join(args.output_dir, 'yolo_segmentation.yaml')
        if not os.path.exists(dataset_yaml):
            print(f"Error: Dataset YAML not found at {dataset_yaml}")
            print("Please run without --skip-conversion first")
            return
    
    print(f"\nUsing dataset: {dataset_yaml}")
    
    if args.both:
        # Train both learning rates
        print("\n1. Training YOLO-Seg with lr=5e-4")
        success_5e4 = run_yolo_training(0.0005, args.project, args.model_size, dataset_yaml)
        
        print("\n2. Training YOLO-Seg with lr=1e-4")
        success_1e4 = run_yolo_training(0.0001, args.project, args.model_size, dataset_yaml)
        
        if success_5e4 and success_1e4:
            print("\n✅ Both training runs completed successfully!")
        else:
            print("\n❌ Some training runs failed. Check logs above.")
            
    else:
        # Train single learning rate
        success = run_yolo_training(args.lr, args.project, args.model_size, dataset_yaml)
        if success:
            print(f"\n✅ Training completed successfully for lr={args.lr}")
        else:
            print(f"\n❌ Training failed for lr={args.lr}")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time/3600:.2f} hours")
    print("Training completed!")

if __name__ == "__main__":
    main()
