#!/usr/bin/env python3
"""
Prepare YOLO Detection Dataset from Severstal Bitmap Annotations
This script converts bitmap annotations to YOLO format bounding boxes

Input: Supervisely format JSON with bitmap annotations from Severstal dataset
Output: YOLO detection dataset structure with images/ and labels/ directories

Hardcoded paths for Severstal dataset:
- Annotations: datasets/Data/splits/
- Images: datasets/Data/splits/ (individual image files)
- Output: data/severstal_det/

Usage:
    python prepare_yolo_detect_dataset.py
"""

import os
import sys
import json
import cv2
import base64
import zlib
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path
import numpy as np

# Hardcoded paths for Severstal dataset
SEVERSTAL_SPLITS_DIR = "datasets/Data/splits"
OUTPUT_DATASET_DIR = "datasets/yolo_detectionion"

def convert_bitmap_to_bbox(bitmap_data, image_width, image_height):
    """Convert bitmap annotation to bounding box coordinates in YOLO format"""
    try:
        # Decode bitmap from base64 and zlib
        decoded_data = base64.b64decode(bitmap_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Convert to PIL Image
        bitmap_image = Image.open(BytesIO(decompressed_data))
        bitmap_array = np.array(bitmap_image)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(bitmap_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (x + w/2) / image_width
            y_center = (y + h/2) / image_height
            width = w / image_width
            height = h / image_height
            
            # Ensure coordinates are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            bboxes.append([x_center, y_center, width, height])
        
        return bboxes
    except Exception as e:
        print(f"Error converting bitmap to bbox: {e}")
        return []


def prepare_detection_dataset(split_name, output_path):
    """Prepare detection dataset by converting bitmap annotations to YOLO format"""
    print(f"Preparing {split_name} detection dataset...")
    
    # Create output directories
    os.makedirs(os.path.join(output_path, "images", split_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", split_name), exist_ok=True)
    
    # Paths for this split
    split_ann_dir = os.path.join(SEVERSTAL_SPLITS_DIR, split_name, "ann")
    split_img_dir = os.path.join(SEVERSTAL_SPLITS_DIR, split_name, "img")
    
    if not os.path.exists(split_ann_dir):
        print(f" Annotations directory not found: {split_ann_dir}")
        return 0
    
    if not os.path.exists(split_img_dir):
        print(f" Images directory not found: {split_img_dir}")
        return 0
    
    # Get all annotation files
    ann_files = [f for f in os.listdir(split_ann_dir) if f.endswith('.json')]
    print(f"Found {len(ann_files)} annotation files in {split_name}")
    
    processed_count = 0
    for ann_file in tqdm(ann_files, desc=f"Processing {split_name}"):
        # Extract image name from annotation filename
        image_name = ann_file.replace('.json', '')
        image_path = os.path.join(split_img_dir, image_name)
        ann_path = os.path.join(split_ann_dir, ann_file)
        
        if not os.path.exists(image_path):
            continue
        
        # Read annotation file
        try:
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
        except Exception as e:
            print(f"Error reading annotation {ann_path}: {e}")
            continue
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue
        
        # Convert annotations to bboxes
        bboxes = []
        for obj in annotation.get('objects', []):
            if 'bitmap' in obj:
                bitmap_bboxes = convert_bitmap_to_bbox(
                    obj['bitmap']['data'],
                    img_width,
                    img_height
                )
                bboxes.extend(bitmap_bboxes)
        
        if bboxes:
            # Save image (create symlink to avoid duplicating files)
            output_image_path = os.path.join(output_path, "images", split_name, image_name)
            if not os.path.exists(output_image_path):
                try:
                    os.symlink(os.path.abspath(image_path), output_image_path)
                except OSError:
                    # If symlink fails, copy the file
                    import shutil
                    shutil.copy2(image_path, output_image_path)
            
            # Save labels in YOLO format
            # Remove .jpg extension to get correct label filename
            label_name = image_name.replace('.jpg', '') + '.txt'
            label_path = os.path.join(output_path, "labels", split_name, label_name)
            
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    # YOLO format: class_id x_center y_center width height
                    f.write(f"0 {' '.join([f'{coord:.6f}' for coord in bbox])}\n")
            
            processed_count += 1
    
    print(f"Processed {processed_count} images for {split_name}")
    return processed_count


def create_detection_yaml(output_path, dataset_path):
    """Create YOLO detection dataset YAML file"""
    yaml_content = {
        'path': dataset_path,
        'train': 'images/train_split',
        'val': 'images/val_split',
        'test': 'images/test_split',
        'names': ['defect']
    }
    
    yaml_path = os.path.join(output_path, "yolo_detection.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset YAML: {yaml_path}")
    return yaml_path


def main():
    print(f"Preparing YOLO detection dataset from Severstal annotations...")
    print(f"Splits directory: {SEVERSTAL_SPLITS_DIR}")
    print(f"Output directory: {OUTPUT_DATASET_DIR}")
    
    # Check if source directories exist
    if not os.path.exists(SEVERSTAL_SPLITS_DIR):
        print(f" Error: Splits directory not found: {SEVERSTAL_SPLITS_DIR}")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
    
    # Prepare train split
    print(f"\nProcessing train split...")
    train_count = prepare_detection_dataset('train_split', OUTPUT_DATASET_DIR)
    
    # Prepare val split
    print(f"\nProcessing validation split...")
    val_count = prepare_detection_dataset('val_split', OUTPUT_DATASET_DIR)
    
    # Prepare test split
    print(f"\nProcessing test split...")
    test_count = prepare_detection_dataset('test_split', OUTPUT_DATASET_DIR)
    
    # Create dataset YAML
    yaml_path = create_detection_yaml(OUTPUT_DATASET_DIR, os.path.abspath(OUTPUT_DATASET_DIR))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset preparation completed!")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DATASET_DIR}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Images processed:")
    print(f"  Train: {train_count}")
    print(f"  Validation: {val_count}")
    print(f"  Test: {test_count}")
    print(f"  Total: {train_count + val_count + test_count}")
    
    # Show directory structure
    print(f"\nDataset structure created:")
    print(f"{OUTPUT_DATASET_DIR}/")
    print(f"├── images/")
    
    # Count files safely
    try:
        train_files = len(os.listdir(os.path.join(OUTPUT_DATASET_DIR, 'images', 'train_split')))
        val_files = len(os.listdir(os.path.join(OUTPUT_DATASET_DIR, 'images', 'val_split')))
        test_files = len(os.listdir(os.path.join(OUTPUT_DATASET_DIR, 'images', 'test_split')))
        print(f"│   ├── train_split/ ({train_files} files)")
        print(f"│   ├── val_split/   ({val_files} files)")
        print(f"│   └── test_split/  ({test_files} files)")
    except Exception as e:
        print(f"│   ├── train_split/")
        print(f"│   ├── val_split/")
        print(f"│   └── test_split/")
    
    print(f"├── labels/")
    try:
        train_labels = len(os.listdir(os.path.join(OUTPUT_DATASET_DIR, 'labels', 'train_split')))
        val_labels = len(os.listdir(os.path.join(OUTPUT_DATASET_DIR, 'labels', 'val_split')))
        test_labels = len(os.listdir(os.path.join(OUTPUT_DATASET_DIR, 'labels', 'test_split')))
        print(f"│   ├── train_split/ ({train_labels} files)")
        print(f"│   ├── val_split/   ({val_labels} files)")
        print(f"│   └── test_split/  ({test_labels} files)")
    except Exception as e:
        print(f"│   ├── train_split/")
        print(f"│   ├── val_split/")
        print(f"│   └── test_split/")
    
            print(f"└── yolo_detection.yaml")
    
    print(f"\nNext steps:")
    print(f"1. Verify the dataset structure")
    print(f"2. Run training with: python new_src/training/train_yolo_detect_full_dataset.py --lr 1e-4 --epochs 100")


if __name__ == "__main__":
    main()
