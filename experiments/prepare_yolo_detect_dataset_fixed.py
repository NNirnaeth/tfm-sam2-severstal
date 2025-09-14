#!/usr/bin/env python3
"""
Prepare YOLO Detection Dataset from Severstal Bitmap Annotations - CORRECTED VERSION
This script converts bitmap annotations to YOLO format bounding boxes

CORRECTED: Now properly uses the 'origin' field from Supervisely bitmap format
- Fixes critical bug where bitmaps were positioned incorrectly  
- Uses cv2 for PNG decoding (same as working segmentation scripts)
- Properly positions bitmap fragments within the full image using origin coordinates

Input: Supervisely format JSON with bitmap annotations from Severstal dataset
Output: YOLO detection dataset structure with images/ and labels/ directories

Hardcoded paths for Severstal dataset:
- Annotations: datasets/Data/splits/
- Images: datasets/Data/splits/ (individual image files)
- Output: datasets/yolo_detection_fixed/ (full dataset)
- Output subsets: datasets/yolo_detection_fixed/subset_XXX/ (subsets)

Usage:
    python prepare_yolo_detect_dataset_fixed.py --min_area 10 --subsets 500,1000,2000
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
import random
import shutil

# Hardcoded paths for Severstal dataset
SEVERSTAL_SPLITS_DIR = "datasets/Data/splits"
OUTPUT_DATASET_DIR = "datasets/yolo_detection_fixed"

def convert_bitmap_to_bbox(bitmap_data, origin, image_width, image_height, min_area=10):
    """
    CORRECT conversion from Supervisely bitmap to YOLO bboxes
    - Uses the origin field to position bitmap correctly in full image
    - Decodes PNG bitmap using cv2 (more robust than PIL for this format)
    - Filters tiny boxes by area threshold
    """
    try:
        origin_x, origin_y = origin[0], origin[1]
        
        # Decode bitmap from base64 and zlib
        decoded_data = base64.b64decode(bitmap_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Use cv2 to decode PNG (same as working scripts)
        nparr = np.frombuffer(decompressed_data, np.uint8)
        bitmap_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if bitmap_img is None:
            print("    Warning: Failed to decode bitmap with cv2")
            return []
        
        # Handle different image formats
        if len(bitmap_img.shape) == 3:
            bitmap_img = bitmap_img[:, :, 0]  # Take first channel
        
        # Ensure binary mask
        bitmap_mask = (bitmap_img > 0).astype(np.uint8)
        
        # Check if mask has any pixels
        if np.sum(bitmap_mask) == 0:
            return []
        
        # Calculate actual position in full image using origin
        mask_height, mask_width = bitmap_mask.shape
        end_y = origin_y + mask_height
        end_x = origin_x + mask_width
        
        # Ensure coordinates are within image bounds
        end_y = min(end_y, image_height)
        end_x = min(end_x, image_width)
        
        # Calculate bounding box in full image coordinates
        min_x, max_x = origin_x, end_x - 1
        min_y, max_y = origin_y, end_y - 1
        
        # Calculate area and filter tiny boxes
        area = (max_x - min_x + 1) * (max_y - min_y + 1)
        if area < min_area:
            return []
        
        # Convert to YOLO format (normalized coordinates)
        x_center = (min_x + max_x) / 2.0 / image_width
        y_center = (min_y + max_y) / 2.0 / image_height
        width = (max_x - min_x + 1) / image_width
        height = (max_y - min_y + 1) / image_height
        
        # Ensure coordinates are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return [[x_center, y_center, width, height]]
        
    except Exception as e:
        print(f"Error converting bitmap to bbox: {e}")
        import traceback
        traceback.print_exc()
        return []

def prepare_detection_dataset(split_name, output_path, selected_images=None):
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
        return 0, []
    
    if not os.path.exists(split_img_dir):
        print(f" Images directory not found: {split_img_dir}")
        return 0, []
    
    # Get all annotation files
    ann_files = [f for f in os.listdir(split_ann_dir) if f.endswith('.json')]
    
    # Filter by selected images if provided
    if selected_images is not None:
        selected_set = set(selected_images)
        # Convert annotation filenames to match selected image names
        ann_files = [f for f in ann_files if f.replace('.json', '').replace('.jpg', '') in selected_set]
    
    print(f"Found {len(ann_files)} annotation files in {split_name}")
    
    processed_count = 0
    processed_images = []
    
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
        
        # Convert annotations to bboxes using corrected method
        bboxes = []
        for obj in annotation.get('objects', []):
            if 'bitmap' in obj:
                bitmap_bboxes = convert_bitmap_to_bbox(
                    obj['bitmap']['data'],
                    obj['bitmap']['origin'],
                    img_width,
                    img_height
                )
                bboxes.extend(bitmap_bboxes)
        
        # Save image (create symlink to avoid duplicating files)
        output_image_path = os.path.join(output_path, "images", split_name, image_name)
        if not os.path.exists(output_image_path):
            try:
                os.symlink(os.path.abspath(image_path), output_image_path)
            except OSError:
                # If symlink fails, copy the file
                shutil.copy2(image_path, output_image_path)
        
        # Save labels in YOLO format (even if no bboxes - create empty file)
        label_name = image_name.replace('.jpg', '') + '.txt'
        label_path = os.path.join(output_path, "labels", split_name, label_name)
        
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                # YOLO format: class_id x_center y_center width height
                f.write(f"0 {' '.join([f'{coord:.6f}' for coord in bbox])}\n")
        
        processed_count += 1
        processed_images.append(image_name.replace('.jpg', ''))
    
    print(f"Processed {processed_count} images for {split_name}")
    return processed_count, processed_images


def get_train_images_list(split_name="train_split"):
    """Get list of all training images for subset creation"""
    split_ann_dir = os.path.join(SEVERSTAL_SPLITS_DIR, split_name, "ann")
    
    if not os.path.exists(split_ann_dir):
        print(f"Error: Annotations directory not found: {split_ann_dir}")
        return []
    
    # Get all annotation files
    ann_files = [f for f in os.listdir(split_ann_dir) if f.endswith('.json')]
    
    # Extract image names (without .json extension)
    image_names = [f.replace('.json', '').replace('.jpg', '') for f in ann_files]
    
    return image_names


def create_subset_dataset(subset_size, base_output_dir):
    """Create a subset dataset with specified number of training images"""
    print(f"\nCreating subset dataset with {subset_size} training images...")
    
    # Get all training images
    all_train_images = get_train_images_list("train_split")
    
    if len(all_train_images) < subset_size:
        print(f"Warning: Requested {subset_size} images but only {len(all_train_images)} available")
        subset_size = len(all_train_images)
    
    # Set seed for reproducible subsets
    random.seed(42)
    
    # Randomly select subset of training images
    selected_train_images = random.sample(all_train_images, subset_size)
    
    # Create subset directory
    subset_dir = os.path.join(base_output_dir, f"subset_{subset_size}")
    os.makedirs(subset_dir, exist_ok=True)
    
    print(f"Selected {len(selected_train_images)} training images for subset")
    
    # Process train split with selected images
    train_count, _ = prepare_detection_dataset("train_split", subset_dir, selected_train_images)
    
    # Process val and test splits completely (use all images)
    val_count, _ = prepare_detection_dataset("val_split", subset_dir)
    test_count, _ = prepare_detection_dataset("test_split", subset_dir)
    
    # Create dataset YAML for subset
    yaml_path = create_detection_yaml_subset(subset_dir, subset_size)
    
    print(f"Subset {subset_size} created successfully:")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images") 
    print(f"  Test: {test_count} images")
    print(f"  YAML: {yaml_path}")
    
    return subset_dir, yaml_path


def create_detection_yaml_subset(output_path, subset_size):
    """Create YOLO detection dataset YAML file for subset"""
    yaml_content = {
        'path': os.path.abspath(output_path),
        'train': 'images/train_split',
        'val': 'images/val_split', 
        'test': 'images/test_split',
        'names': ['defect']
    }
    
    yaml_path = os.path.join(output_path, f"yolo_detection_{subset_size}.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return yaml_path

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
    parser = argparse.ArgumentParser(description='Prepare YOLO detection dataset - CORRECTED VERSION')
    parser.add_argument('--min_area', type=int, default=10,
                       help='Minimum area threshold for bounding boxes (default: 10)')
    parser.add_argument('--subsets', type=str, default=None,
                       help='Comma-separated list of subset sizes to create (e.g., "500,1000,2000")')
    parser.add_argument('--full-dataset', action='store_true', default=True,
                       help='Create full dataset (default: True)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output for first few images')
    
    args = parser.parse_args()
    
    print(f"Preparing YOLO detection dataset from Severstal annotations - CORRECTED VERSION")
    print(f"Splits directory: {SEVERSTAL_SPLITS_DIR}")
    print(f"Output directory: {OUTPUT_DATASET_DIR}")
    print(f"Minimum area threshold: {args.min_area}")
    print(f"Debug mode: {args.debug}")
    
    # Check if source directories exist
    if not os.path.exists(SEVERSTAL_SPLITS_DIR):
        print(f" Error: Splits directory not found: {SEVERSTAL_SPLITS_DIR}")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
    
    # Prepare full dataset if requested
    if args.full_dataset:
        print(f"\n{'='*60}")
        print(f"PREPARING FULL DATASET")
        print(f"{'='*60}")
        
        # Prepare train split
        print(f"\nProcessing train split...")
        train_count, _ = prepare_detection_dataset('train_split', OUTPUT_DATASET_DIR)
        
        # Prepare val split
        print(f"\nProcessing validation split...")
        val_count, _ = prepare_detection_dataset('val_split', OUTPUT_DATASET_DIR)
        
        # Prepare test split
        print(f"\nProcessing test split...")
        test_count, _ = prepare_detection_dataset('test_split', OUTPUT_DATASET_DIR)
        
        # Create dataset YAML
        yaml_path = create_detection_yaml(OUTPUT_DATASET_DIR, os.path.abspath(OUTPUT_DATASET_DIR))
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Full dataset preparation completed!")
        print(f"{'='*60}")
        print(f"Output directory: {OUTPUT_DATASET_DIR}")
        print(f"Dataset YAML: {yaml_path}")
        print(f"Images processed:")
        print(f"  Train: {train_count}")
        print(f"  Validation: {val_count}")
        print(f"  Test: {test_count}")
        print(f"  Total: {train_count + val_count + test_count}")
    
    # Prepare subsets if requested
    if args.subsets:
        subset_sizes = [int(x.strip()) for x in args.subsets.split(',')]
        
        print(f"\n{'='*60}")
        print(f"PREPARING SUBSET DATASETS")
        print(f"{'='*60}")
        print(f"Subset sizes: {subset_sizes}")
        
        for subset_size in subset_sizes:
            subset_dir, yaml_path = create_subset_dataset(subset_size, OUTPUT_DATASET_DIR)
    
    print(f"\n{'='*60}")
    print(f"ALL DATASETS PREPARED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Verify the dataset structure")
    print(f"2. Test with debug script to ensure correct coordinates")
    print(f"3. Run training with:")
    print(f"   - Full dataset: python new_src/training/train_yolo_detect_corrected.py --epochs 100")
    print(f"   - Subsets: python new_src/training/train_yolo_detect_corrected.py --subset-size 500 --epochs 100")
    print(f"\nKey features:")
    print(f"- FIXED: Now properly uses 'origin' field to position bitmaps correctly")
    print(f"- Uses cv2 for PNG decoding (same as working segmentation scripts)")  
    print(f"- Area-based filtering to eliminate tiny bounding boxes")
    print(f"- Correct bounding box extraction for defect detection")
    print(f"- Support for subset creation with reproducible random sampling")

if __name__ == "__main__":
    main()
