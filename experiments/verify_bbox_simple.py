#!/usr/bin/env python3
"""
Simple Bbox Verification Script
This script loads a few images and prints detailed information about
the bitmap to bbox conversion without requiring matplotlib.

Usage:
    python verify_bbox_simple.py --num_images 2 --split train_split
"""

import os
import sys
import json
import base64
import zlib
from PIL import Image, ImageDraw
from io import BytesIO
import argparse
import numpy as np
from pathlib import Path

# Hardcoded paths
SEVERSTAL_SPLITS_DIR = "datasets/Data/splits"
YOLO_DATASET_DIR = "datasets/yolo_detection_fixed"

def convert_bitmap_to_bbox(bitmap_data, image_width, image_height, min_area=10):
    """Convert bitmap to YOLO bbox (same as in prepare script)"""
    try:
        # Decode bitmap from base64 and zlib
        decoded_data = base64.b64decode(bitmap_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Convert to PIL Image
        bitmap_image = Image.open(BytesIO(decompressed_data))
        
        # Convert to numpy array
        bitmap_array = np.array(bitmap_image)
        
        # Handle different image formats
        if bitmap_array.ndim == 3:
            if bitmap_array.shape[2] == 4:  # RGBA
                bitmap_array = bitmap_array[:, :, 3]  # Use alpha channel
            elif bitmap_array.shape[2] == 3:  # RGB
                bitmap_array = bitmap_array[:, :, 0]  # Use first channel
        
        # Ensure binary mask
        bitmap_array = (bitmap_array > 0).astype(np.uint8)
        
        # Find non-zero coordinates using numpy
        non_zero_coords = np.where(bitmap_array > 0)
        if len(non_zero_coords[0]) == 0:
            return []
        
        # Get bounding box coordinates from non-zero pixels
        min_y, max_y = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
        min_x, max_x = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
        
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
        
        return [[x_center, y_center, width, height, min_x, min_y, max_x, max_y, area]]
        
    except Exception as e:
        print(f"Error converting bitmap to bbox: {e}")
        return []

def verify_conversion_simple(split_name, num_images=2):
    """Simple verification without matplotlib"""
    print(f"Simple verification for {split_name} split...")
    
    # Paths
    split_ann_dir = os.path.join(SEVERSTAL_SPLITS_DIR, split_name, "ann")
    split_img_dir = os.path.join(SEVERSTAL_SPLITS_DIR, split_name, "img")
    yolo_labels_dir = os.path.join(YOLO_DATASET_DIR, "labels", split_name)
    
    if not os.path.exists(split_ann_dir):
        print(f"Error: Annotations directory not found: {split_ann_dir}")
        return
    
    if not os.path.exists(yolo_labels_dir):
        print(f"Error: YOLO labels directory not found: {yolo_labels_dir}")
        print("Run prepare_yolo_detect_dataset_fixed.py first!")
        return
    
    # Get annotation files
    ann_files = [f for f in os.listdir(split_ann_dir) if f.endswith('.json')]
    ann_files = ann_files[:num_images]  # Limit to requested number
    
    for i, ann_file in enumerate(ann_files):
        print(f"\n{'='*80}")
        print(f"Image {i+1}/{len(ann_files)}: {ann_file}")
        print(f"{'='*80}")
        
        # Extract image name
        image_name = ann_file.replace('.json', '')
        image_path = os.path.join(split_img_dir, image_name)
        ann_path = os.path.join(split_ann_dir, ann_file)
        label_name = image_name.replace('.jpg', '') + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        if not os.path.exists(label_path):
            print(f"YOLO label not found: {label_path}")
            continue
        
        # Load image
        try:
            image = Image.open(image_path)
            img_width, img_height = image.size
            print(f"Image size: {img_width}x{img_height}")
        except Exception as e:
            print(f"Error loading image: {e}")
            continue
        
        # Load original annotation
        try:
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
        except Exception as e:
            print(f"Error loading annotation: {e}")
            continue
        
        # Load YOLO labels
        try:
            with open(label_path, 'r') as f:
                yolo_lines = f.readlines()
            yolo_bboxes = []
            for line in yolo_lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    bbox = [float(x) for x in parts[1:5]]  # Skip class_id
                    yolo_bboxes.append(bbox)
        except Exception as e:
            print(f"Error loading YOLO labels: {e}")
            continue
        
        # Process bitmap annotations
        converted_bboxes = []
        bitmap_count = 0
        
        print(f"\nProcessing bitmap annotations:")
        for obj_idx, obj in enumerate(annotation.get('objects', [])):
            if 'bitmap' in obj:
                bitmap_count += 1
                print(f"  Bitmap {bitmap_count}:")
                
                # Convert to bbox
                bboxes = convert_bitmap_to_bbox(
                    obj['bitmap']['data'],
                    img_width,
                    img_height
                )
                
                for bbox_idx, bbox in enumerate(bboxes):
                    x_center, y_center, width, height, min_x, min_y, max_x, max_y, area = bbox
                    
                    print(f"    Bbox {bbox_idx + 1}:")
                    print(f"      Pixel coordinates: ({min_x}, {min_y}) -> ({max_x}, {max_y})")
                    print(f"      Size: {max_x - min_x + 1} x {max_y - min_y + 1}")
                    print(f"      Area: {area} pixels")
                    print(f"      YOLO normalized: center=({x_center:.6f}, {y_center:.6f}), size=({width:.6f}, {height:.6f})")
                    
                    converted_bboxes.append([x_center, y_center, width, height])
        
        print(f"\nSummary:")
        print(f"  Total bitmaps found: {bitmap_count}")
        print(f"  Total bboxes converted: {len(converted_bboxes)}")
        print(f"  YOLO labels in file: {len(yolo_bboxes)}")
        
        # Compare with YOLO labels
        print(f"\nYOLO labels comparison:")
        for j, yolo_bbox in enumerate(yolo_bboxes):
            x_center, y_center, width, height = yolo_bbox
            print(f"  YOLO bbox {j+1}: center=({x_center:.6f}, {y_center:.6f}), size=({width:.6f}, {height:.6f})")
        
        # Check if conversion matches YOLO labels
        if len(converted_bboxes) == len(yolo_bboxes):
            print(f"\n✓ Number of bboxes matches!")
            # Check if coordinates match (with small tolerance)
            matches = True
            for j, (conv_bbox, yolo_bbox) in enumerate(zip(converted_bboxes, yolo_bboxes)):
                for k in range(4):
                    if abs(conv_bbox[k] - yolo_bbox[k]) > 1e-6:
                        matches = False
                        break
                if not matches:
                    break
            
            if matches:
                print(f"✓ All coordinates match perfectly!")
            else:
                print(f"⚠ Coordinates don't match exactly")
        else:
            print(f"⚠ Number of bboxes doesn't match: {len(converted_bboxes)} vs {len(yolo_bboxes)}")
        
        # Wait for user input
        input(f"\nPress Enter to continue to next image...")

def main():
    parser = argparse.ArgumentParser(description='Simple bbox verification')
    parser.add_argument('--num_images', type=int, default=2,
                       help='Number of images to verify (default: 2)')
    parser.add_argument('--split', type=str, default='train_split',
                       choices=['train_split', 'val_split', 'test_split'],
                       help='Dataset split to verify (default: train_split)')
    
    args = parser.parse_args()
    
    print(f"Simple Bitmap to Bbox Verification")
    print(f"Split: {args.split}")
    print(f"Number of images: {args.num_images}")
    print(f"Source annotations: {SEVERSTAL_SPLITS_DIR}")
    print(f"YOLO dataset: {YOLO_DATASET_DIR}")
    
    # Check if directories exist
    if not os.path.exists(SEVERSTAL_SPLITS_DIR):
        print(f"Error: Source directory not found: {SEVERSTAL_SPLITS_DIR}")
        return
    
    if not os.path.exists(YOLO_DATASET_DIR):
        print(f"Error: YOLO dataset directory not found: {YOLO_DATASET_DIR}")
        print("Run prepare_yolo_detect_dataset_fixed.py first!")
        return
    
    verify_conversion_simple(args.split, args.num_images)

if __name__ == "__main__":
    main()
