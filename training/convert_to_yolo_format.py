#!/usr/bin/env python3
"""
Convert Severstal annotations to YOLO segmentation format
Handles bitmap data with proper origin correction for accurate segmentation masks

This script converts .jpg.json annotations (Supervisely format) to YOLO format
with proper handling of bitmap origins to ensure accurate mask positioning.

Usage:
    python convert_to_yolo_format.py --subset 1000 --input-dir datasets/Data/splits --output-dir datasets/yolo_segmentation_subsets
    python convert_to_yolo_format.py --subset full --input-dir datasets/Data/splits --output-dir datasets/yolo_segmentation_full
"""

import os
import sys
import argparse
import json
import shutil
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

def mask_to_yolo_seg(mask, origin_x, origin_y, img_width, img_height):
    """Convert binary mask to YOLO segmentation format with origin correction"""
    try:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_segments = []
        for contour in contours:
            # Skip very small contours (noise)
            if cv2.contourArea(contour) < 10:
                continue
                
            # Simplify contour and convert to normalized coordinates
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Ensure we have enough points for a valid polygon
            if len(approx) < 3:
                continue
            
            # Flatten and normalize coordinates
            coords = approx.reshape(-1, 2)
            normalized_coords = []
            
            for x, y in coords:
                # Apply origin offset to get absolute coordinates in full image
                abs_x = x + origin_x
                abs_y = y + origin_y
                
                # Clamp coordinates to image bounds
                abs_x = max(0, min(abs_x, img_width - 1))
                abs_y = max(0, min(abs_y, img_height - 1))
                
                # Normalize coordinates
                norm_x = abs_x / img_width
                norm_y = abs_y / img_height
                normalized_coords.extend([norm_x, norm_y])
            
            # Only add if we have a valid polygon
            if len(normalized_coords) >= 6:  # At least 3 points (3*2 coordinates)
                yolo_segments.append(normalized_coords)
        
        return yolo_segments
    except Exception as e:
        print(f"Error converting mask to YOLO format: {e}")
        return []

def convert_annotations_to_yolo(input_dir, output_dir, subset_name="full"):
    """Convert .jpg.json annotations to YOLO format with proper origin handling"""
    print(f"Converting annotations from {input_dir} to {output_dir}")
    print(f"Subset: {subset_name}")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    
    # Determine train data source based on subset
    if subset_name == "full":
        train_img_dir = os.path.join(input_dir, "train_split", "img")
        train_ann_dir = os.path.join(input_dir, "train_split", "ann")
    else:
        train_img_dir = os.path.join(input_dir, "subsets", f"{subset_name}_subset", "img")
        train_ann_dir = os.path.join(input_dir, "subsets", f"{subset_name}_subset", "ann")
    
    print(f"Train images: {train_img_dir}")
    print(f"Train annotations: {train_ann_dir}")
    
    # Verify directories exist
    if not os.path.exists(train_img_dir):
        print(f"ERROR: Train image directory not found: {train_img_dir}")
        return None
    if not os.path.exists(train_ann_dir):
        print(f"ERROR: Train annotation directory not found: {train_ann_dir}")
        return None
    
    # Process train split
    train_images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    train_converted = 0
    train_with_annotations = 0
    
    print(f"Found {len(train_images)} training images")
    
    for i, img_file in enumerate(train_images):
        if (i + 1) % 100 == 0:
            print(f"Processing train image {i+1}/{len(train_images)}: {img_file}")
        
        img_path = os.path.join(train_img_dir, img_file)
        ann_file = img_file + '.json'
        ann_path = os.path.join(train_ann_dir, ann_file)
        
        # Copy image
        dst_img = os.path.join(output_dir, "images", "train", img_file)
        shutil.copy2(img_path, dst_img)
        
        # Convert annotation if exists
        yolo_lines = []
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                
                # Get image dimensions
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                # Process bitmap objects
                for obj in ann_data.get('objects', []):
                    if obj.get('geometryType') == 'bitmap':
                        bitmap_data = obj.get('bitmap', {}).get('data')
                        bitmap_origin = obj.get('bitmap', {}).get('origin', [0, 0])
                        
                        if bitmap_data:
                            mask = decode_bitmap_to_mask(bitmap_data)
                            if mask is not None:
                                origin_x, origin_y = bitmap_origin
                                segments = mask_to_yolo_seg(mask, origin_x, origin_y, img_width, img_height)
                                for segment in segments:
                                    # YOLO format: class_id + normalized coordinates
                                    line = f"0 {' '.join([f'{coord:.6f}' for coord in segment])}"
                                    yolo_lines.append(line)
                
                if yolo_lines:
                    train_with_annotations += 1
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        # Write YOLO label file (even if empty)
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(output_dir, "labels", "train", label_file)
        with open(label_path, 'w') as f:
            if yolo_lines:
                f.write('\n'.join(yolo_lines))
        
        train_converted += 1
    
    # Process validation split (always use the same val_split)
    val_img_dir = os.path.join(input_dir, "val_split", "img")
    val_ann_dir = os.path.join(input_dir, "val_split", "ann")
    
    print(f"Validation images: {val_img_dir}")
    print(f"Validation annotations: {val_ann_dir}")
    
    val_images = [f for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    val_converted = 0
    val_with_annotations = 0
    
    print(f"Found {len(val_images)} validation images")
    
    for i, img_file in enumerate(val_images):
        if (i + 1) % 100 == 0:
            print(f"Processing val image {i+1}/{len(val_images)}: {img_file}")
        
        img_path = os.path.join(val_img_dir, img_file)
        ann_file = img_file + '.json'
        ann_path = os.path.join(val_ann_dir, ann_file)
        
        # Copy image
        dst_img = os.path.join(output_dir, "images", "val", img_file)
        shutil.copy2(img_path, dst_img)
        
        # Convert annotation if exists
        yolo_lines = []
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                
                # Get image dimensions
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                # Process bitmap objects
                for obj in ann_data.get('objects', []):
                    if obj.get('geometryType') == 'bitmap':
                        bitmap_data = obj.get('bitmap', {}).get('data')
                        bitmap_origin = obj.get('bitmap', {}).get('origin', [0, 0])
                        
                        if bitmap_data:
                            mask = decode_bitmap_to_mask(bitmap_data)
                            if mask is not None:
                                origin_x, origin_y = bitmap_origin
                                segments = mask_to_yolo_seg(mask, origin_x, origin_y, img_width, img_height)
                                for segment in segments:
                                    # YOLO format: class_id + normalized coordinates
                                    line = f"0 {' '.join([f'{coord:.6f}' for coord in segment])}"
                                    yolo_lines.append(line)
                
                if yolo_lines:
                    val_with_annotations += 1
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        # Write YOLO label file (even if empty)
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(output_dir, "labels", "val", label_file)
        with open(label_path, 'w') as f:
            if yolo_lines:
                f.write('\n'.join(yolo_lines))
        
        val_converted += 1
    
    # Create YAML configuration
    yaml_content = f"""path: {os.path.abspath(output_dir)}  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Val images (relative to 'path')

nc: 1  # Number of classes
names: ['defect']  # Class names
"""
    
    yaml_path = os.path.join(output_dir, f'yolo_segmentation_{subset_name}.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETED!")
    print(f"{'='*60}")
    print(f"Subset: {subset_name}")
    print(f"Training: {len(train_images)} images, {train_with_annotations} with annotations")
    print(f"Validation: {len(val_images)} images, {val_with_annotations} with annotations")
    print(f"Output directory: {output_dir}")
    print(f"YAML config: {yaml_path}")
    print(f"{'='*60}")
    
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Convert Severstal annotations to YOLO segmentation format")
    parser.add_argument("--subset", type=str, choices=["500", "1000", "2000", "full"], 
                       help="Dataset subset to convert (500, 1000, 2000, or full)")
    parser.add_argument("--input-dir", type=str, default="datasets/Data/splits",
                       help="Input directory with .jpg.json annotations")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for YOLO format dataset")
    parser.add_argument("--all-subsets", action="store_true",
                       help="Convert all subsets (500, 1000, 2000, full)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.subset and not args.all_subsets:
        parser.error("Must specify either --subset or --all-subsets")
    
    print("Severstal to YOLO Segmentation Converter")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    if args.all_subsets:
        print("Converting all subsets: 500, 1000, 2000, full")
        subsets = ["500", "1000", "2000", "full"]
    else:
        print(f"Converting subset: {args.subset}")
        subsets = [args.subset]
    
    print("-" * 50)
    
    start_time = time.time()
    
    # Process each subset
    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"CONVERTING SUBSET: {subset}")
        print(f"{'='*60}")
        
        # Create subset-specific output directory
        if subset == "full":
            subset_output_dir = os.path.join(args.output_dir, "full_dataset")
        else:
            subset_output_dir = os.path.join(args.output_dir, f"subset_{subset}")
        
        # Convert annotations
        yaml_path = convert_annotations_to_yolo(args.input_dir, subset_output_dir, subset)
        
        if yaml_path:
            print(f"✓ Successfully converted subset {subset}")
            print(f"  Dataset YAML: {yaml_path}")
        else:
            print(f"✗ Failed to convert subset {subset}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TOTAL CONVERSION TIME: {total_time/60:.2f} minutes")
    print("All conversions completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
