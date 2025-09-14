#!/usr/bin/env python3
"""
Verify Bbox Conversion from Bitmap Annotations
This script loads a few images and shows the original bitmap annotations
overlaid with the converted YOLO bounding boxes for manual verification.

Usage:
    python verify_bbox_conversion.py --num_images 2 --split train_split
"""

import os
import sys
import json
import cv2
import base64
import zlib
from PIL import Image, ImageDraw
from io import BytesIO
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Hardcoded paths
SEVERSTAL_SPLITS_DIR = "datasets/Data/splits"
YOLO_DATASET_DIR = "datasets/yolo_detection_fixed"

def load_bitmap_mask(bitmap_data, image_width, image_height):
    """Load bitmap mask from Supervisely format"""
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
        
        return bitmap_array
        
    except Exception as e:
        print(f"Error loading bitmap mask: {e}")
        return None

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
        
        return [[x_center, y_center, width, height]]
        
    except Exception as e:
        print(f"Error converting bitmap to bbox: {e}")
        return []

def yolo_to_xyxy(yolo_bbox, img_width, img_height):
    """Convert YOLO format to xyxy format for visualization"""
    x_center, y_center, width, height = yolo_bbox
    
    # Convert normalized coordinates to pixel coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Convert to xyxy format
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [int(x1), int(y1), int(x2), int(y2)]

def verify_conversion(split_name, num_images=2):
    """Verify bitmap to bbox conversion for a few images"""
    print(f"Verifying conversion for {split_name} split...")
    
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
        print(f"\n{'='*60}")
        print(f"Image {i+1}/{len(ann_files)}: {ann_file}")
        print(f"{'='*60}")
        
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
        bitmap_masks = []
        converted_bboxes = []
        
        for obj in annotation.get('objects', []):
            if 'bitmap' in obj:
                # Load bitmap mask
                mask = load_bitmap_mask(
                    obj['bitmap']['data'],
                    img_width,
                    img_height
                )
                if mask is not None:
                    bitmap_masks.append(mask)
                
                # Convert to bbox
                bboxes = convert_bitmap_to_bbox(
                    obj['bitmap']['data'],
                    img_width,
                    img_height
                )
                converted_bboxes.extend(bboxes)
        
        print(f"Found {len(bitmap_masks)} bitmap masks")
        print(f"Converted to {len(converted_bboxes)} bounding boxes")
        print(f"YOLO labels contain {len(yolo_bboxes)} bounding boxes")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Verification: {image_name}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Image with bitmap masks
        axes[0, 1].imshow(image)
        for mask in bitmap_masks:
            # Resize mask to match image dimensions if needed
            if mask.shape != (img_height, img_width):
                mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            
            # Create colored overlay for mask
            mask_colored = np.zeros((img_height, img_width, 4))
            mask_colored[:, :, 0] = 1.0  # Red channel
            mask_colored[:, :, 3] = mask_resized * 0.5  # Alpha channel
            axes[0, 1].imshow(mask_colored)
        axes[0, 1].set_title('Original Bitmap Masks (Red)')
        axes[0, 1].axis('off')
        
        # Image with converted bboxes
        axes[1, 0].imshow(image)
        for bbox in converted_bboxes:
            x1, y1, x2, y2 = yolo_to_xyxy(bbox, img_width, img_height)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='green', facecolor='none')
            axes[1, 0].add_patch(rect)
        axes[1, 0].set_title('Converted Bboxes (Green)')
        axes[1, 0].axis('off')
        
        # Image with YOLO labels
        axes[1, 1].imshow(image)
        for bbox in yolo_bboxes:
            x1, y1, x2, y2 = yolo_to_xyxy(bbox, img_width, img_height)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='blue', facecolor='none')
            axes[1, 1].add_patch(rect)
        axes[1, 1].set_title('YOLO Labels (Blue)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save verification image
        output_dir = "verification_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"verify_{image_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Verification image saved: {output_path}")
        
        # Show plot
        plt.show()
        
        # Print detailed bbox information
        print(f"\nDetailed bbox information:")
        for j, bbox in enumerate(converted_bboxes):
            x1, y1, x2, y2 = yolo_to_xyxy(bbox, img_width, img_height)
            print(f"  Bbox {j+1}: ({x1}, {y1}) -> ({x2}, {y2}) [size: {x2-x1}x{y2-y1}]")
        
        # Wait for user input
        input(f"\nPress Enter to continue to next image...")

def main():
    parser = argparse.ArgumentParser(description='Verify bitmap to bbox conversion')
    parser.add_argument('--num_images', type=int, default=2,
                       help='Number of images to verify (default: 2)')
    parser.add_argument('--split', type=str, default='train_split',
                       choices=['train_split', 'val_split', 'test_split'],
                       help='Dataset split to verify (default: train_split)')
    
    args = parser.parse_args()
    
    print(f"Bitmap to Bbox Conversion Verification")
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
    
    verify_conversion(args.split, args.num_images)

if __name__ == "__main__":
    main()
