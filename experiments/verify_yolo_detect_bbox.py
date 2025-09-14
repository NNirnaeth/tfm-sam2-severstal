#!/usr/bin/env python3
"""
Verify YOLO Detection Bbox Distribution and Positioning
This script verifies that bounding boxes are correctly positioned using the origin field
and creates visualization comparisons between YOLO bboxes and ground truth bitmaps.

Usage:
    python verify_yolo_detect_bbox.py --subset 500 --num_examples 5
"""

import os
import sys
import json
import cv2
import base64
import zlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO
import argparse
from pathlib import Path
import random

# Hardcoded paths
SEVERSTAL_SPLITS_DIR = "datasets/Data/splits"
YOLO_DETECTION_DIR = "datasets/yolo_detection_fixed"

def load_yolo_bboxes(label_file):
    """Load YOLO bounding boxes from label file"""
    bboxes = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
    return bboxes

def load_gt_bitmaps(ann_file, img_width, img_height):
    """Load ground truth bitmaps from annotation file"""
    bitmaps = []
    if os.path.exists(ann_file):
        try:
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
            
            for obj in annotation.get('objects', []):
                if 'bitmap' in obj:
                    bitmap_data = obj['bitmap']['data']
                    origin = obj['bitmap']['origin']
                    
                    # Decode bitmap
                    decoded_data = base64.b64decode(bitmap_data)
                    decompressed_data = zlib.decompress(decoded_data)
                    
                    # Use cv2 to decode PNG
                    nparr = np.frombuffer(decompressed_data, np.uint8)
                    bitmap_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    
                    if bitmap_img is not None:
                        # Handle different image formats
                        if len(bitmap_img.shape) == 3:
                            bitmap_img = bitmap_img[:, :, 0]
                        
                        # Ensure binary mask
                        bitmap_mask = (bitmap_img > 0).astype(np.uint8)
                        
                        if np.sum(bitmap_mask) > 0:
                            bitmaps.append({
                                'mask': bitmap_mask,
                                'origin': origin
                            })
        except Exception as e:
            print(f"Error loading GT bitmaps from {ann_file}: {e}")
    
    return bitmaps

def yolo_to_xyxy(bbox, img_width, img_height):
    """Convert YOLO format to xyxy format"""
    x_center = bbox['x_center'] * img_width
    y_center = bbox['y_center'] * img_height
    width = bbox['width'] * img_width
    height = bbox['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return [x1, y1, x2, y2]

def create_gt_union_mask(bitmaps, img_height, img_width):
    """Create union mask from all bitmaps with proper origin positioning"""
    union_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for bitmap_data in bitmaps:
        mask = bitmap_data['mask']
        origin_x, origin_y = bitmap_data['origin']
        
        # Calculate actual position in full image
        mask_height, mask_width = mask.shape
        end_y = min(origin_y + mask_height, img_height)
        end_x = min(origin_x + mask_width, img_width)
        
        # Ensure coordinates are within bounds
        start_y = max(0, origin_y)
        start_x = max(0, origin_x)
        
        if end_y > start_y and end_x > start_x:
            # Place bitmap in correct position
            union_mask[start_y:end_y, start_x:end_x] = np.maximum(
                union_mask[start_y:end_y, start_x:end_x],
                mask[:(end_y - start_y), :(end_x - start_x)]
            )
    
    return union_mask

def visualize_comparison(image_path, yolo_bboxes, gt_bitmaps, output_path, img_width, img_height):
    """Create visualization comparing YOLO bboxes with GT bitmaps"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create GT union mask
    gt_union = create_gt_union_mask(gt_bitmaps, img_height, img_width)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Image with YOLO bboxes
    axes[1].imshow(image)
    for bbox in yolo_bboxes:
        x1, y1, x2, y2 = yolo_to_xyxy(bbox, img_width, img_height)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
    axes[1].set_title('YOLO Bboxes (Red)', fontsize=14)
    axes[1].axis('off')
    
    # Image with GT bitmaps
    axes[2].imshow(image)
    # Overlay GT mask with transparency
    gt_colored = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    gt_colored[:, :, 0] = 255  # Red channel
    gt_colored[:, :, 3] = gt_union * 128  # Alpha channel (semi-transparent)
    axes[2].imshow(gt_colored, alpha=0.5)
    axes[2].set_title('GT Bitmaps (Red Overlay)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {output_path}")

def analyze_bbox_distribution(subset_dir, subset_size, num_examples=5):
    """Analyze bbox distribution and create visualizations"""
    
    print(f"\n{'='*60}")
    print(f"VERIFYING SUBSET {subset_size} BBOX DISTRIBUTION")
    print(f"{'='*60}")
    
    # Paths
    labels_dir = os.path.join(subset_dir, "labels", "train_split")
    images_dir = os.path.join(subset_dir, "images", "train_split")
    gt_ann_dir = os.path.join(SEVERSTAL_SPLITS_DIR, "train_split", "ann")
    gt_img_dir = os.path.join(SEVERSTAL_SPLITS_DIR, "train_split", "img")
    
    # Get all label files
    label_files = list(Path(labels_dir).glob("*.txt"))
    print(f"Found {len(label_files)} label files in subset {subset_size}")
    
    if len(label_files) == 0:
        print(f"No label files found in {labels_dir}")
        return
    
    # Analyze bbox distribution
    all_x_centers = []
    all_y_centers = []
    all_widths = []
    all_heights = []
    total_bboxes = 0
    
    for label_file in label_files:
        bboxes = load_yolo_bboxes(str(label_file))
        for bbox in bboxes:
            all_x_centers.append(bbox['x_center'])
            all_y_centers.append(bbox['y_center'])
            all_widths.append(bbox['width'])
            all_heights.append(bbox['height'])
            total_bboxes += 1
    
    print(f"\nBbox Distribution Analysis:")
    print(f"Total bboxes: {total_bboxes}")
    print(f"X-center range: {min(all_x_centers):.4f} - {max(all_x_centers):.4f}")
    print(f"Y-center range: {min(all_y_centers):.4f} - {max(all_y_centers):.4f}")
    print(f"Width range: {min(all_widths):.4f} - {max(all_widths):.4f}")
    print(f"Height range: {min(all_heights):.4f} - {max(all_heights):.4f}")
    
    # Check if bboxes are well distributed
    x_std = np.std(all_x_centers)
    y_std = np.std(all_y_centers)
    print(f"X-center std: {x_std:.4f} (should be > 0.2 for good distribution)")
    print(f"Y-center std: {y_std:.4f} (should be > 0.2 for good distribution)")
    
    if x_std < 0.1:
        print("⚠️  WARNING: X-centers are not well distributed - possible origin field issue!")
    else:
        print("✅ X-centers are well distributed")
    
    if y_std < 0.1:
        print("⚠️  WARNING: Y-centers are not well distributed - possible origin field issue!")
    else:
        print("✅ Y-centers are well distributed")
    
    # Create visualizations
    print(f"\nCreating {num_examples} visualization examples...")
    
    # Select random examples
    random.seed(42)
    selected_files = random.sample(label_files, min(num_examples, len(label_files)))
    
    for i, label_file in enumerate(selected_files):
        image_name = label_file.stem + ".jpg"
        
        # Paths
        yolo_label_path = str(label_file)
        yolo_image_path = os.path.join(images_dir, image_name)
        gt_ann_path = os.path.join(gt_ann_dir, image_name + ".json")
        gt_img_path = os.path.join(gt_img_dir, image_name)
        
        # Load data
        yolo_bboxes = load_yolo_bboxes(yolo_label_path)
        
        if not os.path.exists(gt_img_path):
            print(f"GT image not found: {gt_img_path}")
            continue
        
        # Get image dimensions
        with Image.open(gt_img_path) as img:
            img_width, img_height = img.size
        
        gt_bitmaps = load_gt_bitmaps(gt_ann_path, img_width, img_height)
        
        # Create visualization
        output_path = os.path.join(subset_dir, f"verification_example_{i+1}_{image_name}.jpg")
        visualize_comparison(gt_img_path, yolo_bboxes, gt_bitmaps, output_path, img_width, img_height)
        
        print(f"  Example {i+1}: {image_name} - {len(yolo_bboxes)} bboxes, {len(gt_bitmaps)} GT bitmaps")
    
    print(f"\nVerification completed for subset {subset_size}!")
    print(f"Visualizations saved in: {subset_dir}")

def main():
    parser = argparse.ArgumentParser(description='Verify YOLO detection bbox distribution and positioning')
    parser.add_argument('--subset', type=int, default=500,
                       help='Subset size to verify (500, 1000, 2000)')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of visualization examples to create')
    parser.add_argument('--all_subsets', action='store_true',
                       help='Verify all subsets (500, 1000, 2000)')
    
    args = parser.parse_args()
    
    if args.all_subsets:
        subsets = [500, 1000, 2000]
    else:
        subsets = [args.subset]
    
    for subset_size in subsets:
        subset_dir = os.path.join(YOLO_DETECTION_DIR, f"subset_{subset_size}")
        
        if not os.path.exists(subset_dir):
            print(f"Error: Subset directory not found: {subset_dir}")
            continue
        
        analyze_bbox_distribution(subset_dir, subset_size, args.num_examples)

if __name__ == "__main__":
    main()

