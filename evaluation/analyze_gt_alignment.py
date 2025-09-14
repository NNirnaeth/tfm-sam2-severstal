#!/usr/bin/env python3
"""
Analyze ground truth alignment with actual images
This script helps understand why YOLO predictions and GT annotations are in different areas
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse

def load_gt_annotations(gt_file):
    """Load ground truth annotations from a single file"""
    annotations = []
    
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            annotations.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    
    return annotations

def convert_bbox_to_pixels(bbox, img_width, img_height):
    """Convert normalized bbox to pixel coordinates"""
    x_center = bbox['x_center'] * img_width
    y_center = bbox['y_center'] * img_height
    width = bbox['width'] * img_width
    height = bbox['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    return x1, y1, x2, y2

def analyze_image_gt_alignment(image_path, gt_file, output_dir):
    """Analyze alignment between image and ground truth annotations"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]
    
    print(f"Image: {image_path}")
    print(f"Dimensions: {img_width}x{img_height}")
    
    # Load GT annotations
    gt_annotations = load_gt_annotations(gt_file)
    print(f"GT annotations: {len(gt_annotations)}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Image with GT annotations
    axes[1].imshow(image)
    axes[1].set_title('Image + Ground Truth Annotations')
    axes[1].axis('off')
    
    # Draw GT annotations
    for i, bbox in enumerate(gt_annotations):
        x1, y1, x2, y2 = convert_bbox_to_pixels(bbox, img_width, img_height)
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
        
        # Add label
        axes[1].text(x1, y1-5, f"GT{i}", color='red', fontsize=8, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        print(f"  GT {i}: normalized({bbox['x_center']:.4f}, {bbox['y_center']:.4f}, {bbox['width']:.4f}, {bbox['height']:.4f})")
        print(f"    Pixels: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"    Box size: {x2-x1}x{y2-y1}")
        
        # Check if bbox is reasonable
        if x2 <= x1 or y2 <= y1:
            print(f"    WARNING: Invalid bbox dimensions!")
        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            print(f"    WARNING: Bbox outside image bounds!")
    
    plt.tight_layout()
    
    # Save visualization
    image_name = os.path.basename(image_path).replace('.jpg', '')
    save_path = os.path.join(output_dir, f"gt_alignment_{image_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")
    
    # Analyze spatial distribution
    if gt_annotations:
        x_centers = [bbox['x_center'] for bbox in gt_annotations]
        y_centers = [bbox['y_center'] for bbox in gt_annotations]
        
        print(f"\nSpatial Analysis:")
        print(f"  X centers range: {min(x_centers):.4f} to {max(x_centers):.4f}")
        print(f"  Y centers range: {min(y_centers):.4f} to {max(y_centers):.4f}")
        print(f"  Average X center: {np.mean(x_centers):.4f}")
        print(f"  Average Y center: {np.mean(y_centers):.4f}")
        
        # Check if annotations are clustered on one side
        left_side = [x for x in x_centers if x < 0.5]
        right_side = [x for x in x_centers if x >= 0.5]
        
        print(f"  Left side (<0.5): {len(left_side)} annotations")
        print(f"  Right side (>=0.5): {len(right_side)} annotations")
        
        if len(left_side) > len(right_side) * 3:
            print(f"  WARNING: Annotations heavily clustered on left side!")
        elif len(right_side) > len(left_side) * 3:
            print(f"  WARNING: Annotations heavily clustered on right side!")

def main():
    parser = argparse.ArgumentParser(description='Analyze GT alignment with images')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth annotations')
    parser.add_argument('--output_dir', type=str, 
                       default='new_src/evaluation/gt_alignment_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = list(Path(args.image_dir).glob("*.jpg"))
    print(f"Found {len(image_files)} images")
    
    # Analyze first N images
    for i, image_path in enumerate(image_files[:args.num_images]):
        image_name = image_path.stem
        gt_file = os.path.join(args.gt_dir, f"{image_name}.txt")
        
        if os.path.exists(gt_file):
            print(f"\n{'='*60}")
            print(f"Analyzing image {i+1}/{min(args.num_images, len(image_files))}")
            print(f"{'='*60}")
            
            analyze_image_gt_alignment(str(image_path), gt_file, args.output_dir)
        else:
            print(f"GT file not found for {image_name}")
    
    print(f"\nAnalysis completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
