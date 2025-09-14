#!/usr/bin/env python3
"""
Verify YOLO conversion results by visualizing some examples

This script helps verify that the bitmap to YOLO conversion is working correctly
by showing original images with ground truth masks and YOLO polygon overlays.
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
import zlib
import io
import cv2

def decode_bitmap_to_mask(bitmap_data):
    """Decode PNG bitmap data from Supervisely format"""
    try:
        decoded_data = base64.b64decode(bitmap_data)
        decompressed_data = zlib.decompress(decoded_data)
        mask = Image.open(io.BytesIO(decompressed_data))
        mask_array = np.array(mask)
        return (mask_array > 0).astype(np.uint8)
    except Exception as e:
        print(f"Error decoding bitmap: {e}")
        return None

def yolo_to_pixel_coords(yolo_coords, img_width, img_height):
    """Convert YOLO normalized coordinates to pixel coordinates"""
    pixel_coords = []
    for i in range(0, len(yolo_coords), 2):
        x = yolo_coords[i] * img_width
        y = yolo_coords[i+1] * img_height
        pixel_coords.extend([x, y])
    return pixel_coords

def visualize_conversion(image_path, annotation_path, yolo_label_path, output_path=None):
    """Visualize original annotation vs YOLO conversion"""
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]
    
    # Load original annotation
    with open(annotation_path, 'r') as f:
        ann_data = json.load(f)
    
    # Create ground truth mask
    gt_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for obj in ann_data.get('objects', []):
        if obj.get('geometryType') == 'bitmap':
            bitmap_data = obj.get('bitmap', {}).get('data')
            bitmap_origin = obj.get('bitmap', {}).get('origin', [0, 0])
            if bitmap_data:
                mask = decode_bitmap_to_mask(bitmap_data)
                if mask is not None:
                    origin_x, origin_y = bitmap_origin
                    h, w = mask.shape
                    # Place mask at correct position
                    y1, y2 = origin_y, origin_y + h
                    x1, x2 = origin_x, origin_x + w
                    # Ensure bounds
                    y1, y2 = max(0, y1), min(img_height, y2)
                    x1, x2 = max(0, x1), min(img_width, x2)
                    mask_h, mask_w = y2-y1, x2-x1
                    if mask_h > 0 and mask_w > 0:
                        gt_mask[y1:y2, x1:x2] = mask[:mask_h, :mask_w]
    
    # Load YOLO labels
    yolo_polygons = []
    if os.path.exists(yolo_label_path):
        with open(yolo_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 2:  # class_id + coordinates
                    coords = [float(x) for x in parts[1:]]
                    pixel_coords = yolo_to_pixel_coords(coords, img_width, img_height)
                    # Convert to polygon points
                    polygon = []
                    for i in range(0, len(pixel_coords), 2):
                        polygon.append([pixel_coords[i], pixel_coords[i+1]])
                    yolo_polygons.append(np.array(polygon, dtype=np.int32))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask overlay
    axes[1].imshow(img_array)
    if gt_mask.any():
        axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # YOLO polygons overlay
    axes[2].imshow(img_array)
    for polygon in yolo_polygons:
        if len(polygon) > 2:
            axes[2].plot(polygon[:, 0], polygon[:, 1], 'b-', linewidth=2)
            axes[2].plot([polygon[-1, 0], polygon[0, 0]], [polygon[-1, 1], polygon[0, 1]], 'b-', linewidth=2)
    axes[2].set_title('YOLO Polygons')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Verify YOLO conversion results")
    parser.add_argument("--subset", type=str, choices=["500", "1000", "2000", "full"], 
                       required=True, help="Dataset subset to verify")
    parser.add_argument("--converted-dir", type=str, default="datasets/yolo_segmentation_subsets",
                       help="Directory containing converted YOLO datasets")
    parser.add_argument("--original-dir", type=str, default="datasets/Data/splits",
                       help="Directory containing original annotations")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--output-dir", type=str, default="verification_plots",
                       help="Directory to save verification plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get paths
    if args.subset == "full":
        converted_subset_dir = os.path.join(args.converted_dir, "full_dataset")
        original_img_dir = os.path.join(args.original_dir, "train_split", "img")
        original_ann_dir = os.path.join(args.original_dir, "train_split", "ann")
    else:
        converted_subset_dir = os.path.join(args.converted_dir, f"subset_{args.subset}")
        original_img_dir = os.path.join(args.original_dir, "subsets", f"{args.subset}_subset", "img")
        original_ann_dir = os.path.join(args.original_dir, "subsets", f"{args.subset}_subset", "ann")
    
    yolo_img_dir = os.path.join(converted_subset_dir, "images", "train")
    yolo_label_dir = os.path.join(converted_subset_dir, "labels", "train")
    
    # Verify directories exist
    for directory in [original_img_dir, original_ann_dir, yolo_img_dir, yolo_label_dir]:
        if not os.path.exists(directory):
            print(f"ERROR: Directory not found: {directory}")
            return
    
    # Get sample images
    image_files = [f for f in os.listdir(original_img_dir) if f.endswith('.jpg')][:args.num_samples]
    
    print(f"Verifying conversion for subset {args.subset}")
    print(f"Visualizing {len(image_files)} samples...")
    
    for i, img_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_file}")
        
        # Paths
        image_path = os.path.join(original_img_dir, img_file)
        annotation_path = os.path.join(original_ann_dir, img_file + '.json')
        yolo_label_path = os.path.join(yolo_label_dir, img_file.replace('.jpg', '.txt'))
        output_path = os.path.join(args.output_dir, f"verification_{args.subset}_{img_file.replace('.jpg', '.png')}")
        
        # Visualize
        visualize_conversion(image_path, annotation_path, yolo_label_path, output_path)
    
    print(f"Verification completed! Check plots in {args.output_dir}")

if __name__ == "__main__":
    main()

