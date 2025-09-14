#!/usr/bin/env python3
"""
Verify YOLO Segmentation Conversion from Bitmap Annotations
This script loads images and shows the original bitmap annotations
overlaid with the converted YOLO segmentation polygons for manual verification.

Usage:
    python verify_yolo_seg_conversion.py --subset 1000 --num_images 2
"""

import os
import sys
import json
import base64
import zlib
from PIL import Image
from io import BytesIO
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
import cv2

def decode_png_bitmap(bitmap_data):
    """Decode PNG bitmap data from Supervisely format"""
    try:
        # Decode base64 and decompress PNG
        decoded_data = base64.b64decode(bitmap_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Open as PIL Image and convert to numpy array
        mask = Image.open(BytesIO(decompressed_data))
        mask_array = np.array(mask)
        
        # Convert to binary mask (0 or 1)
        binary_mask = (mask_array > 0).astype(np.uint8)
        return binary_mask
    except Exception as e:
        print(f"Error decoding bitmap: {e}")
        return None

def yolo_seg_to_polygon(yolo_coords, img_width, img_height):
    """Convert YOLO segmentation coordinates to pixel coordinates"""
    try:
        # Parse coordinates (skip class id)
        coords = [float(x) for x in yolo_coords[1:]]
        
        # Convert to pixel coordinates
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_width
            y = coords[i+1] * img_height
            pixel_coords.append([x, y])
        
        return np.array(pixel_coords)
    except Exception as e:
        print(f"Error converting YOLO coords: {e}")
        return None

def verify_conversion(subset, num_images=2):
    """Verify bitmap to YOLO segmentation conversion for a few images"""
    print(f"Verifying YOLO segmentation conversion for subset {subset}...")
    
    # Paths
    if subset in ["500", "1000", "2000"]:
        source_img_dir = f"datasets/Data/splits/subsets/{subset}_subset/img"
        source_ann_dir = f"datasets/Data/splits/subsets/{subset}_subset/ann"
        yolo_dataset_dir = f"datasets/yolo_segmentation_subsets/subset_{subset}"
    else:
        source_img_dir = "datasets/Data/splits/train_split/img"
        source_ann_dir = "datasets/Data/splits/train_split/ann"
        yolo_dataset_dir = "datasets/yolo_segmentation_subsets/full_dataset"
    
    yolo_labels_dir = os.path.join(yolo_dataset_dir, "labels", "train")
    
    if not os.path.exists(source_img_dir):
        print(f"Error: Source images directory not found: {source_img_dir}")
        return
    
    if not os.path.exists(yolo_labels_dir):
        print(f"Error: YOLO labels directory not found: {yolo_labels_dir}")
        print("Run convert_to_yolo_format.py first!")
        return
    
    # Get images
    img_files = [f for f in os.listdir(source_img_dir) if f.endswith('.jpg')]
    img_files = img_files[:num_images]  # Limit to requested number
    
    for i, img_file in enumerate(img_files):
        print(f"\n{'='*60}")
        print(f"Image {i+1}/{len(img_files)}: {img_file}")
        print(f"{'='*60}")
        
        # Paths
        image_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(source_img_dir, img_file)
        ann_path = os.path.join(source_ann_dir, f"{img_file}.json")
        label_path = os.path.join(yolo_labels_dir, f"{image_name}.txt")
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        if not os.path.exists(ann_path):
            print(f"Annotation not found: {ann_path}")
            continue
        
        if not os.path.exists(label_path):
            print(f"YOLO label not found: {label_path}")
            continue
        
        # Load image
        try:
            image = Image.open(img_path)
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
            
            yolo_polygons = []
            for line in yolo_lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 7:  # class + at least 3 points
                        polygon = yolo_seg_to_polygon(parts, img_width, img_height)
                        if polygon is not None:
                            yolo_polygons.append(polygon)
        except Exception as e:
            print(f"Error loading YOLO labels: {e}")
            continue
        
        # Process bitmap annotations
        bitmap_masks = []
        bitmap_origins = []
        
        for obj in annotation.get('objects', []):
            if obj.get('geometryType') == 'bitmap':
                bitmap_data = obj.get('bitmap', {}).get('data')
                bitmap_origin = obj.get('bitmap', {}).get('origin', [0, 0])
                
                if bitmap_data:
                    # Load bitmap mask
                    mask = decode_png_bitmap(bitmap_data)
                    if mask is not None:
                        bitmap_masks.append(mask)
                        bitmap_origins.append(bitmap_origin)
        
        print(f"Found {len(bitmap_masks)} bitmap masks")
        print(f"Converted to {len(yolo_polygons)} YOLO polygons")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'YOLO Segmentation Verification: {img_file} (Subset {subset})', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Image with bitmap masks positioned correctly
        axes[0, 1].imshow(image)
        for mask, origin in zip(bitmap_masks, bitmap_origins):
            # Create full-size mask and place the bitmap at correct origin
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Get bitmap dimensions
            mask_h, mask_w = mask.shape
            x0, y0 = origin
            
            # Ensure coordinates are within image bounds
            x0 = max(0, min(x0, img_width - 1))
            y0 = max(0, min(y0, img_height - 1))
            x1 = min(x0 + mask_w, img_width)
            y1 = min(y0 + mask_h, img_height)
            
            # Adjust mask dimensions if needed
            actual_mask_h = y1 - y0
            actual_mask_w = x1 - x0
            
            if actual_mask_h > 0 and actual_mask_w > 0:
                # Place the mask at correct position
                full_mask[y0:y1, x0:x1] = mask[:actual_mask_h, :actual_mask_w]
                
                # Create colored overlay for mask
                mask_colored = np.zeros((img_height, img_width, 4))
                mask_colored[:, :, 0] = 1.0  # Red channel
                mask_colored[:, :, 3] = full_mask * 0.5  # Alpha channel
                axes[0, 1].imshow(mask_colored)
        
        axes[0, 1].set_title(f'Original Bitmap Masks with Origin (Red) - {len(bitmap_masks)} masks')
        axes[0, 1].axis('off')
        
        # Image with YOLO polygons
        axes[1, 0].imshow(image)
        for j, polygon in enumerate(yolo_polygons):
            poly_patch = MplPolygon(polygon, closed=True, fill=False, 
                                  edgecolor='green', linewidth=2, alpha=0.8)
            axes[1, 0].add_patch(poly_patch)
        axes[1, 0].set_title(f'YOLO Segmentation Polygons (Green) - {len(yolo_polygons)} polygons')
        axes[1, 0].axis('off')
        
        # Overlay comparison
        axes[1, 1].imshow(image)
        # Original masks in red
        for mask, origin in zip(bitmap_masks, bitmap_origins):
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            mask_h, mask_w = mask.shape
            x0, y0 = origin
            x0 = max(0, min(x0, img_width - 1))
            y0 = max(0, min(y0, img_height - 1))
            x1 = min(x0 + mask_w, img_width)
            y1 = min(y0 + mask_h, img_height)
            actual_mask_h = y1 - y0
            actual_mask_w = x1 - x0
            if actual_mask_h > 0 and actual_mask_w > 0:
                full_mask[y0:y1, x0:x1] = mask[:actual_mask_h, :actual_mask_w]
                mask_colored = np.zeros((img_height, img_width, 4))
                mask_colored[:, :, 0] = 1.0  # Red
                mask_colored[:, :, 3] = full_mask * 0.3
                axes[1, 1].imshow(mask_colored)
        
        # YOLO polygons in green
        for polygon in yolo_polygons:
            poly_patch = MplPolygon(polygon, closed=True, fill=False, 
                                  edgecolor='green', linewidth=2, alpha=0.8)
            axes[1, 1].add_patch(poly_patch)
        
        axes[1, 1].set_title('Overlay: Original (Red) vs YOLO (Green)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save verification image
        output_dir = "verification_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"verify_yolo_seg_subset{subset}_{image_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Verification image saved: {output_path}")
        
        # Show plot
        plt.show()
        
        # Print detailed information
        print(f"\nDetailed information:")
        print(f"  Original bitmap masks: {len(bitmap_masks)}")
        for j, (mask, origin) in enumerate(zip(bitmap_masks, bitmap_origins)):
            print(f"    Mask {j+1}: size {mask.shape}, origin {origin}")
        
        print(f"  YOLO polygons: {len(yolo_polygons)}")
        for j, polygon in enumerate(yolo_polygons):
            print(f"    Polygon {j+1}: {len(polygon)} points")
        
        # Wait for user input
        input(f"\nPress Enter to continue to next image...")

def main():
    parser = argparse.ArgumentParser(description='Verify YOLO segmentation conversion from bitmap')
    parser.add_argument('--subset', type=str, default='1000',
                       choices=['500', '1000', '2000', 'full'],
                       help='Dataset subset to verify (default: 1000)')
    parser.add_argument('--num_images', type=int, default=2,
                       help='Number of images to verify (default: 2)')
    
    args = parser.parse_args()
    
    print(f"YOLO Segmentation Conversion Verification")
    print(f"Subset: {args.subset}")
    print(f"Number of images: {args.num_images}")
    
    verify_conversion(args.subset, args.num_images)

if __name__ == "__main__":
    main()
