#!/usr/bin/env python3
"""
Verify Segmentation Conversion from Bitmap Annotations
This script loads a few images and shows the original bitmap annotations
overlaid with the converted YOLO segmentation polygons for manual verification.

Usage:
    python verify_seg_conversion.py --num_images 2 --split train_split
"""

import os
import sys
import json
import cv2
import base64
import zlib
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Hardcoded paths
SEVERSTAL_SPLITS_DIR = "datasets/Data/splits"
YOLO_DATASET_DIR = "datasets/yolo_segmentation_fixed"

def decode_png_bitmap(annotation_data, img_width, img_height):
    """Decode PNG bitmap from Severstal annotation with proper positioning - CORRECTED VERSION"""
    try:
        # Decode bitmap from base64 and zlib (same as corrected detection script)
        decoded_data = base64.b64decode(annotation_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Use cv2 to decode PNG (same as working scripts)
        nparr = np.frombuffer(decompressed_data, np.uint8)
        bitmap_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if bitmap_img is None:
            print("    Warning: Failed to decode bitmap with cv2")
            return None
        
        # Handle different image formats
        if len(bitmap_img.shape) == 3:
            bitmap_img = bitmap_img[:, :, 0]  # Take first channel
        
        # Ensure binary mask
        binary_mask = (bitmap_img > 0).astype(np.uint8)
        
        return binary_mask
        
    except Exception as e:
        print(f"Error decoding annotation: {e}")
        return None

def yolo_seg_to_polygon(yolo_seg, img_width, img_height):
    """Convert YOLO segmentation format to polygon coordinates for visualization"""
    coords = []
    for i in range(0, len(yolo_seg), 2):
        x = yolo_seg[i] * img_width
        y = yolo_seg[i + 1] * img_height
        coords.append([x, y])
    return np.array(coords, dtype=np.int32)

def create_verification_images(image, bitmap_masks, full_masks, yolo_segments, image_name, output_dir):
    """Create verification images showing GT vs predictions for segmentation"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PIL image to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_width, img_height = image.size
    
    # 1. Original image with bitmap masks (GT - at origin 0,0)
    gt_image = image.copy()
    gt_draw = ImageDraw.Draw(gt_image)
    
    for i, mask in enumerate(bitmap_masks):
        # Resize mask to match image dimensions if needed
        if mask.shape != (img_height, img_width):
            mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        
        # Create colored overlay for mask
        mask_overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_overlay)
        
        # Find mask pixels and draw them
        mask_pixels = np.where(mask_resized > 0)
        for y, x in zip(mask_pixels[0], mask_pixels[1]):
            mask_draw.point((x, y), fill=(255, 0, 0, 128))  # Red with transparency
        
        # Composite the overlay
        gt_image = Image.alpha_composite(gt_image.convert('RGBA'), mask_overlay).convert('RGB')
    
    # Add title
    gt_draw = ImageDraw.Draw(gt_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    gt_draw.text((10, 10), "Ground Truth (Bitmap Masks at 0,0)", fill=(255, 255, 255), font=font)
    gt_draw.text((10, 35), f"Masks: {len(bitmap_masks)}", fill=(255, 255, 255), font=font)
    
    # 2. Original image with corrected full masks (at correct origin)
    corrected_image = image.copy()
    corrected_draw = ImageDraw.Draw(corrected_image)
    
    for i, full_mask in enumerate(full_masks):
        # Create colored overlay for mask
        mask_overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_overlay)
        
        # Find mask pixels and draw them
        mask_pixels = np.where(full_mask > 0)
        for y, x in zip(mask_pixels[0], mask_pixels[1]):
            mask_draw.point((x, y), fill=(0, 255, 0, 128))  # Green with transparency
        
        # Composite the overlay
        corrected_image = Image.alpha_composite(corrected_image.convert('RGBA'), mask_overlay).convert('RGB')
    
    # Add title
    corrected_draw = ImageDraw.Draw(corrected_image)
    corrected_draw.text((10, 10), "Corrected Masks (at correct origin)", fill=(255, 255, 255), font=font)
    corrected_draw.text((10, 35), f"Masks: {len(full_masks)}", fill=(255, 255, 255), font=font)
    
    # 3. Original image with YOLO segmentation polygons
    yolo_image = image.copy()
    yolo_draw = ImageDraw.Draw(yolo_image)
    
    for j, segment in enumerate(yolo_segments):
        try:
            polygon_coords = yolo_seg_to_polygon(segment, img_width, img_height)
            if len(polygon_coords) >= 3:
                # Convert to list of tuples for PIL
                polygon_points = [(int(x), int(y)) for x, y in polygon_coords]
                yolo_draw.polygon(polygon_points, outline=(0, 0, 255), width=3)
                
                # Add point markers
                for x, y in polygon_points:
                    yolo_draw.ellipse([x-2, y-2, x+2, y+2], fill=(0, 0, 255))
                
                # Add polygon number
                if polygon_points:
                    yolo_draw.text((polygon_points[0][0], polygon_points[0][1]-25), 
                                 f"Poly {j+1}", fill=(0, 0, 255), font=font)
        except Exception as e:
            print(f"Error drawing polygon {j}: {e}")
            continue
    
    # Add title
    yolo_draw.text((10, 10), "YOLO Segmentation Polygons", fill=(255, 255, 255), font=font)
    yolo_draw.text((10, 35), f"Polygons: {len(yolo_segments)}", fill=(255, 255, 255), font=font)
    
    # 4. Side-by-side comparison
    comparison_width = img_width * 2
    comparison_height = img_height + 100  # Extra space for titles
    
    comparison_image = Image.new('RGB', (comparison_width, comparison_height), (0, 0, 0))
    comparison_draw = ImageDraw.Draw(comparison_image)
    
    # Paste corrected image
    comparison_image.paste(corrected_image, (0, 50))
    
    # Paste yolo image
    comparison_image.paste(yolo_image, (img_width, 50))
    
    # Add main title
    comparison_draw.text((10, 10), f"Segmentation Verification: {image_name}", fill=(255, 255, 255), font=font)
    comparison_draw.text((10, 30), f"Image size: {img_width}x{img_height}", fill=(255, 255, 255), font=font)
    
    # Save all images
    gt_path = os.path.join(output_dir, f"{image_name}_gt.png")
    corrected_path = os.path.join(output_dir, f"{image_name}_corrected.png")
    yolo_path = os.path.join(output_dir, f"{image_name}_yolo.png")
    comparison_path = os.path.join(output_dir, f"{image_name}_comparison.png")
    
    gt_image.save(gt_path)
    corrected_image.save(corrected_path)
    yolo_image.save(yolo_path)
    comparison_image.save(comparison_path)
    
    return {
        'gt': gt_path,
        'corrected': corrected_path,
        'yolo': yolo_path,
        'comparison': comparison_path
    }

def verify_conversion(split_name, num_images=2):
    """Verify bitmap to segmentation conversion for a few images"""
    print(f"Verifying segmentation conversion for {split_name} split...")
    
    # Paths
    split_ann_dir = os.path.join(SEVERSTAL_SPLITS_DIR, split_name, "ann")
    split_img_dir = os.path.join(SEVERSTAL_SPLITS_DIR, split_name, "img")
    
    # Map split names for YOLO dataset (train_split -> train, val_split -> val)
    yolo_split_name = split_name.replace('_split', '')
    yolo_labels_dir = os.path.join(YOLO_DATASET_DIR, "labels", yolo_split_name)
    
    if not os.path.exists(split_ann_dir):
        print(f"Error: Annotations directory not found: {split_ann_dir}")
        return
    
    if not os.path.exists(yolo_labels_dir):
        print(f"Error: YOLO labels directory not found: {yolo_labels_dir}")
        print("Run prepare_yolo_seg_dataset.py first!")
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
            yolo_segments = []
            for line in yolo_lines:
                parts = line.strip().split()
                if len(parts) >= 7:  # class_id + at least 3 points (6 coordinates)
                    coords = [float(x) for x in parts[1:]]  # Skip class_id
                    yolo_segments.append(coords)
        except Exception as e:
            print(f"Error loading YOLO labels: {e}")
            continue
        
        # Process bitmap annotations with origin correction
        bitmap_masks = []
        full_masks = []
        
        for obj in annotation.get('objects', []):
            if 'bitmap' in obj:
                # Get bitmap data and origin
                bitmap_data = obj.get('bitmap', {}).get('data')
                bitmap_origin = obj.get('bitmap', {}).get('origin', [0, 0])
                
                if bitmap_data:
                    # Decode mask
                    mask = decode_png_bitmap(bitmap_data, img_width, img_height)
                    if mask is not None:
                        bitmap_masks.append(mask)
                        
                        # Create full-size mask and place the bitmap at correct origin
                        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        
                        # Get bitmap dimensions
                        mask_h, mask_w = mask.shape
                        x0, y0 = bitmap_origin
                        
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
                            full_masks.append(full_mask)
                            
                            print(f"  Bitmap origin: ({x0}, {y0}), size: {mask_w}x{mask_h}")
                            print(f"  Placed at: ({x0}, {y0}) -> ({x1}, {y1})")
        
        print(f"Found {len(bitmap_masks)} bitmap masks")
        print(f"Created {len(full_masks)} full-size masks")
        print(f"YOLO labels contain {len(yolo_segments)} segmentation polygons")
        
        # Create verification images
        image_paths = create_verification_images(
            image, bitmap_masks, full_masks, yolo_segments, 
            image_name, "new_src/experiments/verify_bbox_seg"
        )
        
        print(f"\nVerification images saved:")
        print(f"  Ground Truth: {image_paths['gt']}")
        print(f"  Corrected Masks: {image_paths['corrected']}")
        print(f"  YOLO Polygons: {image_paths['yolo']}")
        print(f"  Comparison: {image_paths['comparison']}")
        
        # Print detailed polygon information
        print(f"\nDetailed polygon information:")
        for j, segment in enumerate(yolo_segments):
            num_points = len(segment) // 2
            print(f"  Polygon {j+1}: {num_points} points")
            if num_points > 0:
                # Show first few points
                points_str = ""
                for k in range(0, min(6, len(segment)), 2):
                    x = segment[k] * img_width
                    y = segment[k + 1] * img_height
                    points_str += f"({x:.1f},{y:.1f}) "
                if num_points > 3:
                    points_str += "..."
                print(f"    Points: {points_str}")
    
    print(f"\n{'='*80}")
    print(f"Segmentation verification completed!")
    print(f"Check the images in: new_src/experiments/verify_bbox_seg/")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Verify bitmap to segmentation conversion')
    parser.add_argument('--num_images', type=int, default=2,
                       help='Number of images to verify (default: 2)')
    parser.add_argument('--split', type=str, default='train_split',
                       choices=['train_split', 'val_split', 'test_split'],
                       help='Dataset split to verify (default: train_split)')
    
    args = parser.parse_args()
    
    print(f"Bitmap to Segmentation Conversion Verification")
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
        print("Run prepare_yolo_seg_dataset.py first!")
        return
    
    verify_conversion(args.split, args.num_images)

if __name__ == "__main__":
    main()
