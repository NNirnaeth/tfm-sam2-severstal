#!/usr/bin/env python3
"""
Visual Bbox Verification Script
This script generates verification images showing:
1. Original image with bitmap masks (GT)
2. Original image with converted bounding boxes
3. Side-by-side comparison

Usage:
    python verify_bbox_visual.py --num_images 2 --split train_split
"""

import os
import sys
import json
import base64
import zlib
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import argparse
import numpy as np
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
        
        # Resize to match image dimensions if needed
        if bitmap_array.shape != (image_height, image_width):
            bitmap_image = Image.fromarray(bitmap_array * 255)
            bitmap_image = bitmap_image.resize((image_width, image_height), Image.NEAREST)
            bitmap_array = np.array(bitmap_image) > 0
        
        return bitmap_array
        
    except Exception as e:
        print(f"Error loading bitmap mask: {e}")
        return None

def convert_bitmap_to_bbox(bitmap_data, origin, image_width, image_height, min_area=10):
    """Convert bitmap to YOLO bbox (CORRECTED - uses origin field)"""
    try:
        origin_x, origin_y = origin[0], origin[1]
        
        # Decode bitmap from base64 and zlib
        decoded_data = base64.b64decode(bitmap_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Use cv2 to decode PNG (same as working scripts)
        import cv2
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
        
        return [[x_center, y_center, width, height, min_x, min_y, max_x, max_y]]
        
    except Exception as e:
        print(f"Error converting bitmap to bbox: {e}")
        return []

def yolo_to_xyxy(yolo_bbox, img_width, img_height):
    """Convert YOLO format to xyxy format for visualization"""
    x_center, y_center, width, height = yolo_bbox[:4]
    
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

def create_verification_images(image, bitmap_masks, converted_bboxes, yolo_bboxes, image_name, output_dir):
    """Create verification images showing GT vs predictions"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PIL image to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_width, img_height = image.size
    
    # 1. Original image with bitmap masks (GT)
    gt_image = image.copy()
    gt_draw = ImageDraw.Draw(gt_image)
    
    for i, mask in enumerate(bitmap_masks):
        # Create colored overlay for mask
        mask_overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_overlay)
        
        # Find mask pixels and draw them
        mask_pixels = np.where(mask > 0)
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
    gt_draw.text((10, 10), "Ground Truth (Bitmap Masks)", fill=(255, 255, 255), font=font)
    gt_draw.text((10, 35), f"Masks: {len(bitmap_masks)}", fill=(255, 255, 255), font=font)
    
    # 2. Original image with converted bboxes
    bbox_image = image.copy()
    bbox_draw = ImageDraw.Draw(bbox_image)
    
    for i, bbox in enumerate(converted_bboxes):
        x1, y1, x2, y2 = yolo_to_xyxy(bbox, img_width, img_height)
        bbox_draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        bbox_draw.text((x1, y1-25), f"Bbox {i+1}", fill=(0, 255, 0), font=font)
    
    # Add title
    bbox_draw.text((10, 10), "Converted Bounding Boxes", fill=(255, 255, 255), font=font)
    bbox_draw.text((10, 35), f"Boxes: {len(converted_bboxes)}", fill=(255, 255, 255), font=font)
    
    # 3. Original image with YOLO labels
    yolo_image = image.copy()
    yolo_draw = ImageDraw.Draw(yolo_image)
    
    for i, bbox in enumerate(yolo_bboxes):
        x1, y1, x2, y2 = yolo_to_xyxy(bbox, img_width, img_height)
        yolo_draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=3)
        yolo_draw.text((x1, y1-25), f"YOLO {i+1}", fill=(0, 0, 255), font=font)
    
    # Add title
    yolo_draw.text((10, 10), "YOLO Labels", fill=(255, 255, 255), font=font)
    yolo_draw.text((10, 35), f"Labels: {len(yolo_bboxes)}", fill=(255, 255, 255), font=font)
    
    # 4. Side-by-side comparison
    comparison_width = img_width * 2
    comparison_height = img_height + 100  # Extra space for titles
    
    comparison_image = Image.new('RGB', (comparison_width, comparison_height), (0, 0, 0))
    comparison_draw = ImageDraw.Draw(comparison_image)
    
    # Paste GT image
    comparison_image.paste(gt_image, (0, 50))
    
    # Paste bbox image
    comparison_image.paste(bbox_image, (img_width, 50))
    
    # Add main title
    comparison_draw.text((10, 10), f"Verification: {image_name}", fill=(255, 255, 255), font=font)
    comparison_draw.text((10, 30), f"Image size: {img_width}x{img_height}", fill=(255, 255, 255), font=font)
    
    # Save all images
    gt_path = os.path.join(output_dir, f"{image_name}_gt.png")
    bbox_path = os.path.join(output_dir, f"{image_name}_bboxes.png")
    yolo_path = os.path.join(output_dir, f"{image_name}_yolo.png")
    comparison_path = os.path.join(output_dir, f"{image_name}_comparison.png")
    
    gt_image.save(gt_path)
    bbox_image.save(bbox_path)
    yolo_image.save(yolo_path)
    comparison_image.save(comparison_path)
    
    return {
        'gt': gt_path,
        'bboxes': bbox_path,
        'yolo': yolo_path,
        'comparison': comparison_path
    }

def verify_conversion_visual(split_name, num_images=2):
    """Visual verification of bitmap to bbox conversion"""
    print(f"Visual verification for {split_name} split...")
    
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
    
    # Create output directory
    output_dir = "new_src/experiments/verify_bbox_detect"
    os.makedirs(output_dir, exist_ok=True)
    
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
                    obj['bitmap']['origin'],
                    img_width,
                    img_height
                )
                converted_bboxes.extend(bboxes)
        
        print(f"Found {len(bitmap_masks)} bitmap masks")
        print(f"Converted to {len(converted_bboxes)} bounding boxes")
        print(f"YOLO labels contain {len(yolo_bboxes)} bounding boxes")
        
        # Create verification images
        image_paths = create_verification_images(
            image, bitmap_masks, converted_bboxes, yolo_bboxes, 
            image_name, output_dir
        )
        
        print(f"\nVerification images saved:")
        print(f"  Ground Truth: {image_paths['gt']}")
        print(f"  Converted Bboxes: {image_paths['bboxes']}")
        print(f"  YOLO Labels: {image_paths['yolo']}")
        print(f"  Comparison: {image_paths['comparison']}")
        
        # Print detailed bbox information
        print(f"\nDetailed bbox information:")
        for j, bbox in enumerate(converted_bboxes):
            x_center, y_center, width, height, min_x, min_y, max_x, max_y = bbox
            print(f"  Bbox {j+1}: ({min_x}, {min_y}) -> ({max_x}, {max_y}) [size: {max_x-min_x+1}x{max_y-min_y+1}]")
            print(f"    YOLO: center=({x_center:.6f}, {y_center:.6f}), size=({width:.6f}, {height:.6f})")
        
        # Check if conversion matches YOLO labels
        if len(converted_bboxes) == len(yolo_bboxes):
            print(f"\n✓ Number of bboxes matches!")
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
    
    print(f"\n{'='*80}")
    print(f"Visual verification completed!")
    print(f"Check the images in: new_src/experiments/verify_bbox_detect/")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Visual bbox verification')
    parser.add_argument('--num_images', type=int, default=2,
                       help='Number of images to verify (default: 2)')
    parser.add_argument('--split', type=str, default='train_split',
                       choices=['train_split', 'val_split', 'test_split'],
                       help='Dataset split to verify (default: train_split)')
    
    args = parser.parse_args()
    
    print(f"Visual Bitmap to Bbox Verification")
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
    
    verify_conversion_visual(args.split, args.num_images)

if __name__ == "__main__":
    main()
