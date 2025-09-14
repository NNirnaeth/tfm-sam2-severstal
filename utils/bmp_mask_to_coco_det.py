#!/usr/bin/env python3
"""
Convert Bitmap Annotations to COCO Detection Format - CORRECTED VERSION
This script converts bitmap annotations to COCO format bounding boxes

CORRECTED: Now properly uses the 'origin' field from Supervisely bitmap format
- Fixes critical bug where bitmaps were positioned incorrectly  
- Uses cv2 for PNG decoding (same as working segmentation scripts)
- Properly positions bitmap fragments within the full image using origin coordinates

Input: Supervisely format JSON with bitmap annotations
Output: COCO detection dataset format with train/val/test splits

Usage:
    python bmpmask2cocodet.py --dataset_path datasets/Data/splits --output_path datasets/coco_detection --min_area 10
    python bmpmask2cocodet.py --dataset_path datasets/Data/splits --output_path datasets/coco_detection --visualize --max_vis 5
"""

import os
import sys
import json
import cv2
import base64
import zlib
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def convert_bitmap_to_coco_bbox(bitmap_data, origin, image_width, image_height, min_area=10):
    """
    CORRECT conversion from Supervisely bitmap to COCO bboxes
    - Uses the origin field to position bitmap correctly in full image
    - Decodes PNG bitmap using cv2 (more robust than PIL for this format)
    - Filters tiny boxes by area threshold
    - Returns COCO format: [x, y, width, height] (top-left corner + dimensions)
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
        
        # Convert to COCO format (top-left corner + width/height)
        x = float(min_x)
        y = float(min_y)
        width = float(max_x - min_x + 1)
        height = float(max_y - min_y + 1)
        
        # Ensure coordinates are within image bounds
        x = max(0, min(image_width - 1, x))
        y = max(0, min(image_height - 1, y))
        width = max(0, min(image_width - x, width))
        height = max(0, min(image_height - y, height))
        
        # Debug: Print bbox coordinates
        print(f"    Debug: Bitmap at origin ({origin_x}, {origin_y}), size {mask_width}x{mask_height}")
        print(f"    Debug: Full image bbox - x:{min_x}-{max_x}, y:{min_y}-{max_y}, area:{area}")
        print(f"    Debug: COCO - x:{x:.1f}, y:{y:.1f}, w:{width:.1f}, h:{height:.1f}")
        
        return [[x, y, width, height]]
        
    except Exception as e:
        print(f"Error converting bitmap to bbox: {e}")
        import traceback
        traceback.print_exc()
        return []

def visualize_annotations(image_path, bboxes, output_path, title="Annotations"):
    """Visualize bounding boxes on image and save to file"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return False
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # Create figure with non-interactive backend
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Draw bounding boxes
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add annotation number
            ax.text(x, y-5, f'{i+1}', fontsize=10, color='red', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title(f"{title} - {len(bboxes)} annotations")
        ax.axis('off')
        
        # Save visualization
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved: {output_path}")
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"Error visualizing annotations: {e}")
        return False

def prepare_coco_dataset(split_name, dataset_path, output_path, min_area=10, visualize=False, max_vis=5):
    """Prepare COCO detection dataset by converting bitmap annotations to COCO format"""
    print(f"Preparing {split_name} COCO detection dataset...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Paths for this split
    split_ann_dir = os.path.join(dataset_path, split_name, "ann")
    split_img_dir = os.path.join(dataset_path, split_name, "img")
    
    if not os.path.exists(split_ann_dir):
        print(f" Annotations directory not found: {split_ann_dir}")
        return None
    
    if not os.path.exists(split_img_dir):
        print(f" Images directory not found: {split_img_dir}")
        return None
    
    # Get all annotation files
    ann_files = [f for f in os.listdir(split_ann_dir) if f.endswith('.json')]
    print(f"Found {len(ann_files)} annotation files in {split_name}")
    
    # Initialize COCO dataset structure
    coco_dataset = {
        "info": {
            "description": f"Severstal Steel Defect Detection Dataset - {split_name}",
            "version": "1.0",
            "year": 2024,
            "contributor": "Severstal",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "defect",
                "supercategory": "defect"
            }
        ]
    }
    
    image_id = 1
    annotation_id = 1
    processed_count = 0
    vis_count = 0
    
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
        
        # Add image info to COCO dataset
        image_info = {
            "id": image_id,
            "file_name": image_name,
            "width": img_width,
            "height": img_height,
            "date_captured": "",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
        coco_dataset["images"].append(image_info)
        
        # Convert annotations to bboxes using corrected method
        bboxes = []
        for obj in annotation.get('objects', []):
            if 'bitmap' in obj:
                bitmap_bboxes = convert_bitmap_to_coco_bbox(
                    obj['bitmap']['data'],
                    obj['bitmap']['origin'],
                    img_width,
                    img_height,
                    min_area
                )
                bboxes.extend(bitmap_bboxes)
        
        # Add annotations to COCO dataset
        for bbox in bboxes:
            x, y, width, height = bbox
            area = width * height
            
            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # defect category
                "bbox": [x, y, width, height],
                "area": area,
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(annotation_info)
            annotation_id += 1
        
        # Visualization
        if visualize and vis_count < max_vis and bboxes:
            # Create static/visualizations directory
            vis_dir = os.path.join(output_path, "static", "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_output_path = os.path.join(vis_dir, f"visualization_{split_name}_{image_name}.png")
            if visualize_annotations(image_path, bboxes, vis_output_path, 
                                   f"{split_name} - {image_name}"):
                vis_count += 1
        
        image_id += 1
        processed_count += 1
    
    print(f"Processed {processed_count} images for {split_name}")
    return coco_dataset

def save_coco_dataset(coco_dataset, output_path, split_name):
    """Save COCO dataset to JSON file"""
    output_file = os.path.join(output_path, f"{split_name}.json")
    
    with open(output_file, 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print(f"Saved COCO dataset: {output_file}")
    return output_file

def create_dataset_info(output_path):
    """Create dataset information file"""
    info_content = {
        'dataset_name': 'Severstal Steel Defect Detection',
        'format': 'COCO Detection',
        'splits': ['train_split', 'val_split', 'test_split'],
        'categories': ['defect'],
        'description': 'COCO format detection dataset converted from Severstal bitmap annotations',
        'features': [
            'FIXED: Properly uses origin field to position bitmaps correctly',
            'Uses cv2 for PNG decoding (same as working segmentation scripts)',
            'Area-based filtering to eliminate tiny bounding boxes',
            'Correct bounding box extraction for defect detection'
        ]
    }
    
    info_path = os.path.join(output_path, "dataset_info.yaml")
    with open(info_path, 'w') as f:
        yaml.dump(info_content, f, default_flow_style=False)
    
    print(f"Created dataset info: {info_path}")
    return info_path

def main():
    parser = argparse.ArgumentParser(description='Convert bitmap annotations to COCO detection format - CORRECTED VERSION')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory containing splits (e.g., datasets/Data/splits)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory for COCO dataset')
    parser.add_argument('--min_area', type=int, default=10,
                       help='Minimum area threshold for bounding boxes (default: 10)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization images to verify annotations')
    parser.add_argument('--max_vis', type=int, default=5,
                       help='Maximum number of visualizations per split (default: 5)')
    parser.add_argument('--splits', nargs='+', default=['train_split', 'val_split', 'test_split'],
                       help='List of splits to process (default: train_split val_split test_split)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output for first few images')
    
    args = parser.parse_args()
    
    print(f"Converting bitmap annotations to COCO detection format - CORRECTED VERSION")
    print(f"Dataset directory: {args.dataset_path}")
    print(f"Output directory: {args.output_path}")
    print(f"Minimum area threshold: {args.min_area}")
    print(f"Visualization: {args.visualize}")
    print(f"Max visualizations per split: {args.max_vis}")
    print(f"Processing splits: {args.splits}")
    print(f"Debug mode: {args.debug}")
    
    # Check if source directories exist
    if not os.path.exists(args.dataset_path):
        print(f" Error: Dataset directory not found: {args.dataset_path}")
        return
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process each split
    total_images = 0
    total_annotations = 0
    
    for split_name in args.splits:
        print(f"\nProcessing {split_name}...")
        coco_dataset = prepare_coco_dataset(split_name, args.dataset_path, args.output_path, 
                                          args.min_area, args.visualize, args.max_vis)
        
        if coco_dataset:
            # Save COCO dataset
            output_file = save_coco_dataset(coco_dataset, args.output_path, split_name)
            
            # Update totals
            total_images += len(coco_dataset['images'])
            total_annotations += len(coco_dataset['annotations'])
            
            print(f"  Images: {len(coco_dataset['images'])}")
            print(f"  Annotations: {len(coco_dataset['annotations'])}")
        else:
            print(f"  No data processed for {split_name}")
    
    # Create dataset info
    info_path = create_dataset_info(args.output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"COCO dataset conversion completed!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_path}")
    print(f"Dataset info: {info_path}")
    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    
    if args.visualize:
        vis_dir = os.path.join(args.output_path, "static", "visualizations")
        print(f"\nVisualization images saved in: {vis_dir}")
    
    print(f"\nNext steps:")
    print(f"1. Verify the COCO dataset structure")
    print(f"2. Check visualization images to ensure annotations are correct")
    print(f"3. Test with COCO evaluation tools")
    print(f"4. Use with detection frameworks that support COCO format")
    print(f"\nKey features:")
    print(f"- FIXED: Now properly uses 'origin' field to position bitmaps correctly")
    print(f"- Uses cv2 for PNG decoding (same as working segmentation scripts)")  
    print(f"- Area-based filtering to eliminate tiny bounding boxes")
    print(f"- Correct bounding box extraction for defect detection")
    print(f"- Full COCO format compatibility")
    print(f"- Flexible dataset path input")
    print(f"- Optional visualization for verification")

if __name__ == "__main__":
    main()
