#!/usr/bin/env python3
"""
Verify YOLO annotation conversion by visualizing bounding boxes and segmentation masks

This script verifies that the conversion from Severstal format to YOLO format
is working correctly by visualizing original images with:
1. Ground truth masks from original annotations
2. YOLO polygon overlays from converted annotations
3. Bounding boxes derived from YOLO polygons

Saves verification images in each subset directory for easy inspection.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import base64
import zlib
import io
import cv2
from pathlib import Path

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

def get_bbox_from_polygon(polygon_points):
    """Get bounding box from polygon points"""
    if len(polygon_points) == 0:
        return None
    
    x_coords = [p[0] for p in polygon_points]
    y_coords = [p[1] for p in polygon_points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]  # [x, y, width, height]

def visualize_annotations(image_path, annotation_path, yolo_label_path, output_path):
    """Visualize original annotation vs YOLO conversion with bboxes"""
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]
    
    # Load original annotation and create ground truth mask
    gt_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    gt_bboxes = []
    
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            ann_data = json.load(f)
        
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
                            # Get bounding box from mask
                            if np.any(mask):
                                coords = np.column_stack(np.where(mask))
                                if len(coords) > 0:
                                    mask_y_min, mask_x_min = coords.min(axis=0)
                                    mask_y_max, mask_x_max = coords.max(axis=0)
                                    # Convert to image coordinates
                                    bbox_x = origin_x + mask_x_min
                                    bbox_y = origin_y + mask_y_min
                                    bbox_w = mask_x_max - mask_x_min + 1
                                    bbox_h = mask_y_max - mask_y_min + 1
                                    gt_bboxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
    
    # Load YOLO labels
    yolo_polygons = []
    yolo_bboxes = []
    
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
                    
                    if len(polygon) > 2:
                        yolo_polygons.append(np.array(polygon, dtype=np.float32))
                        # Get bounding box from polygon
                        bbox = get_bbox_from_polygon(polygon)
                        if bbox:
                            yolo_bboxes.append(bbox)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Annotation Verification: {os.path.basename(image_path)}', fontsize=16)
    
    # Original image
    axes[0,0].imshow(img_array)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # Ground truth mask + bboxes
    axes[0,1].imshow(img_array)
    if gt_mask.any():
        axes[0,1].imshow(gt_mask, alpha=0.4, cmap='Reds')
    
    # Draw ground truth bboxes
    for bbox in gt_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                               linewidth=2, edgecolor='red', facecolor='none', 
                               linestyle='--', label='GT BBox')
        axes[0,1].add_patch(rect)
    
    axes[0,1].set_title(f'Ground Truth Mask + BBoxes ({len(gt_bboxes)} boxes)')
    axes[0,1].axis('off')
    
    # YOLO polygons
    axes[1,0].imshow(img_array)
    for i, polygon in enumerate(yolo_polygons):
        if len(polygon) > 2:
            # Close the polygon
            closed_polygon = np.vstack([polygon, polygon[0]])
            axes[1,0].plot(closed_polygon[:, 0], closed_polygon[:, 1], 
                          'b-', linewidth=2, alpha=0.8)
            axes[1,0].fill(polygon[:, 0], polygon[:, 1], 'blue', alpha=0.2)
    
    axes[1,0].set_title(f'YOLO Polygons ({len(yolo_polygons)} polygons)')
    axes[1,0].axis('off')
    
    # YOLO bboxes comparison
    axes[1,1].imshow(img_array)
    
    # Draw YOLO bboxes
    for bbox in yolo_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                               linewidth=2, edgecolor='blue', facecolor='none', 
                               label='YOLO BBox')
        axes[1,1].add_patch(rect)
    
    # Draw ground truth bboxes for comparison
    for bbox in gt_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                               linewidth=2, edgecolor='red', facecolor='none', 
                               linestyle='--', alpha=0.7, label='GT BBox')
        axes[1,1].add_patch(rect)
    
    axes[1,1].set_title(f'BBox Comparison: YOLO (blue) vs GT (red dashed)')
    axes[1,1].axis('off')
    
    # Add legend
    if gt_bboxes or yolo_bboxes:
        handles = []
        if gt_bboxes:
            handles.append(patches.Patch(color='red', alpha=0.7, label='Ground Truth BBox'))
        if yolo_bboxes:
            handles.append(patches.Patch(color='blue', alpha=0.7, label='YOLO BBox'))
        axes[1,1].legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print comparison stats
    print(f"  GT masks: {1 if gt_mask.any() else 0}, GT bboxes: {len(gt_bboxes)}")
    print(f"  YOLO polygons: {len(yolo_polygons)}, YOLO bboxes: {len(yolo_bboxes)}")
    
    return len(gt_bboxes), len(yolo_bboxes)

def verify_subset(subset_name, converted_dir, original_dir, num_samples=2):
    """Verify a specific subset"""
    print(f"\n{'='*60}")
    print(f"VERIFYING SUBSET: {subset_name}")
    print(f"{'='*60}")
    
    # Get paths
    if subset_name == "full":
        converted_subset_dir = os.path.join(converted_dir, "full_dataset")
        original_img_dir = os.path.join(original_dir, "train_split", "img")
        original_ann_dir = os.path.join(original_dir, "train_split", "ann")
    else:
        converted_subset_dir = os.path.join(converted_dir, f"subset_{subset_name}")
        original_img_dir = os.path.join(original_dir, "subsets", f"{subset_name}_subset", "img")
        original_ann_dir = os.path.join(original_dir, "subsets", f"{subset_name}_subset", "ann")
    
    yolo_img_dir = os.path.join(converted_subset_dir, "images", "train")
    yolo_label_dir = os.path.join(converted_subset_dir, "labels", "train")
    
    # Create verification output directory
    verification_dir = os.path.join(converted_subset_dir, "verification_plots")
    os.makedirs(verification_dir, exist_ok=True)
    
    # Verify directories exist
    for directory in [original_img_dir, original_ann_dir, yolo_img_dir, yolo_label_dir]:
        if not os.path.exists(directory):
            print(f"ERROR: Directory not found: {directory}")
            return False
    
    print(f"Original images: {original_img_dir}")
    print(f"Original annotations: {original_ann_dir}")
    print(f"YOLO images: {yolo_img_dir}")
    print(f"YOLO labels: {yolo_label_dir}")
    print(f"Verification plots: {verification_dir}")
    
    # Get sample images (prefer images with annotations)
    all_images = [f for f in os.listdir(original_img_dir) if f.endswith('.jpg')]
    
    # Find images with annotations
    images_with_ann = []
    images_without_ann = []
    
    for img_file in all_images[:50]:  # Check first 50 images
        ann_file = img_file + '.json'
        ann_path = os.path.join(original_ann_dir, ann_file)
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                if ann_data.get('objects', []):
                    images_with_ann.append(img_file)
                else:
                    images_without_ann.append(img_file)
            except:
                images_without_ann.append(img_file)
        else:
            images_without_ann.append(img_file)
    
    # Select samples: prefer images with annotations
    sample_images = images_with_ann[:num_samples]
    if len(sample_images) < num_samples:
        sample_images.extend(images_without_ann[:num_samples - len(sample_images)])
    
    print(f"Found {len(images_with_ann)} images with annotations, {len(images_without_ann)} without")
    print(f"Visualizing {len(sample_images)} samples...")
    
    total_gt_boxes = 0
    total_yolo_boxes = 0
    
    for i, img_file in enumerate(sample_images):
        print(f"\nProcessing {i+1}/{len(sample_images)}: {img_file}")
        
        # Paths
        image_path = os.path.join(original_img_dir, img_file)
        annotation_path = os.path.join(original_ann_dir, img_file + '.json')
        yolo_label_path = os.path.join(yolo_label_dir, img_file.replace('.jpg', '.txt'))
        output_path = os.path.join(verification_dir, f"verification_{subset_name}_{img_file.replace('.jpg', '.png')}")
        
        # Visualize
        gt_boxes, yolo_boxes = visualize_annotations(image_path, annotation_path, yolo_label_path, output_path)
        total_gt_boxes += gt_boxes
        total_yolo_boxes += yolo_boxes
        
        print(f"  Saved: {output_path}")
    
    print(f"\n{'-'*40}")
    print(f"SUBSET {subset_name} SUMMARY:")
    print(f"  Total GT boxes: {total_gt_boxes}")
    print(f"  Total YOLO boxes: {total_yolo_boxes}")
    print(f"  Match rate: {total_yolo_boxes/max(total_gt_boxes, 1)*100:.1f}%")
    print(f"  Verification plots saved in: {verification_dir}")
    print(f"{'-'*40}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify YOLO annotation conversion")
    parser.add_argument("--subset", type=str, choices=["500", "1000", "2000", "full"], 
                       help="Dataset subset to verify")
    parser.add_argument("--all-subsets", action="store_true",
                       help="Verify all subsets (500, 1000, 2000)")
    parser.add_argument("--converted-dir", type=str, default="/home/ptp/sam2/datasets/yolo_segmentation_subsets",
                       help="Directory containing converted YOLO datasets")
    parser.add_argument("--original-dir", type=str, default="datasets/Data/splits",
                       help="Directory containing original annotations")
    parser.add_argument("--num-samples", type=int, default=3,
                       help="Number of samples to visualize per subset")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.subset and not args.all_subsets:
        parser.error("Must specify either --subset or --all-subsets")
    
    print("YOLO Annotation Verification")
    print("=" * 50)
    print(f"Converted datasets: {args.converted_dir}")
    print(f"Original annotations: {args.original_dir}")
    print(f"Samples per subset: {args.num_samples}")
    
    # Determine subsets to verify
    if args.all_subsets:
        subsets = ["500", "1000", "2000"]
        print("Verifying subsets: 500, 1000, 2000")
    else:
        subsets = [args.subset]
        print(f"Verifying subset: {args.subset}")
    
    print("-" * 50)
    
    # Verify each subset
    success_count = 0
    for subset in subsets:
        if verify_subset(subset, args.converted_dir, args.original_dir, args.num_samples):
            success_count += 1
        else:
            print(f"✗ Failed to verify subset {subset}")
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETED")
    print(f"Successfully verified: {success_count}/{len(subsets)} subsets")
    print(f"Check verification plots in each subset directory:")
    for subset in subsets:
        if subset == "full":
            subset_dir = os.path.join(args.converted_dir, "full_dataset", "verification_plots")
        else:
            subset_dir = os.path.join(args.converted_dir, f"subset_{subset}", "verification_plots")
        print(f"  - {subset_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

