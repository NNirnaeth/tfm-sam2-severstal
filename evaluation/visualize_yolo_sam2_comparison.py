#!/usr/bin/env python3
"""
Visualize YOLO Detection vs YOLO+SAM2 Pipeline Results
Compare: Original Image, Ground Truth, YOLO Bboxes, SAM2 Masks

This script loads existing results from:
- YOLO detection predictions (bboxes)
- YOLO+SAM2 pipeline results (masks)
- Ground truth annotations

And creates side-by-side visualizations for comparison.
"""

import os
import sys
import numpy as np
import cv2
import json
import base64
import zlib
import argparse
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random

def set_seed(seed=42):
    """Set random seed for reproducible sampling"""
    random.seed(seed)
    np.random.seed(seed)

def load_ground_truth_annotations(ann_dir):
    """Load ground truth annotations from Supervisely format JSON files"""
    gt = {}
    for jf in Path(ann_dir).glob("*.jpg.json"):
        try:
            data = json.load(open(jf))
            objs = []
            for o in data.get("objects", []):
                bmp = o.get("bitmap", {})
                if "data" not in bmp or "origin" not in bmp:
                    continue
                raw = zlib.decompress(base64.b64decode(bmp["data"]))
                patch = np.array(Image.open(BytesIO(raw)))
                if patch.ndim == 3 and patch.shape[2] == 4:
                    patch = patch[:, :, 3]  # Alpha channel
                elif patch.ndim == 3:
                    patch = patch[:, :, 0]  # First channel
                patch = (patch > 0).astype(np.uint8)
                ox, oy = map(int, bmp["origin"])  # origin = [x, y]
                objs.append((patch, (ox, oy)))
            # Extract image name from filename
            image_name = jf.stem.replace('.jpg', '') + ".jpg"
            gt[image_name] = objs
        except Exception as e:
            print(f"Warning: Could not load {jf}: {e}")
            image_name = jf.stem.replace('.jpg', '') + ".jpg"
            gt[image_name] = []
            continue
    
    print(f"Loaded ground truth for {len(gt)} images")
    return gt

def build_gt_union(gt_objs, H, W):
    """Build ground truth union mask from patches with proper origin positioning"""
    union = np.zeros((H, W), np.uint8)
    for patch, (ox, oy) in gt_objs:
        ph, pw = patch.shape
        x1, y1 = max(0, ox), max(0, oy)
        x2, y2 = min(W, ox + pw), min(H, oy + ph)
        if x2 > x1 and y2 > y1:
            union[y1:y2, x1:x2] = np.maximum(
                union[y1:y2, x1:x2], patch[:(y2 - y1), :(x2 - x1)]
            )
    return union

def load_yolo_predictions(labels_dir):
    """Load YOLO bounding box predictions from labels directory"""
    predictions = {}
    
    for label_file in Path(labels_dir).glob("*.txt"):
        image_name = label_file.stem + ".jpg"
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5]) if len(parts) >= 6 else 1.0
                
                bboxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'conf': confidence
                })
        
        predictions[image_name] = bboxes
    
    print(f"Loaded YOLO predictions for {len(predictions)} images")
    return predictions

def load_sam2_masks(masks_dir):
    """Load SAM2 predicted masks (if saved)"""
    masks = {}
    if not os.path.exists(masks_dir):
        print(f"SAM2 masks directory not found: {masks_dir}")
        return masks
    
    for mask_file in Path(masks_dir).glob("*.png"):
        image_name = mask_file.stem + ".jpg"
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks[image_name] = mask
    
    print(f"Loaded SAM2 masks for {len(masks)} images")
    return masks

def convert_yolo_to_xyxy(bbox, img_width, img_height):
    """Convert YOLO bbox to xyxy format for visualization"""
    x_center = bbox['x_center'] * img_width
    y_center = bbox['y_center'] * img_height
    width = bbox['width'] * img_width
    height = bbox['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2

def create_comparison_visualization(image_path, gt_objs, yolo_bboxes, sam2_mask, output_path, image_name):
    """Create side-by-side comparison visualization"""
    
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return False
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]
    
    # Build ground truth mask
    gt_mask = build_gt_union(gt_objs, img_height, img_width)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'YOLO Detection vs YOLO+SAM2 Pipeline Comparison\n{image_name}', fontsize=16, fontweight='bold')
    
    # 1. Original Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Ground Truth
    axes[0, 1].imshow(image)
    # Overlay GT mask in red with transparency
    gt_overlay = np.zeros_like(image)
    gt_overlay[:, :, 0] = gt_mask * 255  # Red channel
    axes[0, 1].imshow(gt_overlay, alpha=0.6)
    axes[0, 1].set_title('Ground Truth (Red Overlay)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. YOLO Detection (Bboxes)
    axes[1, 0].imshow(image)
    # Draw YOLO bboxes
    for bbox in yolo_bboxes:
        x1, y1, x2, y2 = convert_yolo_to_xyxy(bbox, img_width, img_height)
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor='blue', facecolor='none')
        axes[1, 0].add_patch(rect)
        # Add confidence score
        conf_text = f"{bbox['conf']:.2f}"
        axes[1, 0].text(x1, y1-5, conf_text, color='blue', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    axes[1, 0].set_title(f'YOLO Detection ({len(yolo_bboxes)} bboxes)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. SAM2 Refined Masks
    axes[1, 1].imshow(image)
    if sam2_mask is not None:
        # Overlay SAM2 mask in green with transparency
        sam2_overlay = np.zeros_like(image)
        sam2_overlay[:, :, 1] = sam2_mask * 255  # Green channel
        axes[1, 1].imshow(sam2_overlay, alpha=0.6)
        axes[1, 1].set_title('SAM2 Refined Masks (Green Overlay)', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].set_title('SAM2 Masks (Not Available)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add statistics text
    stats_text = f"GT Objects: {len(gt_objs)}\nYOLO Detections: {len(yolo_bboxes)}\nGT Pixels: {np.sum(gt_mask)}"
    if sam2_mask is not None:
        stats_text += f"\nSAM2 Pixels: {np.sum(sam2_mask > 0)}"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def create_detailed_comparison(image_path, gt_objs, yolo_bboxes, sam2_mask, output_path, image_name):
    """Create detailed comparison with individual bbox analysis"""
    
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]
    
    # Build ground truth mask
    gt_mask = build_gt_union(gt_objs, img_height, img_width)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Detailed Analysis: {image_name}', fontsize=16, fontweight='bold')
    
    # 1. YOLO Bboxes with GT overlay
    axes[0].imshow(image)
    # Overlay GT mask
    gt_overlay = np.zeros_like(image)
    gt_overlay[:, :, 0] = gt_mask * 255
    axes[0].imshow(gt_overlay, alpha=0.4)
    
    # Draw YOLO bboxes with different colors based on confidence
    for i, bbox in enumerate(yolo_bboxes):
        x1, y1, x2, y2 = convert_yolo_to_xyxy(bbox, img_width, img_height)
        # Color based on confidence: green (high) to red (low)
        color = 'green' if bbox['conf'] > 0.7 else 'orange' if bbox['conf'] > 0.4 else 'red'
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor=color, facecolor='none')
        axes[0].add_patch(rect)
        # Add bbox number and confidence
        axes[0].text(x1, y1-5, f"{i+1}: {bbox['conf']:.2f}", 
                    color=color, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    axes[0].set_title('YOLO Bboxes + GT Overlay\n(Green: High Conf, Orange: Med, Red: Low)', fontsize=12)
    axes[0].axis('off')
    
    # 2. SAM2 Masks with GT overlay
    axes[1].imshow(image)
    # Overlay GT mask
    gt_overlay = np.zeros_like(image)
    gt_overlay[:, :, 0] = gt_mask * 255
    axes[1].imshow(gt_overlay, alpha=0.4)
    
    if sam2_mask is not None:
        # Overlay SAM2 mask
        sam2_overlay = np.zeros_like(image)
        sam2_overlay[:, :, 1] = sam2_mask * 255
        axes[1].imshow(sam2_overlay, alpha=0.6)
    
    axes[1].set_title('SAM2 Masks + GT Overlay\n(Red: GT, Green: SAM2)', fontsize=12)
    axes[1].axis('off')
    
    # 3. Difference analysis
    axes[2].imshow(image)
    if sam2_mask is not None:
        # Calculate differences
        gt_binary = (gt_mask > 0).astype(np.uint8)
        sam2_binary = (sam2_mask > 0).astype(np.uint8)
        
        # True Positives (both GT and SAM2)
        tp = (gt_binary & sam2_binary) * 255
        # False Positives (SAM2 but not GT)
        fp = ((~gt_binary.astype(bool)) & sam2_binary.astype(bool)) * 255
        # False Negatives (GT but not SAM2)
        fn = (gt_binary & (~sam2_binary.astype(bool))) * 255
        
        # Create colored difference map
        diff_overlay = np.zeros_like(image)
        diff_overlay[:, :, 1] = tp  # Green for TP
        diff_overlay[:, :, 0] = fn  # Red for FN
        diff_overlay[:, :, 2] = fp  # Blue for FP
        
        axes[2].imshow(diff_overlay, alpha=0.6)
        
        # Calculate metrics
        tp_count = np.sum(tp > 0)
        fp_count = np.sum(fp > 0)
        fn_count = np.sum(fn > 0)
        
        axes[2].set_title(f'Difference Analysis\nTP: {tp_count}, FP: {fp_count}, FN: {fn_count}', fontsize=12)
    else:
        axes[2].set_title('Difference Analysis\n(SAM2 masks not available)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO Detection vs YOLO+SAM2 Pipeline Results')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Subset size (e.g., 500, 1000, 2000). Uses full dataset if not specified.')
    parser.add_argument('--yolo_labels_dir', type=str, default=None,
                       help='Directory containing YOLO prediction labels (.txt files)')
    parser.add_argument('--sam2_masks_dir', type=str, default=None,
                       help='Directory containing SAM2 predicted masks (.png files)')
    parser.add_argument('--gt_annotations', type=str, default=None,
                       help='Path to ground truth annotations directory')
    parser.add_argument('--test_images_dir', type=str, default=None,
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='new_src/evaluation/visualization_results',
                       help='Directory to save visualization results')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of random samples to visualize')
    parser.add_argument('--detailed', action='store_true',
                       help='Create detailed comparison with individual bbox analysis')
    
    args = parser.parse_args()
    
    # Auto-detect paths if not provided
    if args.subset_size is not None:
        if args.yolo_labels_dir is None:
            args.yolo_labels_dir = f"new_src/evaluation/evaluation_results/yolo_detection/predict_test_corrected_subset_{args.subset_size}/labels"
        if args.sam2_masks_dir is None:
            args.sam2_masks_dir = f"new_src/evaluation/evaluation_results/yolo_sam2_pipeline/masks_subset_{args.subset_size}"
        if args.gt_annotations is None:
            args.gt_annotations = "datasets/Data/splits/test_split/ann"
        if args.test_images_dir is None:
            args.test_images_dir = f"datasets/yolo_detection_fixed/subset_{args.subset_size}/images/test_split"
    else:
        if args.yolo_labels_dir is None:
            args.yolo_labels_dir = "new_src/evaluation/evaluation_results/yolo_detection/predict_test_corrected/labels"
        if args.sam2_masks_dir is None:
            args.sam2_masks_dir = "new_src/evaluation/evaluation_results/yolo_sam2_pipeline/masks_full"
        if args.gt_annotations is None:
            args.gt_annotations = "datasets/Data/splits/test_split/ann"
        if args.test_images_dir is None:
            args.test_images_dir = "datasets/yolo_detection_fixed/images/test_split"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if required directories exist
    if not os.path.exists(args.yolo_labels_dir):
        print(f"Error: YOLO labels directory not found: {args.yolo_labels_dir}")
        return
    
    if not os.path.exists(args.gt_annotations):
        print(f"Error: Ground truth annotations directory not found: {args.gt_annotations}")
        return
    
    if not os.path.exists(args.test_images_dir):
        print(f"Error: Test images directory not found: {args.test_images_dir}")
        return
    
    print(f"\nVisualization Configuration:")
    print(f"  YOLO labels: {args.yolo_labels_dir}")
    print(f"  SAM2 masks: {args.sam2_masks_dir}")
    print(f"  Ground truth: {args.gt_annotations}")
    print(f"  Test images: {args.test_images_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Detailed analysis: {args.detailed}")
    if args.subset_size:
        print(f"  Subset size: {args.subset_size}")
    else:
        print(f"  Using full dataset")
    
    # Load data
    print(f"\nLoading data...")
    yolo_predictions = load_yolo_predictions(args.yolo_labels_dir)
    gt_annotations = load_ground_truth_annotations(args.gt_annotations)
    sam2_masks = load_sam2_masks(args.sam2_masks_dir)
    
    # Get common images
    common_images = set(yolo_predictions.keys()) & set(gt_annotations.keys())
    if sam2_masks:
        common_images = common_images & set(sam2_masks.keys())
    
    print(f"Found {len(common_images)} common images")
    
    if len(common_images) == 0:
        print("No common images found!")
        return
    
    # Sample random images
    set_seed(42)
    sample_images = random.sample(list(common_images), min(args.num_samples, len(common_images)))
    
    print(f"\nCreating visualizations for {len(sample_images)} images...")
    
    # Create visualizations
    for i, image_name in enumerate(sample_images):
        print(f"Processing {i+1}/{len(sample_images)}: {image_name}")
        
        # Get data for this image
        image_path = os.path.join(args.test_images_dir, image_name)
        gt_objs = gt_annotations.get(image_name, [])
        yolo_bboxes = yolo_predictions.get(image_name, [])
        sam2_mask = sam2_masks.get(image_name, None)
        
        # Create output filename
        base_name = os.path.splitext(image_name)[0]
        if args.detailed:
            output_filename = f"detailed_comparison_{base_name}.png"
        else:
            output_filename = f"comparison_{base_name}.png"
        
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Create visualization
        if args.detailed:
            success = create_detailed_comparison(
                image_path, gt_objs, yolo_bboxes, sam2_mask, output_path, image_name
            )
        else:
            success = create_comparison_visualization(
                image_path, gt_objs, yolo_bboxes, sam2_mask, output_path, image_name
            )
        
        if success:
            print(f"  Saved: {output_path}")
        else:
            print(f"  Failed to create visualization for {image_name}")
    
    print(f"\nVisualization completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total visualizations created: {len(sample_images)}")

if __name__ == "__main__":
    main()



