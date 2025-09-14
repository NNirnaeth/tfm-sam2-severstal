#!/usr/bin/env python3
"""
Debug script for YOLO+SAM2 pipeline
Check each step manually on 2-3 images to identify issues

This script:
1. Loads YOLO predictions and visualizes bboxes
2. Converts bboxes to SAM2 format and generates masks
3. Loads GT annotations (YOLO format) and compares with predictions
4. Provides detailed numerical checks for debugging
"""

import os
import sys
import numpy as np
import torch
import argparse
import json
import cv2
import base64
import zlib
from PIL import Image
from io import BytesIO
from datetime import datetime
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add new_src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import SAM2 utilities
import sys
sys.path.append('/home/ptp/sam2/libs/sam2base')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sam2_model(checkpoint_path, model_type="sam2_hiera_l"):
    """Load SAM2 fine-tuned model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SAM2 model from: {checkpoint_path}")
    print(f"Model type: {model_type}")
    
    # Use the correct config file path
    config_file = f"configs/sam2/{model_type}.yaml"
    
    sam2 = build_sam2(config_file=config_file, ckpt_path=checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2)
    
    return sam2, predictor, device


def load_yolo_predictions(labels_dir):
    """Load YOLO bounding box predictions from labels directory"""
    predictions = {}
    
    # YOLO format: class x_center y_center width height (normalized 0-1)
    for label_file in Path(labels_dir).glob("*.txt"):
        image_name = label_file.stem + ".jpg"  # Assuming .jpg extension
        
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
                
                # Also load confidence if available (6th column)
                confidence = float(parts[5]) if len(parts) > 5 else 1.0
                
                bboxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': confidence
                })
        
        predictions[image_name] = bboxes
    
    print(f"Loaded predictions for {len(predictions)} images")
    return predictions


def load_ground_truth_annotations(annotations_dir):
    """Load ground truth annotations from YOLO format .txt files"""
    gt_annotations = {}
    
    # Process each .txt file in the annotations directory
    for txt_file in Path(annotations_dir).glob("*.txt"):
        image_name = txt_file.stem + ".jpg"  # Assuming .jpg extension
        
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            masks = []
            if lines:  # If file has content
                # For YOLO format, we need to convert bboxes to binary masks
                # This is a simplified approach - in practice you might want to use the original image dimensions
                # For now, we'll create placeholder masks based on bbox coordinates
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Store bbox info for later mask generation
                        masks.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'type': 'bbox'  # Mark as bbox format
                        })
            
            # Always store GT (even if empty masks list)
            gt_annotations[image_name] = masks
        
        except Exception as e:
            print(f"Warning: Could not load {txt_file}: {e}")
            # Still create entry with empty masks for this image
            gt_annotations[image_name] = []
            continue
    
    print(f"Loaded ground truth for {len(gt_annotations)} images")
    return gt_annotations


def convert_bbox_to_sam2_format(bbox, img_width, img_height):
    """Convert YOLO bbox to SAM2 input format"""
    # YOLO format: normalized x_center, y_center, width, height
    x_center = bbox['x_center'] * img_width
    y_center = bbox['y_center'] * img_height
    width = bbox['width'] * img_width
    height = bbox['height'] * img_height
    
    # SAM2 expects: [x1, y1, x2, y2] in pixel coordinates
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    return [x1, y1, x2, y2]


def generate_sam2_masks(predictor, image, bboxes, threshold=0.5):
    """Generate SAM2 masks using bounding boxes as prompts"""
    # Set image in predictor
    predictor.set_image(image)
    
    masks = []
    scores = []
    probs = []
    
    for bbox in bboxes:
        # Convert bbox to SAM2 format (pixel coordinates)
        input_box = convert_bbox_to_sam2_format(bbox, image.shape[1], image.shape[0])
        
        print(f"  SAM2 input box: {input_box}")
        
        # Generate mask with SAM2
        masks_pred, scores_pred, logits_pred = predictor.predict(
            box=input_box,
            multimask_output=True  # Get multiple masks to select best
        )
        
        print(f"  SAM2 returned {len(masks_pred)} masks")
        print(f"  SAM2 scores: {scores_pred}")
        
        # Select mask with highest score
        best_idx = np.argmax(scores_pred)
        best_mask = masks_pred[best_idx]
        best_score = scores_pred[best_idx]
        best_logits = logits_pred[best_idx] if logits_pred is not None else None
        
        print(f"  Selected mask {best_idx} with score {best_score:.4f}")
        print(f"  Mask shape: {best_mask.shape}, dtype: {best_mask.dtype}")
        print(f"  Mask sum: {np.sum(best_mask)}")
        
        # Get probability map from logits
        if best_logits is not None:
            # Convert logits to probabilities using sigmoid
            prob = torch.sigmoid(torch.from_numpy(best_logits)).numpy()
            # Resize to image dimensions if needed
            if prob.shape != image.shape[:2]:
                prob = cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback to binary mask as probability
            prob = best_mask.astype(np.float32)
        
        masks.append(best_mask)
        scores.append(best_score)
        probs.append(prob)
    
    return masks, scores, probs


def build_gt_union(gt_masks, img_height, img_width):
    """Build ground truth union mask from list of YOLO bboxes"""
    gt_merged = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for i, gt_bbox in enumerate(gt_masks):
        if gt_bbox.get('type') == 'bbox':
            # Convert YOLO bbox to pixel coordinates
            x_center = gt_bbox['x_center'] * img_width
            y_center = gt_bbox['y_center'] * img_height
            width = gt_bbox['width'] * img_width
            height = gt_bbox['height'] * img_height
            
            # Convert to [x1, y1, x2, y2] format
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Create binary mask from bbox
            gt_merged[y1:y2, x1:x2] = 1
            
            print(f"  GT bbox {i}: normalized ({gt_bbox['x_center']:.4f}, {gt_bbox['y_center']:.4f}, {gt_bbox['width']:.4f}, {gt_bbox['height']:.4f})")
            print(f"    Pixels: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"    Box size: {x2-x1}x{y2-y1}")
        else:
            # Handle other mask types if needed
            print(f"  GT mask {i}: shape {gt_bbox.shape}, sum {np.sum(gt_bbox)}")
            # Resize GT mask to match image size if needed
            if gt_bbox.shape != (img_height, img_width):
                gt_bbox = cv2.resize(gt_bbox.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                print(f"  Resized GT mask {i}: shape {gt_bbox.shape}, sum {np.sum(gt_bbox)}")
            gt_merged = np.logical_or(gt_merged, gt_bbox.astype(bool))
    
    gt_merged = gt_merged.astype(np.uint8)
    print(f"  Final GT union: shape {gt_merged.shape}, sum {np.sum(gt_merged)}")
    return gt_merged


def calculate_iou(mask1, mask2):
    """Calculate IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    
    iou = intersection_sum / union_sum if union_sum > 0 else 0.0
    return iou, intersection_sum, union_sum


def visualize_debug(image, yolo_bboxes, sam2_masks, gt_mask, image_name, save_dir):
    """Create debug visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Debug Visualization: {image_name}', fontsize=16)
    
    # Original image with YOLO bboxes
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image + YOLO Bboxes')
    axes[0, 0].axis('off')
    
    # Draw YOLO bboxes
    for bbox in yolo_bboxes:
        x1, y1, x2, y2 = convert_bbox_to_sam2_format(bbox, image.shape[1], image.shape[0])
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(rect)
        axes[0, 0].text(x1, y1-5, f"conf: {bbox['confidence']:.3f}", 
                        color='red', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    # SAM2 masks overlay
    axes[0, 1].imshow(image)
    axes[0, 1].set_title('SAM2 Masks Overlay')
    axes[0, 1].axis('off')
    
    if sam2_masks:
        # Overlay all masks with different colors
        for i, mask in enumerate(sam2_masks):
            mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            mask_rgb[mask > 0] = [255, 0, 0]  # Red for masks
            axes[0, 1].imshow(mask_rgb, alpha=0.5)
    
    # Ground truth mask
    axes[1, 0].imshow(gt_mask, cmap='gray')
    axes[1, 0].set_title('Ground Truth Mask')
    axes[1, 0].axis('off')
    
    # Combined visualization
    axes[1, 1].imshow(image)
    axes[1, 1].set_title('Combined: Image + YOLO + SAM2 + GT')
    axes[1, 1].axis('off')
    
    # Draw YOLO bboxes
    for bbox in yolo_bboxes:
        x1, y1, x2, y2 = convert_bbox_to_sam2_format(bbox, image.shape[1], image.shape[0])
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[1, 1].add_patch(rect)
    
    # Overlay SAM2 masks
    if sam2_masks:
        for mask in sam2_masks:
            mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            mask_rgb[mask > 0] = [0, 255, 0]  # Green for SAM2
            axes[1, 1].imshow(mask_rgb, alpha=0.5)
    
    # Overlay GT mask
    gt_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    gt_rgb[gt_mask > 0] = [0, 0, 255]  # Blue for GT
    axes[1, 1].imshow(gt_rgb, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    save_path = os.path.join(save_dir, f"debug_{image_name.replace('.jpg', '')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug visualization saved to: {save_path}")


def debug_single_image(image_path, yolo_bboxes, gt_masks, sam2_predictor, image_name, save_dir):
    """Debug a single image step by step"""
    print(f"\n{'='*60}")
    print(f"DEBUGGING IMAGE: {image_name}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]
    
    print(f"Image loaded: {img_height}x{img_width}")
    print(f"YOLO bboxes found: {len(yolo_bboxes)}")
    
    # Check YOLO bboxes
    if yolo_bboxes:
        print(f"\nYOLO BBOXES:")
        for i, bbox in enumerate(yolo_bboxes):
            print(f"  Bbox {i}: class={bbox['class_id']}, conf={bbox['confidence']:.4f}")
            print(f"    Normalized: cx={bbox['x_center']:.4f}, cy={bbox['y_center']:.4f}, w={bbox['width']:.4f}, h={bbox['height']:.4f}")
            
            # Convert to pixels
            x1, y1, x2, y2 = convert_bbox_to_sam2_format(bbox, img_width, img_height)
            print(f"    Pixels: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"    Box size: {x2-x1}x{y2-y1}")
            
            # Check if bbox is reasonable
            if x2 <= x1 or y2 <= y1:
                print(f"    WARNING: Invalid bbox dimensions!")
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                print(f"    WARNING: Bbox outside image bounds!")
    else:
        print("  No YOLO bboxes found")
    
    # Check GT masks
    print(f"\nGROUND TRUTH:")
    if gt_masks:
        print(f"  Found {len(gt_masks)} GT masks")
        gt_merged = build_gt_union(gt_masks, img_height, img_width)
        print(f"  GT union mask: shape {gt_merged.shape}, sum {np.sum(gt_merged)}")
        
        if np.sum(gt_merged) == 0:
            print(f"  WARNING: GT mask has no positive pixels!")
    else:
        print(f"  No GT masks found")
        gt_merged = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Generate SAM2 masks if we have bboxes
    sam2_masks = []
    if yolo_bboxes:
        print(f"\nSAM2 MASK GENERATION:")
        try:
            sam2_masks, scores, probs = generate_sam2_masks(sam2_predictor, image, yolo_bboxes)
            print(f"  Generated {len(sam2_masks)} SAM2 masks")
            
            # Check mask quality
            for i, mask in enumerate(sam2_masks):
                print(f"    Mask {i}: shape {mask.shape}, sum {np.sum(mask)}, score {scores[i]:.4f}")
                
                if np.sum(mask) == 0:
                    print(f"      WARNING: Empty mask!")
                elif np.sum(mask) == mask.shape[0] * mask.shape[1]:
                    print(f"      WARNING: Full image mask!")
                
        except Exception as e:
            print(f"  ERROR generating SAM2 masks: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  Skipping SAM2 (no bboxes)")
    
    # Calculate metrics if we have both predictions and GT
    if sam2_masks and np.sum(gt_merged) > 0:
        print(f"\nMETRICS CALCULATION:")
        
        # Merge SAM2 masks
        if sam2_masks:
            merged_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            for mask in sam2_masks:
                merged_mask = np.logical_or(merged_mask, mask)
            merged_mask = merged_mask.astype(np.uint8)
            
            print(f"  Merged SAM2 mask: shape {merged_mask.shape}, sum {np.sum(merged_mask)}")
            
            # Calculate IoU
            iou, intersection, union = calculate_iou(merged_mask, gt_merged)
            print(f"  IoU: {iou:.4f} (intersection: {intersection}, union: {union})")
            
            # Calculate Dice
            dice = 2 * intersection / (np.sum(merged_mask) + np.sum(gt_merged)) if (np.sum(merged_mask) + np.sum(gt_merged)) > 0 else 0
            print(f"  Dice: {dice:.4f}")
        else:
            print(f"  No SAM2 masks to evaluate")
    else:
        print(f"\nMETRICS CALCULATION:")
        if not sam2_masks:
            print(f"  No SAM2 masks to evaluate")
        if np.sum(gt_merged) == 0:
            print(f"  No GT pixels to compare against")
    
    # Create visualization
    try:
        visualize_debug(image, yolo_bboxes, sam2_masks, gt_merged, image_name, save_dir)
    except Exception as e:
        print(f"  ERROR creating visualization: {e}")
    
    return {
        'image_name': image_name,
        'yolo_bboxes': yolo_bboxes,
        'gt_masks': gt_masks,
        'sam2_masks': sam2_masks,
        'gt_merged': gt_merged if 'gt_merged' in locals() else None
    }


def main():
    parser = argparse.ArgumentParser(description='Debug YOLO+SAM2 pipeline step by step')
    parser.add_argument('--yolo_labels_dir', type=str, required=True,
                       help='Directory containing YOLO prediction labels (.txt files)')
    parser.add_argument('--sam2_checkpoint', type=str, required=True,
                       help='Path to SAM2 fine-tuned checkpoint')
    parser.add_argument('--sam2_model_type', type=str, default='sam2_hiera_l',
                       choices=['sam2_hiera_l', 'sam2_hiera_b', 'sam2_hiera_t'],
                       help='SAM2 model type')
    parser.add_argument('--gt_annotations', type=str, required=True,
                       help='Path to ground truth annotations directory (YOLO format .txt files)')
    parser.add_argument('--test_images_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, 
                       default='new_src/evaluation/debug_results',
                       help='Directory to save debug results')
    parser.add_argument('--num_images', type=int, default=3,
                       help='Number of images to debug')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if required files exist
    if not os.path.exists(args.yolo_labels_dir):
        print(f"Error: YOLO labels directory not found: {args.yolo_labels_dir}")
        return
    
    if not os.path.exists(args.sam2_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.sam2_checkpoint}")
        return
    
    if not os.path.exists(args.gt_annotations):
        print(f"Error: Ground truth annotations directory not found: {args.gt_annotations}")
        return
    
    if not os.path.exists(args.test_images_dir):
        print(f"Error: Test images directory not found: {args.test_images_dir}")
        return
    
    # Load SAM2 model
    print(f"Loading SAM2 model...")
    try:
        sam, predictor, device = load_sam2_model(args.sam2_checkpoint, args.sam2_model_type)
    except Exception as e:
        print(f"ERROR loading SAM2 model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load YOLO predictions
    print(f"Loading YOLO predictions...")
    yolo_predictions = load_yolo_predictions(args.yolo_labels_dir)
    
    # Load ground truth
    print(f"Loading ground truth annotations...")
    gt_annotations = load_ground_truth_annotations(args.gt_annotations)
    
    # Select images for debugging
    print(f"\nSelecting images for debugging...")
    
    # Find images with detections
    images_with_detections = [img for img, bboxes in yolo_predictions.items() if len(bboxes) > 0]
    images_without_detections = [img for img, bboxes in yolo_predictions.items() if len(bboxes) == 0]
    
    print(f"Images with detections: {len(images_with_detections)}")
    print(f"Images without detections: {len(images_without_detections)}")
    
    # Select debug images
    debug_images = []
    
    # Add 2 images with detections
    if len(images_with_detections) >= 2:
        debug_images.extend(images_with_detections[:2])
    elif len(images_with_detections) == 1:
        debug_images.append(images_with_detections[0])
    
    # Add 1 image without detections
    if len(images_without_detections) >= 1:
        debug_images.append(images_without_detections[0])
    
    # If we don't have enough, add more from either category
    while len(debug_images) < args.num_images and len(debug_images) < len(yolo_predictions):
        remaining = [img for img in yolo_predictions.keys() if img not in debug_images]
        if remaining:
            debug_images.append(remaining[0])
        else:
            break
    
    print(f"Selected {len(debug_images)} images for debugging:")
    for img in debug_images:
        bbox_count = len(yolo_predictions.get(img, []))
        gt_count = len(gt_annotations.get(img, []))
        print(f"  {img}: {bbox_count} YOLO bboxes, {gt_count} GT masks")
    
    # Debug each image
    debug_results = []
    for image_name in debug_images:
        image_path = os.path.join(args.test_images_dir, image_name)
        yolo_bboxes = yolo_predictions.get(image_name, [])
        gt_masks = gt_annotations.get(image_name, [])
        
        result = debug_single_image(
            image_path, yolo_bboxes, gt_masks, predictor, 
            image_name, args.output_dir
        )
        
        if result:
            debug_results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DEBUG SUMMARY")
    print(f"{'='*60}")
    
    for result in debug_results:
        image_name = result['image_name']
        yolo_count = len(result['yolo_bboxes'])
        gt_count = len(result['gt_masks'])
        sam2_count = len(result['sam2_masks']) if result['sam2_masks'] else 0
        
        print(f"\n{image_name}:")
        print(f"  YOLO bboxes: {yolo_count}")
        print(f"  GT masks: {gt_count}")
        print(f"  SAM2 masks: {sam2_count}")
        
        if result['gt_merged'] is not None:
            gt_pixels = np.sum(result['gt_merged'])
            print(f"  GT total pixels: {gt_pixels}")
        
        if sam2_count > 0:
            # Calculate metrics if possible
            if result['gt_merged'] is not None and gt_pixels > 0:
                merged_mask = np.zeros_like(result['gt_merged'])
                for mask in result['sam2_masks']:
                    merged_mask = np.logical_or(merged_mask, mask)
                merged_mask = merged_mask.astype(np.uint8)
                
                iou, intersection, union = calculate_iou(merged_mask, result['gt_merged'])
                print(f"  IoU: {iou:.4f}")
    
    print(f"\nDebug results saved to: {args.output_dir}")
    print(f"Check the generated PNG files for visual debugging")


if __name__ == "__main__":
    main()
