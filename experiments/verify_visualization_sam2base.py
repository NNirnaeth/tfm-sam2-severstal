#!/usr/bin/env python3
"""
Script para visualizar y comparar predicciones de SAM2 base
en diferentes modos de evaluación (auto_prompt, 3 puntos GT, 30 puntos GT)
para modelos Small y Large.

Genera visualizaciones lado a lado para comparar:
- Ground Truth vs Predicción
- Diferentes modos de evaluación
- Modelos Small vs Large
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
from PIL import Image
import base64
import zlib
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random
import argparse

# Add paths
sys.path.append('/home/ptp/sam2/libs/sam2base')
sys.path.append('/home/ptp/sam2/new_src/utils')
from metrics import SegmentationMetrics

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Constants
CONFIDENCE_THRESHOLD = 0.3
POINT_BATCH_SIZE = 4
BOX_BATCH_SIZE = 8
GRID_SPACING = 128
BOX_SCALES = [128, 256]
NMS_IOU_THRESHOLD = 0.65
TOP_K_FILTER = 200

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_binary_gt_mask(ann_path, target_shape):
    """
    Build a single binary GT mask for one image from a Supervisely JSON
    with PNG-compressed bitmaps. We honor per-object origin and alpha channel.
    """
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)

        if 'objects' not in data or not data['objects']:
            return None

        H, W = target_shape  # target_shape must be (height, width)
        full_mask = np.zeros((H, W), dtype=bool)

        for obj in data['objects']:
            if 'bitmap' not in obj or 'data' not in obj['bitmap'] or 'origin' not in obj['bitmap']:
                continue

            # 1) Decode PNG from zlib+base64
            compressed = base64.b64decode(obj['bitmap']['data'])
            png_bytes = zlib.decompress(compressed)

            # 2) Read PNG and extract alpha (or convert to L)
            with Image.open(io.BytesIO(png_bytes)) as im:
                if im.mode in ('RGBA', 'LA'):
                    alpha = np.array(im.split()[-1])  # use only alpha
                else:
                    alpha = np.array(im.convert('L'))  # 0..255

            # 3) Binarize *without* resizing
            obj_mask = alpha > 0  # True on painted pixels of the crop

            # 4) Paste the crop at its origin on a canvas the size of the image
            ox, oy = obj['bitmap']['origin']  # [x, y]
            h, w = obj_mask.shape
            # guard rails: clip if needed
            y1, y2 = max(0, oy), min(H, oy + h)
            x1, x2 = max(0, ox), min(W, ox + w)
            cy1, cy2 = y1 - oy, y2 - oy  # crop indices
            cx1, cx2 = x1 - ox, x2 - ox

            if y1 < y2 and x1 < x2:
                canvas = np.zeros((H, W), dtype=bool)
                canvas[y1:y2, x1:x2] = obj_mask[cy1:cy2, cx1:cx2]
                full_mask |= canvas

        return full_mask if full_mask.any() else None

    except Exception as e:
        print(f"Error loading GT from {ann_path}: {e}")
        return None

def generate_auto_prompts(image, grid_spacing=GRID_SPACING, box_scales=BOX_SCALES):
    """Generate automatic prompts from image without GT information"""
    height, width = image.shape[:2]
    
    # Strategy 1: Grid-based point sampling
    grid_points = []
    grid_labels = []
    
    for y in range(grid_spacing, height - grid_spacing, grid_spacing):
        for x in range(grid_spacing, width - grid_spacing, grid_spacing):
            grid_points.append([x, y])
            grid_labels.append(1)  # All points are positive prompts
    
    # Strategy 2: Multi-scale box sweeping
    auto_boxes = []
    for scale in box_scales:
        stride = scale
        for y in range(0, height - scale, stride):
            for x in range(0, width - scale, stride):
                if x + scale <= width and y + scale <= height:
                    auto_boxes.append([x, y, x + scale, y + scale])
    
    return {
        'points': np.array(grid_points) if grid_points else np.zeros((0, 2)),
        'point_labels': np.array(grid_labels) if grid_labels else np.zeros((0,)),
        'boxes': np.array(auto_boxes) if auto_boxes else np.zeros((0, 4))
    }

def generate_gt_points(gt_mask, num_points=3):
    """Generate 1-3 GT points from ground truth mask"""
    from scipy.ndimage import label, distance_transform_edt
    if gt_mask is None or gt_mask.sum() == 0:
        return np.zeros((0,2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    labeled, n = label(gt_mask)
    if n == 0:
        return np.zeros((0,2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    # componente principal
    sizes = [np.sum(labeled == i) for i in range(1, n+1)]
    comp = np.argmax(sizes) + 1
    main = (labeled == comp).astype(np.uint8)

    pts, lbs = [], []

    # 1) centroide
    ys, xs = np.where(main)
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    pts.append([cx, cy]); lbs.append(1)

    # 2) punto más interior
    if num_points >= 2:
        dist = distance_transform_edt(main)
        my, mx = np.unravel_index(np.argmax(dist), dist.shape)
        if abs(mx-cx) > 3 or abs(my-cy) > 3:
            pts.append([mx, my]); lbs.append(1)

    # 3) punto negativo fuera del borde
    if num_points >= 3:
        k = np.ones((3,3), np.uint8)
        outer = cv2.dilate(main, k, iterations=6)
        neg_candidates = np.where((outer > 0) & (main == 0))
        if len(neg_candidates[0]) > 0:
            i = np.random.randint(len(neg_candidates[0]))
            ny, nx = int(neg_candidates[0][i]), int(neg_candidates[1][i])
            pts.append([nx, ny]); lbs.append(0)

    return np.array(pts, dtype=np.float32), np.array(lbs, dtype=np.int32)

def generate_30_gt_points(gt_mask):
    """Generate 30 GT points from ground truth mask"""
    if gt_mask is None or gt_mask.sum() == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    
    from scipy.ndimage import label
    labeled_mask, num_components = label(gt_mask)
    
    if num_components == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    
    # Get the largest component
    component_sizes = []
    for i in range(1, num_components + 1):
        component_sizes.append((labeled_mask == i).sum())
    
    largest_component_idx = np.argmax(component_sizes) + 1
    main_mask = (labeled_mask == largest_component_idx)
    
    # Sample 30 points from the mask
    y_coords, x_coords = np.where(main_mask)
    
    if len(y_coords) < 30:
        # If mask is smaller than 30 pixels, repeat some points
        points = []
        labels = []
        for i in range(30):
            idx = i % len(y_coords)
            points.append([x_coords[idx], y_coords[idx]])
            labels.append(1)
        return np.array(points), np.array(labels)
    
    # Randomly sample 30 points
    indices = np.random.choice(len(y_coords), 30, replace=False)
    points = []
    labels = []
    
    for idx in indices:
        points.append([x_coords[idx], y_coords[idx]])
        labels.append(1)  # All positive points
    
    return np.array(points), np.array(labels)

def compute_mask_iou(mask1, mask2):
    """Compute IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-8)

def apply_nms(masks, scores, iou_threshold=NMS_IOU_THRESHOLD):
    """Apply Non-Maximum Suppression to remove overlapping masks"""
    if len(masks) <= 1:
        return masks, scores
    
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []
    
    for idx in sorted_indices:
        keep = True
        for kept_idx in keep_indices:
            iou = compute_mask_iou(masks[idx], masks[kept_idx])
            if iou > iou_threshold:
                keep = False
                break
        
        if keep:
            keep_indices.append(idx)
    
    return [masks[i] for i in keep_indices], [scores[i] for i in keep_indices]

def predict_with_auto_prompts(predictor, image, auto_prompts, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Predict using automatic prompts"""
    height, width = image.shape[:2]
    
    all_masks = []
    all_scores = []
    
    # Predict with grid points
    if len(auto_prompts['points']) > 0:
        try:
            points = auto_prompts['points']
            labels = auto_prompts['point_labels']
            
            for i in range(0, len(points), POINT_BATCH_SIZE):
                batch_points = points[i:i + POINT_BATCH_SIZE]
                batch_labels = labels[i:i + POINT_BATCH_SIZE]
                
                point_masks, point_scores, _ = predictor.predict(
                    point_coords=batch_points,
                    point_labels=batch_labels,
                    multimask_output=True
                )
                
                if len(batch_points) == 1:
                    for j, score in enumerate(point_scores):
                        if score > confidence_threshold:
                            mask = (point_masks[j].astype(np.float32) > 0.5).astype(bool)
                            all_masks.append(mask)
                            all_scores.append(score)
                else:
                    best_idx = np.argmax(point_scores)
                    if point_scores[best_idx] > confidence_threshold:
                        mask = (point_masks[best_idx].astype(np.float32) > 0.5).astype(bool)
                        all_masks.append(mask)
                        all_scores.append(point_scores[best_idx])
                        
        except Exception as e:
            print(f"Error with point prompts: {e}")
    
    # Predict with auto boxes
    if len(auto_prompts['boxes']) > 0:
        try:
            boxes = auto_prompts['boxes']
            
            for box in boxes:
                box_masks, box_scores, _ = predictor.predict(
                    box=box,
                    multimask_output=True
                )
                
                for j, score in enumerate(box_scores):
                    if score > confidence_threshold:
                        mask = (box_masks[j].astype(np.float32) > 0.5).astype(bool)
                        all_masks.append(mask)
                        all_scores.append(score)
                        
        except Exception as e:
            print(f"Error with box prompts: {e}")
    
    if not all_masks:
        return np.zeros((height, width), dtype=bool), 0
    
    # Apply top-K filtering
    if len(all_masks) > TOP_K_FILTER:
        sorted_indices = np.argsort(all_scores)[::-1][:TOP_K_FILTER]
        all_masks = [all_masks[i] for i in sorted_indices]
        all_scores = [all_scores[i] for i in sorted_indices]
    
    # Apply NMS
    filtered_masks, filtered_scores = apply_nms(all_masks, all_scores)
    
    if not filtered_masks:
        return np.zeros((height, width), dtype=bool), 0
    
    # Combine masks using logical OR
    combined_mask = np.zeros((height, width), dtype=bool)
    for mask in filtered_masks:
        binary_mask = mask.astype(bool)
        combined_mask = np.logical_or(combined_mask, binary_mask)
    
    return combined_mask, len(filtered_masks)

def predict_with_gt_points(predictor, image, gt_points, gt_labels, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Predict using GT points"""
    height, width = image.shape[:2]
    
    if len(gt_points) == 0:
        return np.zeros((height, width), dtype=bool), 0
    
    try:
        point_masks, point_scores, _ = predictor.predict(
            point_coords=gt_points,
            point_labels=gt_labels,
            multimask_output=True
        )
        
        best_idx = np.argmax(point_scores)
        best_score = point_scores[best_idx]
        
        if best_score > confidence_threshold:
            pred_mask = (point_masks[best_idx].astype(np.float32) > 0.5).astype(bool)
            return pred_mask, 1
        else:
            return np.zeros((height, width), dtype=bool), 0
            
    except Exception as e:
        print(f"Error with GT point prompts: {e}")
        return np.zeros((height, width), dtype=bool), 0

def create_comparison_visualization(image, gt_mask, predictions, img_name, save_dir):
    """Create comprehensive comparison visualization with contours"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Comparison: {img_name}', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground Truth - show as contour
    axes[0, 1].imshow(image)
    axes[0, 1].contour(gt_mask, levels=[0.5], colors=['green'], linewidths=2)
    axes[0, 1].set_title('Ground Truth (Green Contour)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Small model predictions - show as contours
    axes[0, 2].imshow(image)
    axes[0, 2].contour(predictions['small_auto'], levels=[0.5], colors=['blue'], linewidths=2)
    axes[0, 2].set_title('Small + Auto-prompt', fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(image)
    axes[0, 3].contour(predictions['small_3pts'], levels=[0.5], colors=['red'], linewidths=2)
    axes[0, 3].set_title('Small + 3 GT points', fontweight='bold')
    axes[0, 3].axis('off')
    
    # Large model predictions - show as contours
    axes[1, 0].imshow(image)
    axes[1, 0].contour(predictions['large_auto'], levels=[0.5], colors=['purple'], linewidths=2)
    axes[1, 0].set_title('Large + Auto-prompt', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image)
    axes[1, 1].contour(predictions['large_3pts'], levels=[0.5], colors=['orange'], linewidths=2)
    axes[1, 1].set_title('Large + 3 GT points', fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(image)
    axes[1, 2].contour(predictions['small_30pts'], levels=[0.5], colors=['yellow'], linewidths=2)
    axes[1, 2].set_title('Small + 30 GT points', fontweight='bold')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(image)
    axes[1, 3].contour(predictions['large_30pts'], levels=[0.5], colors=['cyan'], linewidths=2)
    axes[1, 3].set_title('Large + 30 GT points', fontweight='bold')
    axes[1, 3].axis('off')
    
    # Save visualization
    viz_path = os.path.join(save_dir, f"{img_name}_comparison.png")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

def create_gt_verification(image, gt_mask, img_name, save_dir):
    """Create GT verification visualization to check for overflow"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'GT Verification: {img_name}', fontsize=16, fontweight='bold')
    
    # Original image with GT contour
    axes[0].imshow(image)
    axes[0].contour(gt_mask, levels=[0.5], colors=['green'], linewidths=2)
    axes[0].set_title('GT Contour Overlay', fontweight='bold')
    axes[0].axis('off')
    
    # GT mask only
    axes[1].imshow(gt_mask, cmap='Greens')
    axes[1].set_title('GT Mask Only', fontweight='bold')
    axes[1].axis('off')
    
    # Save verification
    viz_path = os.path.join(save_dir, f"{img_name}_gt_verification.png")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

def load_model(model_size, device="cuda"):
    """Load SAM2 model"""
    try:
        # Change to sam2base directory for proper config resolution
        original_dir = os.getcwd()
        sam2base_dir = "/home/ptp/sam2/libs/sam2base"
        os.chdir(sam2base_dir)
        
        if model_size.lower() == "small":
            config_file = "configs/sam2/sam2_hiera_s.yaml"
        elif model_size.lower() == "large":
            config_file = "configs/sam2/sam2_hiera_l.yaml"
        else:
            raise ValueError(f"Invalid model_size: {model_size}")
        
        sam2_model = build_sam2(config_file, ckpt_path=None, device="cpu")
        sam2_model.to(device)
        sam2_model.eval()
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        # Restore original directory
        os.chdir(original_dir)
        
        return sam2_predictor
        
    except Exception as e:
        print(f"Error loading {model_size} model: {e}")
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualize SAM2 base predictions comparison')
    parser.add_argument('--num_images', type=int, default=2,
                       help='Number of images to visualize')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/ptp/sam2/new_src/evaluation/evaluation_results/sam2_base',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    print("SAM2 Base Models Visualization Comparison")
    print("=" * 60)
    print(f"Visualizing {args.num_images} images")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "comparison_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Test split paths
    test_img_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/img"
    test_ann_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/ann"
    
    # Get test images
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    selected_images = random.sample(test_images, min(args.num_images, len(test_images)))
    
    print(f"Selected images: {selected_images}")
    
    # Load models
    print("Loading models...")
    small_predictor = load_model("small")
    large_predictor = load_model("large")
    
    if small_predictor is None or large_predictor is None:
        print("Error: Could not load models")
        return
    
    print("Models loaded successfully!")
    
    # Process each image
    for i, img_file in enumerate(selected_images):
        print(f"\nProcessing image {i+1}/{len(selected_images)}: {img_file}")
        
        # Load image
        img_path = os.path.join(test_img_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Load ground truth
        ann_path = os.path.join(test_ann_dir, img_file + '.json')
        gt_mask = get_binary_gt_mask(ann_path, (height, width))
        
        if gt_mask is None:
            print(f"Skipping {img_file}: No valid GT")
            continue
        
        # Generate predictions for all modes
        predictions = {}
        
        # Small model predictions
        small_predictor.set_image(image)
        
        # Auto-prompt
        auto_prompts = generate_auto_prompts(image)
        predictions['small_auto'], _ = predict_with_auto_prompts(small_predictor, image, auto_prompts)
        
        # 3 GT points
        gt_points_3, gt_labels_3 = generate_gt_points(gt_mask, 3)
        predictions['small_3pts'], _ = predict_with_gt_points(small_predictor, image, gt_points_3, gt_labels_3)
        
        # 30 GT points
        gt_points_30, gt_labels_30 = generate_30_gt_points(gt_mask)
        predictions['small_30pts'], _ = predict_with_gt_points(small_predictor, image, gt_points_30, gt_labels_30)
        
        # Large model predictions
        large_predictor.set_image(image)
        
        # Auto-prompt
        predictions['large_auto'], _ = predict_with_auto_prompts(large_predictor, image, auto_prompts)
        
        # 3 GT points
        predictions['large_3pts'], _ = predict_with_gt_points(large_predictor, image, gt_points_3, gt_labels_3)
        
        # 30 GT points
        predictions['large_30pts'], _ = predict_with_gt_points(large_predictor, image, gt_points_30, gt_labels_30)
        
        # Create GT verification first
        img_name = os.path.splitext(img_file)[0]
        gt_viz_path = create_gt_verification(image, gt_mask, img_name, viz_dir)
        print(f"GT verification saved: {gt_viz_path}")
        
        # Create comparison visualization
        viz_path = create_comparison_visualization(image, gt_mask, predictions, img_name, viz_dir)
        print(f"Comparison visualization saved: {viz_path}")
        
        # Print IoU for each prediction
        print(f"IoU scores for {img_name}:")
        for mode, pred_mask in predictions.items():
            iou = compute_mask_iou(pred_mask, gt_mask)
            print(f"  {mode}: {iou:.4f}")
        
        # Print GT mask statistics
        print(f"GT mask statistics for {img_name}:")
        print(f"  GT mask shape: {gt_mask.shape}")
        print(f"  GT mask sum: {gt_mask.sum()} pixels")
        print(f"  GT mask coverage: {gt_mask.sum() / (gt_mask.shape[0] * gt_mask.shape[1]):.4f}")
    
    print(f"\nVisualization completed!")
    print(f"Results saved to: {viz_dir}")

if __name__ == "__main__":
    main()
