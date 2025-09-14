#!/usr/bin/env python3
"""
Script simple para corregir solo las visualizaciones existentes.
Toma las visualizaciones problemáticas y las corrige sin necesidad de recargar el modelo.
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
import base64
import zlib
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

def get_binary_gt_mask_fixed(ann_path, target_shape):
    """
    CORREGIDA: Load and combine all defect objects into a single binary mask from PNG bitmap
    Usa correctamente las coordenadas de origen (origin) para posicionar los bitmaps
    """
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        if 'objects' not in data or not data['objects']:
            return np.zeros(target_shape, dtype=bool)
        
        # Get target dimensions
        H, W = target_shape
        
        # Create empty mask
        full_mask = np.zeros((H, W), dtype=bool)
        
        # Process all objects with proper bitmap positioning
        for obj in data['objects']:
            if 'bitmap' not in obj or 'data' not in obj['bitmap']:
                continue
            
            bmp = obj['bitmap']
            
            # Decompress bitmap data (PNG compressed)
            compressed_data = base64.b64decode(bmp['data'])
            decompressed_data = zlib.decompress(compressed_data)
            
            # Load PNG image from bytes
            png_image = Image.open(io.BytesIO(decompressed_data))
            
            # Handle different PNG modes correctly
            if png_image.mode == 'RGBA':
                # Use alpha channel for transparency
                patch = np.array(png_image.split()[-1])  # Alpha channel
            elif png_image.mode == 'RGB':
                # Convert RGB to grayscale
                patch = np.array(png_image.convert('L'))
            else:
                # Already grayscale
                patch = np.array(png_image)
            
            # Convert to binary mask
            patch_binary = (patch > 0).astype(np.uint8)
            
            # Get origin coordinates from bitmap (origin = [x, y])
            if 'origin' in bmp:
                ox, oy = map(int, bmp['origin'])
            else:
                ox, oy = 0, 0
            
            ph, pw = patch_binary.shape
            
            # Calculate placement coordinates (clamped to image boundaries)
            x1 = max(0, ox)
            y1 = max(0, oy)
            x2 = min(W, ox + pw)
            y2 = min(H, oy + ph)
            
            # Place the bitmap at the correct position
            if x2 > x1 and y2 > y1:
                # Calculate the portion of the patch to use
                patch_h = y2 - y1
                patch_w = x2 - x1
                
                # Extract the relevant portion of the patch
                patch_portion = patch_binary[:patch_h, :patch_w]
                
                # Place in the full mask using OR operation
                full_mask[y1:y2, x1:x2] = np.logical_or(
                    full_mask[y1:y2, x1:x2],
                    patch_portion.astype(bool)
                )
        
        return full_mask
        
    except Exception as e:
        print(f"Error loading GT from {ann_path}: {e}")
        return np.zeros(target_shape, dtype=bool)

def create_corrected_visualization(image_path, ann_path, output_path, img_name):
    """Create corrected visualization for a single image"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return False
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Load ground truth with CORRECTED function
    gt_mask = get_binary_gt_mask_fixed(ann_path, (height, width))
    
    if gt_mask is None:
        print(f"Could not load GT for: {img_name}")
        return False
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'CORRECTED Analysis: {img_name}', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground Truth - CORREGIDO: Usar overlay rojo en lugar de verde
    axes[0, 1].imshow(image)
    # Create red overlay for GT
    gt_overlay = np.zeros_like(image)
    gt_overlay[:, :, 0] = gt_mask * 255  # Red channel
    axes[0, 1].imshow(gt_overlay, alpha=0.6)
    axes[0, 1].set_title('Ground Truth (Red Overlay)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Prediction - Placeholder (since we don't have predictions)
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Prediction (Not Available)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Ground Truth Only - Show the corrected GT mask
    axes[1, 1].imshow(image)
    # Create red overlay for GT
    gt_overlay = np.zeros_like(image)
    gt_overlay[:, :, 0] = gt_mask * 255  # Red channel
    axes[1, 1].imshow(gt_overlay, alpha=0.8)
    axes[1, 1].set_title('Corrected Ground Truth', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Fix SAM2 base visualizations by correcting GT mask loading')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/ptp/sam2/new_src/evaluation/evaluation_results/sam2_base/sam2_large_base_no_fine-tuning/visualizations',
                       help='Directory with problematic visualizations')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/ptp/sam2/new_src/evaluation/evaluation_results/sam2_base/sam2_large_base_FIXED/visualizations',
                       help='Directory to save corrected visualizations')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images to process')
    
    args = parser.parse_args()
    
    print("SAM2 Base Visualizations - CORRECTION SCRIPT")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing {args.num_images} images")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test split paths
    test_img_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/img"
    test_ann_dir = "/home/ptp/sam2/datasets/Data/splits/test_split/ann"
    
    # Get list of existing visualization files
    if os.path.exists(args.input_dir):
        existing_files = [f for f in os.listdir(args.input_dir) if f.endswith('_analysis.png')]
        selected_files = existing_files[:args.num_images]
    else:
        print(f"Input directory does not exist: {args.input_dir}")
        return
    
    print(f"Found {len(existing_files)} existing visualizations")
    print(f"Processing {len(selected_files)} files")
    
    # Process each image
    for i, viz_file in enumerate(selected_files):
        print(f"Processing {i+1}/{len(selected_files)}: {viz_file}")
        
        # Extract image name from visualization filename
        img_name = viz_file.replace('_analysis.png', '')
        
        # Construct paths
        img_path = os.path.join(test_img_dir, f"{img_name}.jpg")
        ann_path = os.path.join(test_ann_dir, f"{img_name}.jpg.json")
        output_path = os.path.join(args.output_dir, f"{img_name}_CORRECTED_analysis.png")
        
        # Check if files exist
        if not os.path.exists(img_path):
            print(f"  Image not found: {img_path}")
            continue
        if not os.path.exists(ann_path):
            print(f"  Annotation not found: {ann_path}")
            continue
        
        # Create corrected visualization
        success = create_corrected_visualization(img_path, ann_path, output_path, img_name)
        
        if success:
            print(f"  Saved: {output_path}")
        else:
            print(f"  Failed to create visualization for {img_name}")
    
    print(f"\n{'='*60}")
    print("CORRECTED visualizations completed!")
    print("Key fixes applied:")
    print("1. Fixed get_binary_gt_mask to properly handle bitmap origin coordinates")
    print("2. Changed GT visualization from green overlay to red overlay")
    print("3. Proper bitmap positioning using origin coordinates")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
