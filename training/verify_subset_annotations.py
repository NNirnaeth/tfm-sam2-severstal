#!/usr/bin/env python3
"""
Verify that subset annotations now correctly use origin coordinates
Visualize a few images with their YOLO annotations to check positioning
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
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
        binary_mask = (mask_array > 0).astype(np.uint8)
        return binary_mask
    except Exception as e:
        print(f"Error decoding bitmap: {e}")
        return None

def mask_to_yolo_seg(mask, origin_x, origin_y, img_width, img_height):
    """Convert binary mask to YOLO segmentation format with origin correction"""
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yolo_segments = []
        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            coords = approx.reshape(-1, 2)
            normalized_coords = []
            for x, y in coords:
                # Apply origin offset to get absolute coordinates in full image
                abs_x = x + origin_x
                abs_y = y + origin_y
                # Normalize coordinates
                norm_x = abs_x / img_width
                norm_y = abs_y / img_height
                normalized_coords.extend([norm_x, norm_y])
            yolo_segments.append(normalized_coords)
        return yolo_segments
    except Exception as e:
        print(f"Error converting mask to YOLO format: {e}")
        return []

def visualize_annotations(subset_name, num_images=3):
    """Visualize annotations for a subset to verify origin correction"""
    
    # Paths
    input_dir = "/home/ptp/sam2/datasets/Data/splits"
    if subset_name == "full":
        img_dir = os.path.join(input_dir, "train_split", "img")
        ann_dir = os.path.join(input_dir, "train_split", "ann")
    else:
        img_dir = os.path.join(input_dir, "subsets", f"{subset_name}_subset", "img")
        ann_dir = os.path.join(input_dir, "subsets", f"{subset_name}_subset", "ann")
    
    # Get sample images
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:num_images]
    
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    if num_images == 1:
        axes = [axes]
    
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, img_file + '.json')
        
        if not os.path.exists(ann_path):
            continue
            
        # Load image
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Load annotation
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
        
        # Create visualization
        draw = ImageDraw.Draw(img)
        
        # Process bitmap objects
        for obj in ann_data.get('objects', []):
            if obj.get('geometryType') == 'bitmap':
                bitmap_data = obj.get('bitmap', {}).get('data')
                bitmap_origin = obj.get('bitmap', {}).get('origin', [0, 0])
                
                if bitmap_data:
                    # Decode mask
                    mask = decode_bitmap_to_mask(bitmap_data)
                    if mask is not None:
                        origin_x, origin_y = bitmap_origin
                        
                        # Convert to YOLO format with origin correction
                        segments = mask_to_yolo_seg(mask, origin_x, origin_y, img_width, img_height)
                        
                        # Draw YOLO segments
                        for segment in segments:
                            if len(segment) >= 6:  # At least 3 points
                                # Convert normalized coordinates back to pixel coordinates
                                pixel_coords = []
                                for j in range(0, len(segment), 2):
                                    x = int(segment[j] * img_width)
                                    y = int(segment[j+1] * img_height)
                                    pixel_coords.append((x, y))
                                
                                # Draw polygon
                                if len(pixel_coords) >= 3:
                                    draw.polygon(pixel_coords, outline='red', width=2)
                                    
                                    # Draw origin point
                                    draw.ellipse([origin_x-3, origin_y-3, origin_x+3, origin_y+3], 
                                               fill='blue', outline='blue')
        
        # Display
        axes[i].imshow(img)
        axes[i].set_title(f'{subset_name} - {img_file}')
        axes[i].axis('off')
        
        print(f"Processed {img_file}: {len(ann_data.get('objects', []))} objects")
    
    plt.tight_layout()
    plt.savefig(f'/home/ptp/sam2/new_src/training/verify_{subset_name}_annotations.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as verify_{subset_name}_annotations.png")

def main():
    print("Verifying subset annotations with origin correction...")
    
    # Test different subsets
    subsets = ["500", "1000", "2000", "full"]
    
    for subset in subsets:
        print(f"\n{'='*50}")
        print(f"Verifying subset: {subset}")
        print(f"{'='*50}")
        
        try:
            visualize_annotations(subset, num_images=2)
        except Exception as e:
            print(f"Error processing subset {subset}: {e}")

if __name__ == "__main__":
    main()


