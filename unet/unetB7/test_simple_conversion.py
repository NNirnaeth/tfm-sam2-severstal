#!/usr/bin/env python3
"""
Simple test for Severstal conversion without TensorFlow
"""

import os
import json
import base64
import zlib
import numpy as np
import cv2


def test_simple_conversion():
    """Test simple conversion without TensorFlow dependencies"""
    
    data_dir = "/home/javi/projects/SAM2/datasets/severstal/splits"
    train_image_dir = os.path.join(data_dir, "train_split", "img")
    train_annotation_dir = os.path.join(data_dir, "train_split", "ann")
    output_mask_dir = os.path.join(data_dir, "train_split", "test_masks")
    
    print("🧪 Testing Simple Conversion")
    print("=" * 40)
    
    # Create output directory
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Get files
    image_files = [f for f in os.listdir(train_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotation_files = [f for f in os.listdir(train_annotation_dir) if f.endswith('.json')]
    
    print(f"Found {len(image_files)} images")
    print(f"Found {len(annotation_files)} annotations")
    
    # Test first few files
    successful = 0
    failed = 0
    
    for i, ann_file in enumerate(annotation_files[:5]):  # Test first 5
        try:
            # Find corresponding image
            base_name = ann_file.replace('.json', '')
            image_file = None
            
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_image = base_name + ext
                if os.path.exists(os.path.join(train_image_dir, potential_image)):
                    image_file = potential_image
                    break
            
            if image_file is None:
                print(f"❌ {ann_file}: No matching image found")
                failed += 1
                continue
            
            print(f"✅ {ann_file} -> {image_file}")
            
            # Process annotation
            ann_path = os.path.join(train_annotation_dir, ann_file)
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
            
            # Get image dimensions
            height = annotation['size']['height']
            width = annotation['size']['width']
            
            # Create combined mask
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Process each object
            for obj in annotation['objects']:
                if obj['geometryType'] == 'bitmap':
                    bitmap_data = obj['bitmap']['data']
                    origin = obj['bitmap']['origin']
                    
                    # Decode bitmap
                    compressed_data = base64.b64decode(bitmap_data)
                    decompressed_data = zlib.decompress(compressed_data)
                    
                    # Create bitmap (this is simplified)
                    # In reality, you'd need to properly decode the bitmap format
                    bitmap = np.frombuffer(decompressed_data, dtype=np.uint8)
                    
                    # Place bitmap at origin (simplified)
                    x_start, y_start = origin
                    if x_start < width and y_start < height:
                        # Simple placement - in reality you'd need proper bitmap decoding
                        combined_mask[y_start:min(y_start+10, height), x_start:min(x_start+10, width)] = 255
            
            # Save mask
            mask_filename = base_name + '_mask.png'
            mask_path = os.path.join(output_mask_dir, mask_filename)
            cv2.imwrite(mask_path, combined_mask)
            
            print(f"  Saved mask: {mask_filename}")
            successful += 1
            
        except Exception as e:
            print(f"❌ {ann_file}: Error - {e}")
            failed += 1
    
    print(f"\n📊 Results:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Masks saved to: {output_mask_dir}")


if __name__ == "__main__":
    test_simple_conversion()
