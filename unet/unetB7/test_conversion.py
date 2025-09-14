#!/usr/bin/env python3
"""
Test script to debug Severstal conversion issues
"""

import os
import sys
from pathlib import Path

# Add the libs directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from unetB7.severstal_adapter import SeverstalAdapter


def test_conversion():
    """Test conversion with detailed debugging"""
    
    # Paths
    data_dir = "/home/javi/projects/SAM2/datasets/severstal/splits"
    train_image_dir = os.path.join(data_dir, "train_split", "img")
    train_annotation_dir = os.path.join(data_dir, "train_split", "ann")
    output_mask_dir = os.path.join(data_dir, "train_split", "test_masks")
    
    print("🔍 Testing Severstal Conversion")
    print("=" * 40)
    print(f"Data directory: {data_dir}")
    print(f"Train image dir: {train_image_dir}")
    print(f"Train annotation dir: {train_annotation_dir}")
    print(f"Output mask dir: {output_mask_dir}")
    
    # Check directories
    print(f"\n📁 Directory checks:")
    print(f"Data dir exists: {os.path.exists(data_dir)}")
    print(f"Train image dir exists: {os.path.exists(train_image_dir)}")
    print(f"Train annotation dir exists: {os.path.exists(train_annotation_dir)}")
    
    if not os.path.exists(train_image_dir):
        print(f"❌ Train image directory not found: {train_image_dir}")
        return
    
    if not os.path.exists(train_annotation_dir):
        print(f"❌ Train annotation directory not found: {train_annotation_dir}")
        return
    
    # List files
    print(f"\n📄 File listings:")
    image_files = [f for f in os.listdir(train_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotation_files = [f for f in os.listdir(train_annotation_dir) if f.endswith('.json')]
    
    print(f"Image files: {len(image_files)}")
    if len(image_files) > 0:
        print(f"Sample images: {image_files[:3]}")
    
    print(f"Annotation files: {len(annotation_files)}")
    if len(annotation_files) > 0:
        print(f"Sample annotations: {annotation_files[:3]}")
    
    # Check for matching pairs
    print(f"\n🔗 Matching pairs:")
    matches = 0
    for ann_file in annotation_files[:5]:  # Check first 5
        base_name = ann_file.replace('.json', '')
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_image = base_name + ext
            if os.path.exists(os.path.join(train_image_dir, potential_image)):
                print(f"✅ {ann_file} -> {potential_image}")
                matches += 1
                break
        else:
            print(f"❌ {ann_file} -> No matching image found")
    
    print(f"Matches found: {matches}/{min(5, len(annotation_files))}")
    
    # Test single conversion
    if len(annotation_files) > 0:
        print(f"\n🧪 Testing single conversion:")
        try:
            adapter = SeverstalAdapter(train_image_dir, train_annotation_dir, output_mask_dir)
            
            # Test with first annotation file
            test_ann = annotation_files[0]
            print(f"Testing with: {test_ann}")
            
            mask, metadata = adapter.process_annotation(os.path.join(train_annotation_dir, test_ann))
            print(f"Mask shape: {mask.shape}")
            print(f"Metadata: {metadata}")
            
            # Save test mask
            os.makedirs(output_mask_dir, exist_ok=True)
            import cv2
            test_mask_path = os.path.join(output_mask_dir, "test_mask.png")
            cv2.imwrite(test_mask_path, mask * 255)
            print(f"Test mask saved to: {test_mask_path}")
            
        except Exception as e:
            print(f"❌ Error in single conversion: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Test completed!")


if __name__ == "__main__":
    test_conversion()
