#!/usr/bin/env python3
"""
Debug script to check dataset structure without TensorFlow dependencies
"""

import os
import json
import base64
import zlib
import numpy as np


def debug_dataset():
    """Debug dataset structure and annotation format"""
    
    # Paths
    data_dir = "/home/javi/projects/SAM2/datasets/severstal/splits"
    train_image_dir = os.path.join(data_dir, "train_split", "img")
    train_annotation_dir = os.path.join(data_dir, "train_split", "ann")
    
    print("🔍 Debugging Severstal Dataset")
    print("=" * 40)
    print(f"Data directory: {data_dir}")
    print(f"Train image dir: {train_image_dir}")
    print(f"Train annotation dir: {train_annotation_dir}")
    
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
    
    # Test annotation parsing
    if len(annotation_files) > 0:
        print(f"\n🧪 Testing annotation parsing:")
        try:
            test_ann = annotation_files[0]
            ann_path = os.path.join(train_annotation_dir, test_ann)
            
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
            
            print(f"Annotation keys: {list(annotation.keys())}")
            print(f"Size: {annotation.get('size', 'Not found')}")
            print(f"Objects count: {len(annotation.get('objects', []))}")
            
            if 'objects' in annotation and len(annotation['objects']) > 0:
                obj = annotation['objects'][0]
                print(f"First object keys: {list(obj.keys())}")
                print(f"Class title: {obj.get('classTitle', 'Not found')}")
                print(f"Geometry type: {obj.get('geometryType', 'Not found')}")
                
                if 'bitmap' in obj:
                    bitmap = obj['bitmap']
                    print(f"Bitmap keys: {list(bitmap.keys())}")
                    print(f"Origin: {bitmap.get('origin', 'Not found')}")
                    print(f"Data length: {len(bitmap.get('data', ''))}")
                    
                    # Test bitmap decoding
                    try:
                        bitmap_data = bitmap['data']
                        compressed_data = base64.b64decode(bitmap_data)
                        decompressed_data = zlib.decompress(compressed_data)
                        print(f"✅ Bitmap decoding successful: {len(decompressed_data)} bytes")
                    except Exception as e:
                        print(f"❌ Bitmap decoding failed: {e}")
            
        except Exception as e:
            print(f"❌ Error parsing annotation: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Debug completed!")


if __name__ == "__main__":
    debug_dataset()
