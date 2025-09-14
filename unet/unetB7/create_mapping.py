#!/usr/bin/env python3
"""
Create mapping between images and annotations for Severstal dataset
"""

import os
import json
from collections import defaultdict


def create_image_annotation_mapping():
    """Create mapping between images and annotations"""
    
    data_dir = "/home/javi/projects/SAM2/datasets/severstal/splits"
    train_image_dir = os.path.join(data_dir, "train_split", "img")
    train_annotation_dir = os.path.join(data_dir, "train_split", "ann")
    
    print("🔗 Creating Image-Annotation Mapping")
    print("=" * 40)
    
    # Get all files
    image_files = [f for f in os.listdir(train_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotation_files = [f for f in os.listdir(train_annotation_dir) if f.endswith('.json')]
    
    print(f"Found {len(image_files)} images")
    print(f"Found {len(annotation_files)} annotations")
    
    # Create mapping based on file content analysis
    mapping = {}
    
    # For each annotation, try to find the corresponding image
    for ann_file in annotation_files[:10]:  # Test with first 10
        ann_path = os.path.join(train_annotation_dir, ann_file)
        
        try:
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
            
            # Get image dimensions from annotation
            ann_height = annotation.get('size', {}).get('height')
            ann_width = annotation.get('size', {}).get('width')
            
            print(f"\nAnnotation: {ann_file}")
            print(f"  Dimensions: {ann_height}x{ann_width}")
            print(f"  Objects: {len(annotation.get('objects', []))}")
            
            # Try to find matching image by checking dimensions
            # This is a heuristic approach
            for img_file in image_files:
                img_path = os.path.join(train_image_dir, img_file)
                
                # Check if we can get image dimensions
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                        
                        if img_height == ann_height and img_width == ann_width:
                            print(f"  ✅ Potential match: {img_file} ({img_width}x{img_height})")
                            mapping[ann_file] = img_file
                            break
                except Exception as e:
                    continue
            
            if ann_file not in mapping:
                print(f"  ❌ No match found")
                
        except Exception as e:
            print(f"  ❌ Error processing {ann_file}: {e}")
    
    print(f"\n📊 Mapping Results:")
    print(f"Successful mappings: {len(mapping)}")
    
    # Save mapping to file
    mapping_file = os.path.join(data_dir, "image_annotation_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Mapping saved to: {mapping_file}")
    
    return mapping


if __name__ == "__main__":
    create_image_annotation_mapping()
