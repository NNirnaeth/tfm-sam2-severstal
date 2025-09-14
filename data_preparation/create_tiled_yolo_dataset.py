#!/usr/bin/env python3
"""
Create tiled YOLO detection dataset from Severstal data
Tiles: 1600×256 → 800×256 (2 tiles per image, no overlap)

Process:
1. Tile images: split 1600×256 → two 800×256 tiles
2. Recalculate annotations in tile coordinates
3. Filter annotations by minimum area (5% of tile area)
4. Keep negative tiles (no defects) for training stability
5. Coherent split: all tiles from same source image go to same split

Output: YOLO format dataset ready for detection training
"""

import os
import sys
import json
import cv2
import numpy as np
import base64
import zlib
from PIL import Image
from io import BytesIO
from pathlib import Path
import argparse
import yaml
from datetime import datetime
import shutil
from tqdm import tqdm
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def load_supervisely_annotation(ann_path):
    """Load Supervisely format annotation with proper error handling - CORRECTED VERSION"""
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        objects = []
        for obj in data.get("objects", []):
            bitmap = obj.get("bitmap", {})
            if "data" not in bitmap or "origin" not in bitmap:
                continue
            
            # Decode bitmap using cv2 (same as working script)
            try:
                decoded_data = base64.b64decode(bitmap["data"])
                decompressed_data = zlib.decompress(decoded_data)
                
                # Use cv2 to decode PNG (more robust than PIL)
                nparr = np.frombuffer(decompressed_data, np.uint8)
                patch = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                
                if patch is None:
                    print(f"    Warning: Failed to decode bitmap with cv2 for {ann_path}")
                    continue
                
                # Handle different image formats
                if len(patch.shape) == 3:
                    patch = patch[:, :, 0]  # Take first channel
                
                # Ensure binary mask
                patch = (patch > 0).astype(np.uint8)
                
                # Validate that patch has content
                if np.sum(patch) == 0:
                    print(f"    Warning: Empty patch in {ann_path}")
                    continue
                
            except Exception as e:
                print(f"    Warning: Error decoding bitmap in {ann_path}: {e}")
                continue
            
            origin_x, origin_y = map(int, bitmap["origin"])  # [x, y]
            
            # Get class info
            class_title = obj.get("classTitle", "defect")
            class_id = get_class_id(class_title)
            
            # Debug: print patch info
            print(f"    Debug: Loaded patch from {ann_path} - origin: ({origin_x}, {origin_y}), size: {patch.shape}, pixels: {np.sum(patch)}")
            
            objects.append({
                "patch": patch,
                "origin": (origin_x, origin_y),
                "class_id": class_id,
                "class_title": class_title
            })
        
        print(f"  Loaded {len(objects)} objects from {ann_path}")
        return objects
    
    except Exception as e:
        print(f"Warning: Could not load annotation {ann_path}: {e}")
        return []

def get_class_id(class_title):
    """Map Severstal class titles to IDs"""
    class_mapping = {
        "defect": 0,  # Binary case
        "1": 0,       # Severstal class 1
        "2": 1,       # Severstal class 2  
        "3": 2,       # Severstal class 3
        "4": 3        # Severstal class 4
    }
    return class_mapping.get(class_title, 0)

def create_tile_from_image(image, tile_idx, tile_width=800):
    """Create tile from image"""
    height, width = image.shape[:2]
    
    if tile_idx == 0:
        # Left tile: [0, tile_width)
        tile = image[:, :tile_width]
        tile_offset_x = 0
    else:
        # Right tile: [tile_width, width)
        tile = image[:, tile_width:]
        tile_offset_x = tile_width
    
    return tile, tile_offset_x

def project_annotation_to_tile(obj, tile_offset_x, tile_width, tile_height):
    """Project annotation object to tile coordinates"""
    patch = obj["patch"]
    origin_x, origin_y = obj["origin"]
    
    # Calculate patch bounds in original image
    patch_h, patch_w = patch.shape
    patch_x1 = origin_x
    patch_y1 = origin_y
    patch_x2 = origin_x + patch_w
    patch_y2 = origin_y + patch_h
    
    # Calculate tile bounds
    tile_x1 = tile_offset_x
    tile_x2 = tile_offset_x + tile_width
    tile_y1 = 0
    tile_y2 = tile_height
    
    # Calculate intersection
    inter_x1 = max(patch_x1, tile_x1)
    inter_y1 = max(patch_y1, tile_y1)
    inter_x2 = min(patch_x2, tile_x2)
    inter_y2 = min(patch_y2, tile_y2)
    
    # Check if there's valid intersection
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return None
    
    # Extract intersecting patch region
    patch_start_x = inter_x1 - patch_x1
    patch_start_y = inter_y1 - patch_y1
    patch_end_x = patch_start_x + (inter_x2 - inter_x1)
    patch_end_y = patch_start_y + (inter_y2 - inter_y1)
    
    intersected_patch = patch[patch_start_y:patch_end_y, patch_start_x:patch_end_x]
    
    # New origin in tile coordinates
    new_origin_x = inter_x1 - tile_offset_x
    new_origin_y = inter_y1 - tile_y1
    
    return {
        "patch": intersected_patch,
        "origin": (new_origin_x, new_origin_y),
        "class_id": obj["class_id"],
        "class_title": obj["class_title"]
    }

def patch_to_bbox(patch, origin_x, origin_y, img_width, img_height):
    """Convert patch to bounding box in YOLO format - CORRECTED VERSION"""
    # Check if mask has any pixels
    if np.sum(patch) == 0:
        return None
    
    # Find non-zero pixels in the patch
    nonzero_y, nonzero_x = np.nonzero(patch)
    
    if len(nonzero_x) == 0:
        return None
    
    # Calculate bbox in patch coordinates (min/max of non-zero pixels)
    patch_min_x, patch_max_x = nonzero_x.min(), nonzero_x.max()
    patch_min_y, patch_max_y = nonzero_y.min(), nonzero_y.max()
    
    # Convert to absolute image coordinates using origin
    # The origin is where the patch starts in the full image
    bbox_x1 = origin_x + patch_min_x
    bbox_y1 = origin_y + patch_min_y
    bbox_x2 = origin_x + patch_max_x
    bbox_y2 = origin_y + patch_max_y
    
    # Ensure coordinates are within image bounds
    bbox_x1 = max(0, min(img_width - 1, bbox_x1))
    bbox_y1 = max(0, min(img_height - 1, bbox_y1))
    bbox_x2 = max(0, min(img_width - 1, bbox_x2))
    bbox_y2 = max(0, min(img_height - 1, bbox_y2))
    
    # Calculate center and dimensions (following YOLO convention)
    center_x = (bbox_x1 + bbox_x2) / 2.0
    center_y = (bbox_y1 + bbox_y2) / 2.0
    width = bbox_x2 - bbox_x1 + 1
    height = bbox_y2 - bbox_y1 + 1
    
    # Calculate area for filtering
    area = width * height
    
    return {
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "area": area,
        "x1": bbox_x1,
        "y1": bbox_y1,
        "x2": bbox_x2,
        "y2": bbox_y2
    }

def filter_by_minimum_area(tile_objects, tile_width, tile_height, min_area_ratio=0.05):
    """Filter annotations by minimum area threshold"""
    tile_area = tile_width * tile_height
    min_area = tile_area * min_area_ratio
    
    filtered_objects = []
    for obj in tile_objects:
        bbox = patch_to_bbox(obj["patch"], obj["origin"][0], obj["origin"][1], tile_width, tile_height)
        if bbox and bbox["area"] >= min_area:
            obj["bbox"] = bbox
            filtered_objects.append(obj)
    
    return filtered_objects

def save_yolo_annotation(objects, tile_width, tile_height, output_path):
    """Save YOLO format annotation file"""
    with open(output_path, 'w') as f:
        for obj in objects:
            bbox = obj["bbox"]
            class_id = obj["class_id"]
            
            # Normalize coordinates
            norm_center_x = bbox["center_x"] / tile_width
            norm_center_y = bbox["center_y"] / tile_height
            norm_width = bbox["width"] / tile_width
            norm_height = bbox["height"] / tile_height
            
            # Write YOLO format: class x_center y_center width height
            f.write(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

def create_coherent_splits(image_names, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create coherent splits ensuring all tiles from same image go to same split"""
    set_seed(seed)
    
    # Get unique source images (remove tile suffixes)
    source_images = list(set([name.replace('_tile0', '').replace('_tile1', '') for name in image_names]))
    random.shuffle(source_images)
    
    n_total = len(source_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_sources = set(source_images[:n_train])
    val_sources = set(source_images[n_train:n_train + n_val])
    test_sources = set(source_images[n_train + n_val:])
    
    # Assign tiles based on source image
    splits = {"train": [], "val": [], "test": []}
    
    for img_name in image_names:
        source = img_name.replace('_tile0', '').replace('_tile1', '')
        
        if source in train_sources:
            splits["train"].append(img_name)
        elif source in val_sources:
            splits["val"].append(img_name)
        else:
            splits["test"].append(img_name)
    
    return splits

def create_dataset_yaml(output_dir, class_names=None):
    """Create YOLO dataset configuration YAML"""
    if class_names is None:
        class_names = ["defect"]  # Binary segmentation
    
    dataset_config = {
        "path": str(Path(output_dir).absolute()),
        "train": "images/train",
        "val": "images/val", 
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names
    }
    
    yaml_path = os.path.join(output_dir, "yolo_tiled_dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset YAML created: {yaml_path}")
    return yaml_path

def process_dataset(images_dir, annotations_dir, output_dir, tile_width=800, tile_height=256,
                   min_area_ratio=0.05, train_ratio=0.7, val_ratio=0.15):
    """Process full dataset with tiling"""
    
    # Create output directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
    
    # Get all images
    image_files = list(Path(images_dir).glob("*.jpg"))
    print(f"Found {len(image_files)} images to process")
    
    # Process each image and create tiles
    all_tile_names = []
    tile_data = {}  # Store tile info for later processing
    
    print("Creating tiles...")
    for img_path in tqdm(image_files, desc="Processing images"):
        img_name = img_path.stem
        ann_path = Path(annotations_dir) / f"{img_name}.jpg.json"
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image_rgb.shape[:2]
        
        # Validate image dimensions
        if orig_width != 1600 or orig_height != 256:
            print(f"Warning: Image {img_name} has unexpected dimensions {orig_width}x{orig_height}, expected 1600x256")
            continue
        
        # Load annotations
        annotations = load_supervisely_annotation(ann_path)
        
        # Create two tiles
        for tile_idx in range(2):
            tile_name = f"{img_name}_tile{tile_idx}"
            all_tile_names.append(tile_name)
            
            # Create tile image
            tile_image, tile_offset_x = create_tile_from_image(image_rgb, tile_idx, tile_width)
            
            # Project annotations to tile
            tile_objects = []
            for obj in annotations:
                projected_obj = project_annotation_to_tile(obj, tile_offset_x, tile_width, tile_height)
                if projected_obj:
                    tile_objects.append(projected_obj)
            
            # Filter by minimum area
            filtered_objects = filter_by_minimum_area(tile_objects, tile_width, tile_height, min_area_ratio)
            
            # Store tile data
            tile_data[tile_name] = {
                "image": tile_image,
                "objects": filtered_objects,
                "has_defects": len(filtered_objects) > 0
            }
    
    print(f"Created {len(all_tile_names)} tiles")
    
    # Create coherent splits
    print("Creating coherent splits...")
    splits = create_coherent_splits(all_tile_names, train_ratio, val_ratio)
    
    # Save tiles to appropriate splits
    print("Saving tiles...")
    stats = {"train": {"total": 0, "positive": 0}, "val": {"total": 0, "positive": 0}, "test": {"total": 0, "positive": 0}}
    
    for split_name, tile_names in splits.items():
        for tile_name in tqdm(tile_names, desc=f"Saving {split_name} tiles"):
            data = tile_data[tile_name]
            
            # Save image
            img_path = os.path.join(output_dir, "images", split_name, f"{tile_name}.jpg")
            image_bgr = cv2.cvtColor(data["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, image_bgr)
            
            # Save annotation
            ann_path = os.path.join(output_dir, "labels", split_name, f"{tile_name}.txt")
            save_yolo_annotation(data["objects"], tile_width, tile_height, ann_path)
            
            # Update stats
            stats[split_name]["total"] += 1
            if data["has_defects"]:
                stats[split_name]["positive"] += 1
    
    # Print statistics
    print("\nDataset Statistics:")
    for split_name, split_stats in stats.items():
        total = split_stats["total"]
        positive = split_stats["positive"]
        negative = total - positive
        pos_ratio = positive / total * 100 if total > 0 else 0
        print(f"  {split_name.upper()}: {total} tiles ({positive} positive, {negative} negative, {pos_ratio:.1f}% positive)")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Create tiled YOLO detection dataset from Severstal data")
    parser.add_argument("--images-dir", type=str, required=True,
                       help="Directory containing source images (1600x256)")
    parser.add_argument("--annotations-dir", type=str, required=True,
                       help="Directory containing Supervisely format annotations")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for tiled YOLO dataset")
    parser.add_argument("--tile-width", type=int, default=800,
                       help="Width of each tile")
    parser.add_argument("--tile-height", type=int, default=256,
                       help="Height of each tile")
    parser.add_argument("--min-area-ratio", type=float, default=0.05,
                       help="Minimum area ratio for filtering annotations (0.05 = 5%)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation split ratio")
    parser.add_argument("--binary-classes", action="store_true",
                       help="Use binary classification (defect/no-defect)")
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return
    
    if not os.path.exists(args.annotations_dir):
        print(f"Error: Annotations directory not found: {args.annotations_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Tiled YOLO Dataset Creation")
    print(f"  Images: {args.images_dir}")
    print(f"  Annotations: {args.annotations_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Tile size: {args.tile_width}x{args.tile_height}")
    print(f"  Min area ratio: {args.min_area_ratio}")
    print(f"  Splits: train={args.train_ratio}, val={args.val_ratio}, test={1-args.train_ratio-args.val_ratio}")
    print(f"  Binary classes: {args.binary_classes}")
    
    # Process dataset
    start_time = datetime.now()
    
    stats = process_dataset(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        min_area_ratio=args.min_area_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # Create dataset YAML
    class_names = ["defect"] if args.binary_classes else ["1", "2", "3", "4"]
    yaml_path = create_dataset_yaml(args.output_dir, class_names)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\nDataset creation completed in {duration:.2f} seconds")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Ready for YOLO detection training!")

if __name__ == "__main__":
    main()
