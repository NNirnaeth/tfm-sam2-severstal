#!/usr/bin/env python3
"""
Script to create stratified subsets from training data for SAM2 experiments.
Creates subsets of different sizes while maintaining defect ratio and mask area distribution.
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)

def load_annotation(ann_path: Path) -> Dict:
    """Load annotation file and return parsed JSON."""
    with open(ann_path, 'r') as f:
        return json.load(f)

def calculate_mask_area(bitmap_data: str) -> int:
    """Calculate mask area from base64 encoded bitmap data."""
    import base64
    import zlib
    from PIL import Image
    import io
    
    try:
        # Decode base64 and decompress
        decoded_data = base64.b64decode(bitmap_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Convert to PIL Image and count non-zero pixels
        img = Image.open(io.BytesIO(decompressed_data))
        return np.sum(np.array(img) > 0)
    except Exception as e:
        print(f"Error calculating mask area: {e}")
        return 0

def analyze_training_data(train_img_dir: Path, train_ann_dir: Path) -> pd.DataFrame:
    """Analyze training data to get statistics for stratification."""
    print("Analyzing training data...")
    
    data_info = []
    img_files = list(train_img_dir.glob("*.jpg"))
    
    for img_file in img_files:
        # Extract stem (name without .jpg extension) and look for .jpg.json annotation
        stem = img_file.stem
        ann_file = train_ann_dir / f"{stem}.jpg.json"
        
        if not ann_file.exists():
            # Image without annotations
            data_info.append({
                'image_name': img_file.name,
                'defect_count': 0,
                'total_mask_area': 0,
                'has_defects': False
            })
            continue
            
        ann = load_annotation(ann_file)
        objects = ann.get('objects', [])
        
        defect_count = len(objects)
        total_mask_area = 0
        
        for obj in objects:
            if 'bitmap' in obj and 'data' in obj['bitmap']:
                area = calculate_mask_area(obj['bitmap']['data'])
                total_mask_area += area
        
        data_info.append({
            'image_name': img_file.name,
            'defect_count': defect_count,
            'total_mask_area': total_mask_area,
            'has_defects': defect_count > 0
        })
    
    df = pd.DataFrame(data_info)
    print(f"Total images: {len(df)}")
    print(f"Images with defects: {df['has_defects'].sum()}")
    print(f"Defect ratio: {df['has_defects'].mean():.3f}")
    print(f"Mean defect count: {df['defect_count'].mean():.2f}")
    print(f"Mean mask area: {df['total_mask_area'].mean():.2f}")
    
    return df

def create_area_bins(mask_areas: List[int], n_bins: int = 5) -> List[int]:
    """Create area bins for stratification."""
    areas = np.array(mask_areas)
    bins = np.percentile(areas[areas > 0], np.linspace(0, 100, n_bins + 1))
    return bins

def assign_area_bin(mask_area: int, area_bins: List[int]) -> int:
    """Assign mask area to a bin."""
    if mask_area == 0:
        return 0  # No defects
    
    for i in range(len(area_bins) - 1):
        if area_bins[i] <= mask_area < area_bins[i + 1]:
            return i + 1
    
    return len(area_bins) - 1

def stratified_sampling(df: pd.DataFrame, subset_size: int) -> List[str]:
    """Perform stratified sampling based on defect presence and area bins."""
    print(f"Creating stratified subset of size {subset_size}...")
    
    # Create area bins from non-zero areas
    non_zero_areas = df[df['total_mask_area'] > 0]['total_mask_area'].values
    if len(non_zero_areas) > 0:
        area_bins = create_area_bins(non_zero_areas)
        df['area_bin'] = df['total_mask_area'].apply(lambda x: assign_area_bin(x, area_bins))
    else:
        df['area_bin'] = 0
    
    # Create stratification groups
    df['strata'] = df['has_defects'].astype(str) + '_' + df['area_bin'].astype(str)
    
    # Calculate target samples per stratum
    strata_counts = df['strata'].value_counts()
    total_samples = min(subset_size, len(df))
    
    # Proportional allocation
    stratum_ratios = strata_counts / len(df)
    target_per_stratum = (stratum_ratios * total_samples).round().astype(int)
    
    # Ensure we don't exceed total samples
    while target_per_stratum.sum() > total_samples:
        # Reduce largest stratum
        max_stratum = target_per_stratum.idxmax()
        target_per_stratum[max_stratum] -= 1
    
    # Sample from each stratum
    selected_images = []
    for stratum, target_count in target_per_stratum.items():
        if target_count > 0:
            stratum_df = df[df['strata'] == stratum]
            if len(stratum_df) >= target_count:
                sampled = stratum_df.sample(n=target_count, random_state=SEED)
            else:
                # If stratum is smaller than target, take all
                sampled = stratum_df
            selected_images.extend(sampled['image_name'].tolist())
    
    # If we still need more images, sample randomly from remaining
    remaining_needed = total_samples - len(selected_images)
    if remaining_needed > 0:
        remaining_df = df[~df['image_name'].isin(selected_images)]
        if len(remaining_df) > 0:
            additional = remaining_df.sample(n=min(remaining_needed, len(remaining_df)), random_state=SEED)
            selected_images.extend(additional['image_name'].tolist())
    
    print(f"Selected {len(selected_images)} images for subset")
    return selected_images

def create_subset(subset_name: str, selected_images: List[str], 
                  train_img_dir: Path, train_ann_dir: Path,
                  output_base: Path):
    """Create subset directory with selected images and annotations."""
    subset_dir = output_base / subset_name
    subset_img_dir = subset_dir / "img"
    subset_ann_dir = subset_dir / "ann"
    
    # Create directories
    subset_img_dir.mkdir(parents=True, exist_ok=True)
    subset_ann_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating subset: {subset_name}")
    print(f"Copying {len(selected_images)} images and annotations...")
    
    copied_imgs = 0
    copied_anns = 0
    
    for img_name in selected_images:
        # Copy image
        src_img = train_img_dir / img_name
        dst_img = subset_img_dir / img_name
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            copied_imgs += 1
        
        # Copy annotation
        stem = Path(img_name).stem
        ann_name = f"{stem}.jpg.json"
        src_ann = train_ann_dir / ann_name
        dst_ann = subset_ann_dir / ann_name
        if src_ann.exists():
            shutil.copy2(src_ann, dst_ann)
            copied_anns += 1
        else:
            print(f"Warning: Annotation not found for {img_name}")
    
    print(f"Subset {subset_name} created successfully")
    print(f"  - Images copied: {copied_imgs}")
    print(f"  - Annotations copied: {copied_anns}")

def create_full_dataset(train_img_dir: Path, train_ann_dir: Path, val_img_dir: Path, val_ann_dir: Path,
                       output_base: Path):
    """Create full dataset (train + val) for comparison."""
    full_dir = output_base / "full_dataset"
    full_train_img = full_dir / "train" / "img"
    full_train_ann = full_dir / "train" / "ann"
    full_val_img = full_dir / "val" / "img"
    full_val_ann = full_dir / "val" / "ann"
    
    # Create directories
    full_train_img.mkdir(parents=True, exist_ok=True)
    full_train_ann.mkdir(parents=True, exist_ok=True)
    full_val_img.mkdir(parents=True, exist_ok=True)
    full_val_ann.mkdir(parents=True, exist_ok=True)
    
    print("Creating full dataset...")
    
    # Copy all training data
    for img_file in train_img_dir.glob("*.jpg"):
        shutil.copy2(img_file, full_train_img / img_file.name)
        stem = img_file.stem
        ann_file = train_ann_dir / f"{stem}.jpg.json"
        if ann_file.exists():
            shutil.copy2(ann_file, full_train_ann / f"{stem}.jpg.json")
    
    # Copy all validation data
    for img_file in val_img_dir.glob("*.jpg"):
        shutil.copy2(img_file, full_val_img / img_file.name)
        stem = img_file.stem
        ann_file = val_ann_dir / f"{stem}.jpg.json"
        if ann_file.exists():
            shutil.copy2(ann_file, full_val_ann / f"{stem}.jpg.json")
    
    print("Full dataset created successfully")

def main():
    parser = argparse.ArgumentParser(description="Create stratified subsets from training data")
    parser.add_argument("--train_img_dir", type=str, 
                       default="/home/ptp/sam2/datasets/Data/splits/train_split/img",
                       help="Path to training images directory")
    parser.add_argument("--train_ann_dir", type=str,
                       default="/home/ptp/sam2/datasets/Data/splits/train_split/ann",
                       help="Path to training annotations directory")
    parser.add_argument("--val_img_dir", type=str,
                       default="/home/ptp/sam2/datasets/Data/splits/val_split/img",
                       help="Path to validation images directory")
    parser.add_argument("--val_ann_dir", type=str,
                       default="/home/ptp/sam2/datasets/Data/splits/val_split/ann",
                       help="Path to validation annotations directory")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ptp/sam2/datasets/Data/splits/subsets",
                       help="Output directory for subsets")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    train_img_dir = Path(args.train_img_dir)
    train_ann_dir = Path(args.train_ann_dir)
    val_img_dir = Path(args.val_img_dir)
    val_ann_dir = Path(args.val_ann_dir)
    output_dir = Path(args.output_dir)
    
    # Verify directories exist
    for dir_path in [train_img_dir, train_ann_dir, val_img_dir, val_ann_dir]:
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze training data
    df = analyze_training_data(train_img_dir, train_ann_dir)
    
    # Define subset sizes
    subset_sizes = [100, 200, 500, 1000, 2000, 4000]
    
    # Create subsets
    for size in subset_sizes:
        if size <= len(df):
            subset_name = f"{size}_subset"
            selected_images = stratified_sampling(df, size)
            create_subset(subset_name, selected_images, train_img_dir, train_ann_dir, output_dir)
        else:
            print(f"Skipping {size}_subset: requested size larger than available data")
    
    # Create full dataset
    create_full_dataset(train_img_dir, train_ann_dir, val_img_dir, val_ann_dir, output_dir)
    
    print("\nAll subsets created successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
