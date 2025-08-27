#!/usr/bin/env python3
"""
Data preparation script for Severstal dataset.
Splits dataset into train/val/test sets with stratification.
Creates physical directory structure with copied images and annotations.
"""

import os
import json
import random
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple


def load_annotation_data(ann_dir: Path) -> dict:
    """
    Load annotation data and determine positive/negative samples.
    
    Args:
        ann_dir: Path to annotations directory
        
    Returns:
        Dict mapping image stem to positive flag
    """
    annotation_data = {}
    
    for json_file in ann_dir.glob("*.jpg.json"):
        # Extract the base name without .jpg.json
        stem = json_file.name[:-9]  # Remove .jpg.json
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if there are objects in the annotation
            has_objects = len(data.get('objects', [])) > 0
            annotation_data[stem] = has_objects
            
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            annotation_data[stem] = False
    
    return annotation_data


def stratify_split(annotation_data: dict, 
                   test_size: int = 2000,
                   val_ratio: float = 0.2,
                   seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Stratified split of dataset into train/val/test.
    
    Args:
        annotation_data: Dict mapping image stem to positive flag
        test_size: Fixed test set size
        val_ratio: Validation ratio from remaining data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_list, val_list, test_list)
    """
    random.seed(seed)
    
    # Separate positive and negative samples
    positive_samples = [stem for stem, is_positive in annotation_data.items() if is_positive]
    negative_samples = [stem for stem, is_positive in annotation_data.items() if not is_positive]
    
    print(f"Total samples: {len(annotation_data)}")
    print(f"Positive samples: {len(positive_samples)}")
    print(f"Negative samples: {len(negative_samples)}")
    
    # Fixed test set (stratified)
    if len(negative_samples) == 0:
        # Only positive samples available - use all positive for test
        test_positive = random.sample(positive_samples, min(test_size, len(positive_samples)))
        test_negative = []
    else:
        # Both positive and negative samples available
        test_positive = random.sample(positive_samples, min(test_size // 2, len(positive_samples)))
        test_negative = random.sample(negative_samples, min(test_size - len(test_positive), len(negative_samples)))
    
    test_samples = test_positive + test_negative
    
    # Remove test samples from available data
    remaining_positive = [s for s in positive_samples if s not in test_samples]
    remaining_negative = [s for s in negative_samples if s not in test_samples]
    
    # Calculate validation sizes
    val_positive_size = int(len(remaining_positive) * val_ratio)
    val_negative_size = int(len(remaining_negative) * val_ratio) if len(remaining_negative) > 0 else 0
    
    # Validation set
    val_positive = random.sample(remaining_positive, val_positive_size)
    val_negative = random.sample(remaining_negative, val_negative_size) if len(remaining_negative) > 0 else []
    val_samples = val_positive + val_negative
    
    # Training set (remaining samples)
    train_samples = [s for s in remaining_positive if s not in val_samples] + \
                   [s for s in remaining_negative if s not in val_samples]
    
    print(f"Test set: {len(test_samples)} samples")
    print(f"Validation set: {len(val_samples)} samples")
    print(f"Training set: {len(train_samples)} samples")
    
    return train_samples, val_samples, test_samples


def create_physical_splits(data_dir: Path,
                          train_samples: List[str],
                          val_samples: List[str], 
                          test_samples: List[str]) -> None:
    """
    Create physical directory structure with copied images and annotations.
    
    Args:
        data_dir: Base data directory
        train_samples: List of training image stems
        val_samples: List of validation image stems
        test_samples: List of test image stems
    """
    # Create splits directory structure
    splits_dir = data_dir / "splits"
    test_dir = splits_dir / "test_split"
    train_dir = splits_dir / "train_split"
    val_dir = splits_dir / "val_split"
    
    # Clean existing directories if they exist
    for dir_path in [test_dir, train_dir, val_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    
    # Create directories
    splits_dir.mkdir(parents=True, exist_ok=True)
    for dir_path in [test_dir, train_dir, val_dir]:
        dir_path.mkdir(exist_ok=True)
        (dir_path / "img").mkdir(exist_ok=True)
        (dir_path / "ann").mkdir(exist_ok=True)
    
    # Source directories
    raw_dir = data_dir / "raw" / "severstal"
    img_source = raw_dir / "img"
    ann_source = raw_dir / "ann"
    
    print(f"Creating physical splits in {splits_dir}")
    
    def copy_sample(stem: str, target_dir: Path):
        """Helper function to copy a single sample"""
        # Copy image
        src_img = img_source / f"{stem}.jpg"
        dst_img = target_dir / "img" / f"{stem}.jpg"
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        else:
            print(f"Warning: Image not found: {src_img}")
            return False
        
        # Copy annotation
        src_ann = ann_source / f"{stem}.jpg.json"
        dst_ann = target_dir / "ann" / f"{stem}.jpg.json"
        if src_ann.exists():
            shutil.copy2(src_ann, dst_ann)
        else:
            print(f"Warning: Annotation not found: {src_ann}")
            return False
        
        return True
    
    # Copy test split
    print("Copying test split...")
    test_success = 0
    for stem in test_samples:
        if copy_sample(stem, test_dir):
            test_success += 1
    
    # Copy train split
    print("Copying train split...")
    train_success = 0
    for stem in train_samples:
        if copy_sample(stem, train_dir):
            train_success += 1
    
    # Copy validation split
    print("Copying validation split...")
    val_success = 0
    for stem in val_samples:
        if copy_sample(stem, val_dir):
            val_success += 1
    
    print(f"Physical splits created successfully!")
    print(f"Test split: {test_dir} ({test_success}/{len(test_samples)} samples copied)")
    print(f"Train split: {train_dir} ({train_success}/{len(train_samples)} samples copied)")
    print(f"Val split: {val_dir} ({val_success}/{len(val_samples)} samples copied)")


def main():
    parser = argparse.ArgumentParser(description="Prepare Severstal dataset splits")
    parser.add_argument("--data_dir", type=str, default="datasets/Data",
                       help="Path to data directory")
    parser.add_argument("--test_size", type=int, default=2000,
                       help="Fixed test set size")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation ratio from remaining data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set paths
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw" / "severstal"
    ann_dir = raw_dir / "ann"
    splits_dir = data_dir / "splits"
    
    print(f"Data directory: {data_dir}")
    print(f"Raw directory: {raw_dir}")
    print(f"Annotations directory: {ann_dir}")
    print(f"Splits directory: {splits_dir}")
    
    # Check if directories exist
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")
    
    # Load annotation data
    print("Loading annotation data...")
    annotation_data = load_annotation_data(ann_dir)
    
    if not annotation_data:
        raise ValueError("No annotation files found")
    
    # Perform stratified split
    print("Performing stratified split...")
    train_samples, val_samples, test_samples = stratify_split(
        annotation_data, 
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Create physical splits (always)
    print("Creating physical directory structure...")
    create_physical_splits(data_dir, train_samples, val_samples, test_samples)
    
    print("Data preparation completed successfully!")
    print(f"\nDataset structure created in: {splits_dir}")
    print("Next steps:")
    print("1. Update evaluation scripts to use: datasets/Data/splits/test_split/img")
    print("2. Update training scripts to use: datasets/Data/splits/train_split/img")


if __name__ == "__main__":
    main()