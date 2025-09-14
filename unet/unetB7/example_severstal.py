#!/usr/bin/env python3
"""
Example script for training UNet with EfficientNet-B7 on Severstal Steel Defect Dataset
"""

import os
import sys
from pathlib import Path

# Add the libs directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from unetB7.severstal_adapter import SeverstalAdapter


def main():
    """Example usage with your Severstal dataset"""
    
    # Paths to your dataset (adjust these according to your structure)
    data_dir = "/home/javi/projects/SAM2/datasets/severstal/splits/train_split"
    image_dir = os.path.join(data_dir, "img")  # Adjust if your images are in a different subdirectory
    annotation_dir = os.path.join(data_dir, "ann")  # Your JSON annotations
    output_mask_dir = os.path.join(data_dir, "masks")  # Where to save converted masks
    
    print("Severstal Steel Defect Dataset Example")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Image directory: {image_dir}")
    print(f"Annotation directory: {annotation_dir}")
    print(f"Output mask directory: {output_mask_dir}")
    
    # Check if directories exist
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        print("Please check your dataset structure")
        return
    
    if not os.path.exists(annotation_dir):
        print(f"❌ Annotation directory not found: {annotation_dir}")
        print("Please check your dataset structure")
        return
    
    # Create adapter
    print("\n🔧 Creating Severstal adapter...")
    adapter = SeverstalAdapter(image_dir, annotation_dir, output_mask_dir)
    
    # Convert dataset
    print("\n🔄 Converting JSON annotations to image masks...")
    stats = adapter.convert_dataset()
    
    # Print statistics
    print(f"\n📊 Conversion Statistics:")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Successful conversions: {stats['successful_conversions']}")
    print(f"Failed conversions: {stats['failed_conversions']}")
    print(f"Total defects found: {stats['total_defects']}")
    print(f"Defect types: {stats['defect_types']}")
    
    # Visualize samples
    print(f"\n🖼️  Visualizing sample conversions...")
    try:
        adapter.visualize_sample("sample.jpg", num_samples=4)
    except Exception as e:
        print(f"Could not visualize samples: {e}")
    
    # Create data splits
    print(f"\n📂 Creating train/validation/test splits...")
    splits = adapter.create_data_splits(test_size=0.2, val_size=0.1)
    
    print(f"Train samples: {len(splits['train']['images'])}")
    print(f"Validation samples: {len(splits['val']['images'])}")
    print(f"Test samples: {len(splits['test']['images'])}")
    
    # Example training command with separate splits
    print(f"\n🚀 To train the model with separate splits, run:")
    print(f"python train.py \\")
    print(f"    --data_dir /home/javi/projects/SAM2/datasets/severstal/splits \\")
    print(f"    --use_severstal_format \\")
    print(f"    --train_split train_split \\")
    print(f"    --val_split val_split \\")
    print(f"    --test_split test_split \\")
    print(f"    --image_dir img \\")
    print(f"    --annotation_dir ann \\")
    print(f"    --epochs 100 \\")
    print(f"    --batch_size 2 \\")
    print(f"    --learning_rate 1e-4 \\")
    print(f"    --output_dir ./outputs \\")
    print(f"    --experiment_name severstal_defect_detection")
    
    print(f"\n✅ Dataset preparation completed!")
    print(f"Masks saved to: {output_mask_dir}")


if __name__ == "__main__":
    main()
