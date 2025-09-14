#!/usr/bin/env python3
"""
Convert Severstal Steel Defect Dataset from JSON annotations to image masks
"""

import os
import sys
import argparse
from pathlib import Path

# Add the libs directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from unetB7.severstal_adapter import SeverstalAdapter


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Convert Severstal dataset to image masks')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory containing the dataset')
    parser.add_argument('--image_dir', type=str, default='images',
                       help='Subdirectory containing images')
    parser.add_argument('--annotation_dir', type=str, default='annotations',
                       help='Subdirectory containing JSON annotations')
    parser.add_argument('--output_mask_dir', type=str, default='masks',
                       help='Output directory for converted masks')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample conversions')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples to visualize')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create full paths
    image_dir = os.path.join(args.data_dir, args.image_dir)
    annotation_dir = os.path.join(args.data_dir, args.annotation_dir)
    output_mask_dir = os.path.join(args.data_dir, args.output_mask_dir)
    
    # Verify input directories exist
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    if not os.path.exists(annotation_dir):
        raise ValueError(f"Annotation directory not found: {annotation_dir}")
    
    print(f"Converting Severstal dataset...")
    print(f"Image directory: {image_dir}")
    print(f"Annotation directory: {annotation_dir}")
    print(f"Output mask directory: {output_mask_dir}")
    
    # Create adapter and convert dataset
    adapter = SeverstalAdapter(image_dir, annotation_dir, output_mask_dir)
    stats = adapter.convert_dataset()
    
    # Save conversion statistics
    stats_file = os.path.join(args.data_dir, 'conversion_stats.json')
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Conversion statistics saved to: {stats_file}")
    
    # Visualize samples if requested
    if args.visualize:
        print(f"\nVisualizing {args.num_samples} sample conversions...")
        adapter.visualize_sample("sample.jpg", num_samples=args.num_samples)
    
    print("\nConversion completed successfully!")
    print(f"Masks saved to: {output_mask_dir}")


if __name__ == "__main__":
    main()
