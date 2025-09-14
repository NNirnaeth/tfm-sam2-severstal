#!/usr/bin/env python3
"""
Evaluation script for UNet with EfficientNet-B7
"""

import os
import sys
import argparse
from pathlib import Path

# Add the libs directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from unetB7.model import UNetEfficientNetB7
from unetB7.data_utils import DataLoader, DataAugmentation, DataPreprocessor
from unetB7.evaluator import Evaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate UNet with EfficientNet-B7')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 1600],
                       help='Input image size (height width)')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of output classes')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test images and masks')
    parser.add_argument('--image_dir', type=str, default='images',
                       help='Subdirectory containing images')
    parser.add_argument('--mask_dir', type=str, default='masks',
                       help='Subdirectory containing masks')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary predictions')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Print configuration
    print("Evaluation Configuration:")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("="*50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data paths
    image_dir = os.path.join(args.data_dir, args.image_dir)
    mask_dir = os.path.join(args.data_dir, args.mask_dir)
    
    # Verify data directories exist
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    if not os.path.exists(mask_dir):
        raise ValueError(f"Mask directory not found: {mask_dir}")
    
    # Create data splits (use all data for evaluation)
    print("\nLoading test data...")
    splits = DataPreprocessor.create_data_splits(
        image_dir=image_dir,
        mask_dir=mask_dir,
        test_size=0.0,  # Use all data
        val_size=0.0,   # Use all data
        random_state=42
    )
    
    print(f"Test samples: {len(splits['train']['images'])}")
    
    # Create data augmentation (minimal for evaluation)
    val_augmentation = DataAugmentation(
        input_size=tuple(args.input_size),
        is_training=False
    )
    
    # Create data loader
    test_loader = DataLoader(
        image_paths=splits['train']['images'],
        mask_paths=splits['train']['masks'],
        batch_size=args.batch_size,
        input_size=tuple(args.input_size),
        num_classes=args.num_classes,
        augmentation=val_augmentation,
        shuffle=False
    )
    
    # Create model
    print("\nLoading model...")
    model = UNetEfficientNetB7(
        input_shape=(args.input_size[0], args.input_size[1], 3),
        num_classes=args.num_classes
    )
    
    # Load trained model
    model.load_model(args.model_path)
    print(f"Model loaded from: {args.model_path}")
    
    # Create evaluator
    evaluator = Evaluator(model.model)
    
    # Generate comprehensive evaluation report
    print("\nGenerating evaluation report...")
    metrics = evaluator.generate_report(
        data_loader=test_loader,
        threshold=args.threshold,
        save_path=os.path.join(args.output_dir, "evaluation_report.txt")
    )
    
    # Plot metrics distribution
    print("\nPlotting metrics distribution...")
    evaluator.plot_metrics_distribution(
        data_loader=test_loader,
        threshold=args.threshold,
        save_path=os.path.join(args.output_dir, "metrics_distribution.png")
    )
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    evaluator.plot_confusion_matrix(
        data_loader=test_loader,
        threshold=args.threshold,
        save_path=os.path.join(args.output_dir, "confusion_matrix.png")
    )
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    evaluator.visualize_predictions(
        data_loader=test_loader,
        num_samples=8,
        threshold=args.threshold,
        save_path=os.path.join(args.output_dir, "predictions_visualization.png")
    )
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
