#!/usr/bin/env python3
"""
Training script for UNet with EfficientNet-B7
High-resolution image segmentation (256x1600)
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the libs directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from unetB7.model import UNetEfficientNetB7
from unetB7.data_utils import DataLoader, DataAugmentation, DataPreprocessor
from unetB7.trainer import Trainer
from unetB7.evaluator import Evaluator
from unetB7.severstal_adapter import SeverstalAdapter


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train UNet with EfficientNet-B7')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory containing the dataset')
    parser.add_argument('--train_split', type=str, default='train_split',
                       help='Training split directory name')
    parser.add_argument('--val_split', type=str, default='val_split',
                       help='Validation split directory name')
    parser.add_argument('--test_split', type=str, default='test_split',
                       help='Test split directory name')
    parser.add_argument('--image_dir', type=str, default='img',
                       help='Subdirectory containing images within each split')
    parser.add_argument('--mask_dir', type=str, default='masks',
                       help='Subdirectory containing masks (for standard format)')
    parser.add_argument('--annotation_dir', type=str, default='ann',
                       help='Subdirectory containing JSON annotations (for Severstal format)')
    parser.add_argument('--use_severstal_format', action='store_true',
                       help='Use Severstal JSON annotation format')
    
    # Model arguments
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 1600],
                       help='Input image size (height width)')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of output classes')
    parser.add_argument('--pretrained_weights', type=str, default='imagenet',
                       help='Pretrained weights for EfficientNet')
    parser.add_argument('--freeze_encoder_layers', type=int, default=10,
                       help='Number of encoder layers to freeze')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--loss', type=str, default='bce_dice',
                       choices=['dice_loss', 'bce_dice', 'focal_loss'],
                       help='Loss function to use')
    parser.add_argument('--use_cosine_schedule', action='store_true',
                       help='Use cosine learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')
    
    # Data split arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Fraction of remaining data for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--experiment_name', type=str, default='unet_efficientnetb7',
                       help='Name of the experiment')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary predictions')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Print configuration
    print("Training Configuration:")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("="*50)
    
    # Set random seeds for reproducibility
    import numpy as np
    import tensorflow as tf
    
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Create data paths for each split
    train_image_dir = os.path.join(args.data_dir, args.train_split, args.image_dir)
    val_image_dir = os.path.join(args.data_dir, args.val_split, args.image_dir)
    test_image_dir = os.path.join(args.data_dir, args.test_split, args.image_dir)
    
    # Handle Severstal format vs standard format
    if args.use_severstal_format:
        # Severstal format with separate splits
        train_annotation_dir = os.path.join(args.data_dir, args.train_split, args.annotation_dir)
        val_annotation_dir = os.path.join(args.data_dir, args.val_split, args.annotation_dir)
        test_annotation_dir = os.path.join(args.data_dir, args.test_split, args.annotation_dir)
        
        train_mask_dir = os.path.join(args.data_dir, args.train_split, 'converted_masks')
        val_mask_dir = os.path.join(args.data_dir, args.val_split, 'converted_masks')
        test_mask_dir = os.path.join(args.data_dir, args.test_split, 'converted_masks')
        
        # Verify directories exist
        for split_name, img_dir, ann_dir in [('train', train_image_dir, train_annotation_dir),
                                           ('val', val_image_dir, val_annotation_dir),
                                           ('test', test_image_dir, test_annotation_dir)]:
            if not os.path.exists(img_dir):
                raise ValueError(f"{split_name} image directory not found: {img_dir}")
            if not os.path.exists(ann_dir):
                raise ValueError(f"{split_name} annotation directory not found: {ann_dir}")
        
        # Convert Severstal annotations to masks for each split
        print("\nConverting Severstal annotations to masks...")
        
        # Train split
        print("Converting train split...")
        train_adapter = SeverstalAdapter(train_image_dir, train_annotation_dir, train_mask_dir)
        train_stats = train_adapter.convert_dataset()
        
        # Val split
        print("Converting validation split...")
        val_adapter = SeverstalAdapter(val_image_dir, val_annotation_dir, val_mask_dir)
        val_stats = val_adapter.convert_dataset()
        
        # Test split
        print("Converting test split...")
        test_adapter = SeverstalAdapter(test_image_dir, test_annotation_dir, test_mask_dir)
        test_stats = test_adapter.convert_dataset()
        
        # Create data splits using the converted masks
        print("\nCreating data splits...")
        splits = {
            'train': {
                'images': [os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                'masks': [os.path.join(train_mask_dir, f.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')) 
                         for f in os.listdir(train_image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            },
            'val': {
                'images': [os.path.join(val_image_dir, f) for f in os.listdir(val_image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                'masks': [os.path.join(val_mask_dir, f.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')) 
                         for f in os.listdir(val_image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            },
            'test': {
                'images': [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                'masks': [os.path.join(test_mask_dir, f.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')) 
                         for f in os.listdir(test_image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            }
        }
        
        # Filter out non-existent masks
        for split_name in ['train', 'val', 'test']:
            valid_pairs = []
            for img_path, mask_path in zip(splits[split_name]['images'], splits[split_name]['masks']):
                if os.path.exists(mask_path):
                    valid_pairs.append((img_path, mask_path))
            
            splits[split_name]['images'] = [pair[0] for pair in valid_pairs]
            splits[split_name]['masks'] = [pair[1] for pair in valid_pairs]
        
        print(f"Train conversion stats: {train_stats}")
        print(f"Val conversion stats: {val_stats}")
        print(f"Test conversion stats: {test_stats}")
        
    else:
        # Standard format with separate splits
        train_mask_dir = os.path.join(args.data_dir, args.train_split, args.mask_dir)
        val_mask_dir = os.path.join(args.data_dir, args.val_split, args.mask_dir)
        test_mask_dir = os.path.join(args.data_dir, args.test_split, args.mask_dir)
        
        # Verify data directories exist
        for split_name, img_dir, mask_dir in [('train', train_image_dir, train_mask_dir),
                                           ('val', val_image_dir, val_mask_dir),
                                           ('test', test_image_dir, test_mask_dir)]:
            if not os.path.exists(img_dir):
                raise ValueError(f"{split_name} image directory not found: {img_dir}")
            if not os.path.exists(mask_dir):
                raise ValueError(f"{split_name} mask directory not found: {mask_dir}")
        
        # Create data splits from existing directories
        print("\nLoading data from separate splits...")
        splits = {
            'train': {
                'images': [os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                'masks': [os.path.join(train_mask_dir, f.replace('.jpg', '.png').replace('.jpeg', '.png')) 
                         for f in os.listdir(train_image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            },
            'val': {
                'images': [os.path.join(val_image_dir, f) for f in os.listdir(val_image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                'masks': [os.path.join(val_mask_dir, f.replace('.jpg', '.png').replace('.jpeg', '.png')) 
                         for f in os.listdir(val_image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            },
            'test': {
                'images': [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                'masks': [os.path.join(test_mask_dir, f.replace('.jpg', '.png').replace('.jpeg', '.png')) 
                         for f in os.listdir(test_image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            }
        }
        
        # Filter out non-existent masks
        for split_name in ['train', 'val', 'test']:
            valid_pairs = []
            for img_path, mask_path in zip(splits[split_name]['images'], splits[split_name]['masks']):
                if os.path.exists(mask_path):
                    valid_pairs.append((img_path, mask_path))
            
            splits[split_name]['images'] = [pair[0] for pair in valid_pairs]
            splits[split_name]['masks'] = [pair[1] for pair in valid_pairs]
    
    # Print data split information
    print(f"Train samples: {len(splits['train']['images'])}")
    print(f"Validation samples: {len(splits['val']['images'])}")
    print(f"Test samples: {len(splits['test']['images'])}")
    
    # Create data augmentation
    train_augmentation = DataAugmentation(
        input_size=tuple(args.input_size),
        is_training=True
    )
    
    val_augmentation = DataAugmentation(
        input_size=tuple(args.input_size),
        is_training=False
    )
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        image_paths=splits['train']['images'],
        mask_paths=splits['train']['masks'],
        batch_size=args.batch_size,
        input_size=tuple(args.input_size),
        num_classes=args.num_classes,
        augmentation=train_augmentation,
        shuffle=True
    )
    
    val_loader = DataLoader(
        image_paths=splits['val']['images'],
        mask_paths=splits['val']['masks'],
        batch_size=args.batch_size,
        input_size=tuple(args.input_size),
        num_classes=args.num_classes,
        augmentation=val_augmentation,
        shuffle=False
    )
    
    # Create model
    print("\nCreating model...")
    model = UNetEfficientNetB7(
        input_shape=(args.input_size[0], args.input_size[1], 3),
        num_classes=args.num_classes,
        pretrained_weights=args.pretrained_weights,
        freeze_encoder_layers=args.freeze_encoder_layers
    )
    
    # Build and compile model
    model.build_model()
    model.compile_model(
        learning_rate=args.learning_rate,
        loss=args.loss,
        metrics=['accuracy', 'dice_coefficient']
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.get_model_summary()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_data=train_loader,
        val_data=val_loader,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
    
    # Calculate class weights if needed
    print("\nCalculating class weights...")
    class_weights = DataPreprocessor.calculate_class_weights(splits['train']['masks'])
    print(f"Class weights: {class_weights}")
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        epochs=args.epochs,
        initial_learning_rate=args.learning_rate,
        loss=args.loss,
        class_weights=class_weights,
        use_cosine_schedule=args.use_cosine_schedule,
        warmup_epochs=args.warmup_epochs
    )
    
    # Plot training history
    print("\nPlotting training history...")
    trainer.plot_training_history(save_plots=True)
    
    # Evaluate model if requested
    if args.evaluate:
        print("\nEvaluating model...")
        
        # Create test data loader
        test_loader = DataLoader(
            image_paths=splits['test']['images'],
            mask_paths=splits['test']['masks'],
            batch_size=args.batch_size,
            input_size=tuple(args.input_size),
            num_classes=args.num_classes,
            augmentation=val_augmentation,
            shuffle=False
        )
        
        # Create evaluator
        evaluator = Evaluator(model.model)
        
        # Generate evaluation report
        metrics = evaluator.generate_report(
            data_loader=test_loader,
            threshold=args.threshold,
            save_path=os.path.join(args.output_dir, args.experiment_name, "evaluation_report.txt")
        )
        
        # Plot metrics distribution
        evaluator.plot_metrics_distribution(
            data_loader=test_loader,
            threshold=args.threshold,
            save_path=os.path.join(args.output_dir, args.experiment_name, "plots", "metrics_distribution.png")
        )
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            data_loader=test_loader,
            threshold=args.threshold,
            save_path=os.path.join(args.output_dir, args.experiment_name, "plots", "confusion_matrix.png")
        )
        
        # Visualize predictions
        evaluator.visualize_predictions(
            data_loader=test_loader,
            num_samples=6,
            threshold=args.threshold,
            save_path=os.path.join(args.output_dir, args.experiment_name, "plots", "predictions_visualization.png")
        )
    
    print(f"\nTraining completed! Results saved to: {args.output_dir}/{args.experiment_name}")


if __name__ == "__main__":
    main()
