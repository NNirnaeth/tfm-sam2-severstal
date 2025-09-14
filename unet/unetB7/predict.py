#!/usr/bin/env python3
"""
Prediction script for UNet with EfficientNet-B7
Generate predictions on new images
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

# Add the libs directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from unetB7.model import UNetEfficientNetB7
from unetB7.data_utils import DataAugmentation


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict with UNet and EfficientNet-B7')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 1600],
                       help='Input image size (height width)')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of output classes')
    
    # Input arguments
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Output directory for predictions')
    
    # Prediction arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary predictions')
    parser.add_argument('--save_masks', action='store_true',
                       help='Save predicted masks as images')
    parser.add_argument('--save_overlay', action='store_true',
                       help='Save overlay of predictions on original images')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for prediction')
    
    return parser.parse_args()


def load_image(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image file
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_size = image.shape[:2]
    
    # Resize to target size
    image_resized = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    return image_normalized, original_size


def postprocess_prediction(prediction: np.ndarray, 
                          original_size: Tuple[int, int],
                          threshold: float = 0.5) -> np.ndarray:
    """
    Postprocess prediction mask
    
    Args:
        prediction: Raw prediction from model
        original_size: Original image size (height, width)
        threshold: Threshold for binary classification
        
    Returns:
        Postprocessed mask
    """
    # Apply threshold
    binary_mask = (prediction > threshold).astype(np.uint8)
    
    # Resize to original size
    mask_resized = cv2.resize(
        binary_mask, 
        (original_size[1], original_size[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    return mask_resized


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create overlay of mask on original image
    
    Args:
        image: Original image
        mask: Binary mask
        alpha: Transparency of overlay
        
    Returns:
        Overlay image
    """
    # Convert image to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(image_uint8)
    colored_mask[mask > 0] = [255, 0, 0]  # Red color for defects
    
    # Create overlay
    overlay = cv2.addWeighted(image_uint8, 1-alpha, colored_mask, alpha, 0)
    
    return overlay


def predict_single_image(model: UNetEfficientNetB7, 
                        image_path: str, 
                        target_size: Tuple[int, int],
                        threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict on a single image
    
    Args:
        model: Trained model
        image_path: Path to image
        target_size: Target size for model input
        threshold: Threshold for binary classification
        
    Returns:
        Tuple of (original_image, prediction, binary_mask)
    """
    # Load and preprocess image
    image_normalized, original_size = load_image(image_path, target_size)
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    # Make prediction
    prediction = model.predict(image_batch)[0]
    
    # Postprocess prediction
    binary_mask = postprocess_prediction(prediction, original_size, threshold)
    
    # Load original image for visualization
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    return original_image, prediction, binary_mask


def main():
    """Main prediction function"""
    args = parse_args()
    
    # Print configuration
    print("Prediction Configuration:")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("="*50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model = UNetEfficientNetB7(
        input_shape=(args.input_size[0], args.input_size[1], 3),
        num_classes=args.num_classes
    )
    
    model.load_model(args.model_path)
    print(f"Model loaded from: {args.model_path}")
    
    # Get input files
    if os.path.isfile(args.input_path):
        input_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        input_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            input_files.extend(Path(args.input_path).glob(ext))
        input_files = [str(f) for f in input_files]
    else:
        raise ValueError(f"Input path not found: {args.input_path}")
    
    print(f"Found {len(input_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(input_files):
        print(f"\nProcessing {i+1}/{len(input_files)}: {os.path.basename(image_path)}")
        
        try:
            # Predict
            original_image, prediction, binary_mask = predict_single_image(
                model, image_path, tuple(args.input_size), args.threshold
            )
            
            # Save results
            base_name = Path(image_path).stem
            
            # Save mask if requested
            if args.save_masks:
                mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, binary_mask * 255)
                print(f"  Mask saved to: {mask_path}")
            
            # Save overlay if requested
            if args.save_overlay:
                overlay = create_overlay(original_image, binary_mask)
                overlay_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"  Overlay saved to: {overlay_path}")
            
            # Calculate statistics
            defect_pixels = np.sum(binary_mask > 0)
            total_pixels = binary_mask.size
            defect_percentage = (defect_pixels / total_pixels) * 100
            
            print(f"  Defect pixels: {defect_pixels} ({defect_percentage:.2f}%)")
            
        except Exception as e:
            print(f"  Error processing {image_path}: {str(e)}")
            continue
    
    print(f"\nPrediction completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
