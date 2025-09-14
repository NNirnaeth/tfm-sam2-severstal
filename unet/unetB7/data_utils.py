"""
Data utilities for UNet training and evaluation
Handles high-resolution images (256x1600) with proper augmentation
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any


class DataAugmentation:
    """
    Data augmentation pipeline using Albumentations
    Optimized for high-resolution steel defect images
    """
    
    def __init__(self, input_size=(256, 1600), is_training=True):
        """
        Initialize augmentation pipeline
        
        Args:
            input_size: Target image size (height, width)
            is_training: Whether to apply training augmentations
        """
        self.input_size = input_size
        self.is_training = is_training
        
        if is_training:
            self.transform = A.Compose([
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=10, p=0.5),
                A.Transpose(p=0.3),
                
                # Spatial transformations
                A.RandomResizedCrop(
                    size=(input_size[0], input_size[1]),
                    scale=(0.8, 1.0), 
                    ratio=(0.9, 1.1), 
                    p=0.5
                ),
                
                # Ensure all images are resized to target size
                A.Resize(height=input_size[0], width=input_size[1]),
                
                # Color transformations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, 
                    sat_shift_limit=30, 
                    val_shift_limit=20, 
                    p=0.3
                ),
                
                # Noise and blur
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Validation/test augmentations (minimal)
            self.transform = A.Compose([
                A.Resize(height=input_size[0], width=input_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentations to image and mask
        
        Args:
            image: Input image (H, W, C)
            mask: Input mask (H, W) or None
            
        Returns:
            Augmented image and mask
        """
        if mask is not None:
            transformed = self.transform(image=image, mask=mask)
            return transformed['image'], transformed['mask']
        else:
            transformed = self.transform(image=image)
            return transformed['image'], None


class DataLoader(Sequence):
    """
    Custom data loader for UNet training
    Handles high-resolution images efficiently
    """
    
    def __init__(self, 
                 image_paths: List[str], 
                 mask_paths: List[str],
                 batch_size: int = 4,
                 input_size: Tuple[int, int] = (256, 1600),
                 num_classes: int = 1,
                 augmentation: Optional[DataAugmentation] = None,
                 shuffle: bool = True):
        """
        Initialize data loader
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            batch_size: Batch size for training
            input_size: Target image size (height, width)
            num_classes: Number of output classes
            augmentation: Data augmentation pipeline
            shuffle: Whether to shuffle data
        """
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Return number of batches"""
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        """Get batch of data"""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = []
        batch_masks = []
        
        for i in batch_indices:
            # Load image
            image = self._load_image(self.image_paths[i])
            
            # Load mask
            mask = self._load_mask(self.mask_paths[i])
            
            # Apply augmentation if provided
            if self.augmentation is not None:
                image, mask = self.augmentation(image, mask)
            else:
                # Resize without augmentation
                image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
                mask = cv2.resize(mask, (self.input_size[1], self.input_size[0]))
                
                # Normalize image
                image = image.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0
            
            batch_images.append(image)
            batch_masks.append(mask)
        
        # Convert to numpy arrays
        batch_images = np.array(batch_images, dtype=np.float32)
        batch_masks = np.array(batch_masks, dtype=np.float32)
        
        # Add channel dimension to masks if needed
        if len(batch_masks.shape) == 3:
            batch_masks = np.expand_dims(batch_masks, axis=-1)
        
        return batch_images, batch_masks
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # If image is grayscale (all channels identical), convert to proper grayscale
        # and then replicate to 3 channels for pretrained model compatibility
        if len(image.shape) == 3 and np.array_equal(image[:,:,0], image[:,:,1]) and np.array_equal(image[:,:,1], image[:,:,2]):
            # Convert to grayscale first
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Replicate to 3 channels
            image = np.stack([gray, gray, gray], axis=-1)
        
        return image
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load and preprocess mask"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        return mask
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


class DataPreprocessor:
    """
    Data preprocessing utilities
    """
    
    @staticmethod
    def create_data_splits(image_dir: str, 
                          mask_dir: str, 
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          random_state: int = 42) -> Dict[str, List[str]]:
        """
        Create train/validation/test splits
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            test_size: Fraction of data for testing
            val_size: Fraction of remaining data for validation
            random_state: Random seed
            
        Returns:
            Dictionary with train/val/test image and mask paths
        """
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        
        # Verify corresponding masks exist
        valid_pairs = []
        for img_file in image_files:
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            if os.path.exists(os.path.join(mask_dir, mask_file)):
                valid_pairs.append((img_file, mask_file))
        
        print(f"Found {len(valid_pairs)} valid image-mask pairs")
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val_files, test_files = train_test_split(
            valid_pairs, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_size, random_state=random_state
        )
        
        # Create full paths
        splits = {
            'train': {
                'images': [os.path.join(image_dir, f[0]) for f in train_files],
                'masks': [os.path.join(mask_dir, f[1]) for f in train_files]
            },
            'val': {
                'images': [os.path.join(image_dir, f[0]) for f in val_files],
                'masks': [os.path.join(mask_dir, f[1]) for f in val_files]
            },
            'test': {
                'images': [os.path.join(image_dir, f[0]) for f in test_files],
                'masks': [os.path.join(mask_dir, f[1]) for f in test_files]
            }
        }
        
        return splits
    
    @staticmethod
    def visualize_batch(images: np.ndarray, 
                       masks: np.ndarray, 
                       predictions: Optional[np.ndarray] = None,
                       num_samples: int = 4,
                       figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize a batch of images with masks and predictions
        
        Args:
            images: Batch of images (B, H, W, C)
            masks: Batch of ground truth masks (B, H, W, 1)
            predictions: Batch of predicted masks (B, H, W, 1)
            num_samples: Number of samples to display
            figsize: Figure size
        """
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(3 if predictions is not None else 2, num_samples, figsize=figsize)
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(images[i])
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            # Ground truth mask
            axes[1, i].imshow(masks[i].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Ground Truth {i+1}')
            axes[1, i].axis('off')
            
            # Prediction (if available)
            if predictions is not None:
                axes[2, i].imshow(predictions[i].squeeze(), cmap='gray')
                axes[2, i].set_title(f'Prediction {i+1}')
                axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def calculate_class_weights(mask_paths: List[str]) -> Dict[int, float]:
        """
        Calculate class weights for handling class imbalance
        
        Args:
            mask_paths: List of mask file paths
            
        Returns:
            Dictionary with class weights
        """
        total_pixels = 0
        class_pixels = {0: 0, 1: 0}  # Background and foreground
        
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(int)
            
            total_pixels += mask.size
            class_pixels[0] += np.sum(mask == 0)
            class_pixels[1] += np.sum(mask == 1)
        
        # Calculate weights (inverse frequency)
        class_weights = {}
        for class_id, pixel_count in class_pixels.items():
            if pixel_count > 0:
                class_weights[class_id] = float(total_pixels / (2 * pixel_count))
            else:
                class_weights[class_id] = 1.0
        
        return class_weights
