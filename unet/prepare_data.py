#!/usr/bin/env python3
"""
Data preparation module for UNet training on Severstal dataset
Loads and prepares data for training/validation

TFM Requirements:
- Horizontal aspect preservation (resize_longer + pad to 1024×256)
- No vertical flip (preserves steel rolling physics)
- Limited rotation (±5-10°) to avoid defect distortion
- Specific augmentation for steel defects
- Stratification by defect presence
"""

import os
import json
import cv2
import base64
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SeverstalDataset(Dataset):
    """Dataset for Severstal steel defect segmentation"""
    
    def __init__(self, data_path, annotation_path, transform=None, is_training=True):
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.transform = transform
        self.is_training = is_training
        
        # Load data
        self.data = self._load_data()
        
        # Stratified sampling for training (TFM requirement)
        if is_training:
            self.data = self._stratified_sampling()
    
    def _load_data(self):
        """Load image and annotation paths"""
        data = []
        
        # Get all image files (only .jpg for Severstal dataset)
        image_files = [f for f in os.listdir(self.data_path) if f.endswith('.jpg')]
        
        print(f"Found {len(image_files)} image files in {self.data_path}")
        
        for img_file in image_files:
            img_path = os.path.join(self.data_path, img_file)
            ann_file = img_file + '.json'  # .jpg + .json = .jpg.json
            ann_path = os.path.join(self.annotation_path, ann_file)
            
            if os.path.exists(ann_path):
                data.append({
                    'image': img_path,
                    'annotation': ann_path
                })
            else:
                print(f"Warning: No annotation found for {img_file}")
        
        print(f"Successfully paired {len(data)} image-annotation pairs")
        
        if len(data) == 0:
            raise ValueError(f"No valid image-annotation pairs found in {self.data_path} and {self.annotation_path}")
        
        return data
    
    def _stratified_sampling(self):
        """Stratified sampling based on defect presence (TFM requirement)"""
        # Check which images have defects
        has_defects = []
        for item in self.data:
            try:
                with open(item['annotation'], 'r') as f:
                    ann_data = json.load(f)
                has_defects.append(len(ann_data.get('objects', [])) > 0)
            except:
                has_defects.append(False)
        
        # Stratified split (TFM requirement for balanced training)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, _ in skf.split(self.data, has_defects):
            break  # Take first fold
        
        return [self.data[i] for i in train_idx]
    
    def _decode_bitmap_annotation(self, annotation_path):
        """Decode bitmap annotation from Supervisely format (TFM requirement)"""
        try:
            with open(annotation_path, 'r') as f:
                json_data = json.load(f)
            
            height = json_data["size"]["height"]
            width = json_data["size"]["width"]
            ann_map = np.zeros((height, width), dtype=np.uint8)
            
            for obj in json_data.get("objects", []):
                if obj.get("geometryType") != "bitmap":
                    continue
                
                bitmap_data = obj["bitmap"]["data"]
                origin_x, origin_y = obj["bitmap"]["origin"]
                
                # Decode bitmap data (PNG compressed in JSON - TFM requirement)
                try:
                    decoded_data = base64.b64decode(bitmap_data)
                    decompressed_data = zlib.decompress(decoded_data)
                    
                    # Convert PNG data to numpy array
                    nparr = np.frombuffer(decompressed_data, np.uint8)
                    sub_mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    
                    if sub_mask is not None:
                        if len(sub_mask.shape) == 3:
                            sub_mask = sub_mask[:, :, 0]  # Take first channel
                        sub_mask = (sub_mask > 0).astype(np.uint8)
                    else:
                        continue
                        
                except Exception as e:
                    continue
                
                end_y = origin_y + sub_mask.shape[0]
                end_x = origin_x + sub_mask.shape[1]
                
                if end_y > height or end_x > width:
                    continue
                
                ann_map[origin_y:end_y, origin_x:end_x] = np.maximum(
                    ann_map[origin_y:end_y, origin_x:end_x], sub_mask)
            
            return ann_map
            
        except Exception as e:
            print(f"Error decoding annotation {annotation_path}: {e}")
            return np.zeros((256, 1024), dtype=np.uint8)
    
    def _resize_maintaining_aspect(self, image, target_size=(1024, 256)):
        """Resize image maintaining horizontal aspect ratio with padding (TFM requirement)"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor (preserve horizontal aspect - TFM requirement)
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded
    
    def _resize_mask_maintaining_aspect(self, mask, target_size=(1024, 256)):
        """Resize mask maintaining horizontal aspect ratio with padding (TFM requirement)"""
        h, w = mask.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor (preserve horizontal aspect - TFM requirement)
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize mask
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create padded mask
        padded = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized mask
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = cv2.imread(item['image'])
        if image is None:
            print(f"Error reading image: {item['image']}")
            # Return dummy data
            image = np.zeros((256, 1024, 3), dtype=np.uint8)
            mask = np.zeros((256, 1024), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load and decode annotation
            mask = self._decode_bitmap_annotation(item['annotation'])
            
            # Resize maintaining horizontal aspect ratio (TFM requirement)
            image = self._resize_maintaining_aspect(image, (1024, 256))
            mask = self._resize_mask_maintaining_aspect(mask, (1024, 256))
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensor and normalize
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        elif isinstance(image, torch.Tensor):
            # If already a tensor, just normalize
            if image.dtype != torch.float32:
                image = image.float()
            if image.max() > 1.0:
                image = image / 255.0
            if image.shape[0] != 3:  # If channels are not first
                image = image.permute(2, 0, 1)
        
        # Handle mask (could be numpy array or tensor from transforms)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        elif isinstance(mask, torch.Tensor):
            if mask.dtype != torch.float32:
                mask = mask.float()
        
        return image, mask


def get_transforms(img_size=(1024, 256)):
    """Get data augmentation transforms (TFM-specific for steel defects)"""
    
    # TFM requirements: No vertical flip (preserves steel rolling physics)
    # Limited rotation (±5-10°) to avoid defect distortion
    # Specific augmentation for steel defects
    
    train_transform = A.Compose([
        # Horizontal flip only (no vertical flip - TFM requirement)
        A.HorizontalFlip(p=0.5),
        
        # Limited rotation to preserve defect shape (TFM requirement)
        A.Affine(
            rotate=(-10, 10),  # ±10° max (TFM requirement)
            scale=(0.9, 1.1),  # Mild scaling
            translate_percent=(-0.1, 0.1),  # Mild translation
            p=0.5
        ),
        
        # Mild geometric transformations (avoid strong elastic that deforms defects)
        A.OneOf([
            A.GridDistortion(num_steps=3, distort_limit=0.1, p=1),  # Mild distortion
            A.OpticalDistortion(distort_limit=0.5, p=1),  # Mild optical (removed shift_limit)
        ], p=0.3),
        
        # Noise and blur (mild for steel defects)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), p=1),  # Mild noise (correct parameter)
            A.ISONoise(color_shift=(0.01, 0.05), p=1),  # Mild ISO noise
        ], p=0.3),
        
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1),  # Mild motion blur
            A.MedianBlur(blur_limit=3, p=1),  # Mild median blur
        ], p=0.2),
        
        # Color adjustments (mild for steel defects)
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=1),  # Mild contrast enhancement
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # Mild brightness change
                contrast_limit=0.1,    # Mild contrast change
                p=1
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1),  # Mild gamma
        ], p=0.3),
        
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        ToTensorV2(),
    ])
    
    return train_transform, val_transform


def create_data_loaders(train_path, val_path, batch_size=8, img_size=(1024, 256), num_workers=4):
    """Create training and validation data loaders"""
    
    # Get transforms (TFM-specific)
    train_transform, val_transform = get_transforms(img_size)
    
    # Create datasets
    train_dataset = SeverstalDataset(
        os.path.join(train_path, 'img'), 
        os.path.join(train_path, 'ann'),
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = SeverstalDataset(
        os.path.join(val_path, 'img'),
        os.path.join(val_path, 'ann'),
        transform=val_transform,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, train_size, val_size = create_data_loaders(
        train_path='/home/ptp/sam2/datasets/Data/splits/train_split',
        val_path='/home/ptp/sam2/datasets/Data/splits/val_split',
        batch_size=4
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Test one batch
    for images, masks in train_loader:
        print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
        break
