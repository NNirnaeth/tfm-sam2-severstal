"""
Adapter for Severstal Steel Defect Dataset
Converts JSON annotations with bitmap data to image masks
"""

import os
import json
import base64
import zlib
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
from PIL import Image
import matplotlib.pyplot as plt


class SeverstalAdapter:
    """
    Adapter for Severstal Steel Defect Dataset
    Handles JSON annotations with bitmap data and converts to image masks
    """
    
    def __init__(self, image_dir: str, annotation_dir: str, output_mask_dir: str):
        """
        Initialize adapter
        
        Args:
            image_dir: Directory containing original images
            annotation_dir: Directory containing JSON annotations
            output_mask_dir: Directory to save converted masks
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.output_mask_dir = output_mask_dir
        
        # Create output directory
        os.makedirs(output_mask_dir, exist_ok=True)
        
    def decode_bitmap(self, bitmap_data: str, origin: List[int], image_size: Tuple[int, int]) -> np.ndarray:
        """
        Decode bitmap data from JSON annotation
        
        Args:
            bitmap_data: Base64 encoded bitmap data
            origin: [x, y] coordinates of bitmap origin
            image_size: (height, width) of the image
            
        Returns:
            Binary mask as numpy array
        """
        # Decode base64 data
        compressed_data = base64.b64decode(bitmap_data)
        
        # Decompress using zlib
        decompressed_data = zlib.decompress(compressed_data)
        
        # Convert to numpy array
        mask_data = np.frombuffer(decompressed_data, dtype=np.uint8)
        
        # The bitmap data is stored as a 1D array, we need to reshape it
        # The size is typically stored in the first few bytes
        if len(mask_data) > 4:
            # Try to determine the bitmap dimensions
            # This is a heuristic - you might need to adjust based on your data
            bitmap_width = int(np.sqrt(len(mask_data)))
            bitmap_height = len(mask_data) // bitmap_width
            
            # Reshape to 2D
            bitmap = mask_data[:bitmap_width * bitmap_height].reshape(bitmap_height, bitmap_width)
            
            # Create full image mask
            full_mask = np.zeros(image_size, dtype=np.uint8)
            
            # Place bitmap at origin
            x_start, y_start = origin
            x_end = min(x_start + bitmap_width, image_size[1])
            y_end = min(y_start + bitmap_height, image_size[0])
            
            # Ensure we don't go out of bounds
            bitmap_h = y_end - y_start
            bitmap_w = x_end - x_start
            
            if bitmap_h > 0 and bitmap_w > 0:
                full_mask[y_start:y_end, x_start:x_end] = bitmap[:bitmap_h, :bitmap_w]
            
            return full_mask
        else:
            # Return empty mask if data is too small
            return np.zeros(image_size, dtype=np.uint8)
    
    def process_annotation(self, annotation_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single JSON annotation file
        
        Args:
            annotation_path: Path to JSON annotation file
            
        Returns:
            Tuple of (combined_mask, metadata)
        """
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        # Get image dimensions
        height = annotation['size']['height']
        width = annotation['size']['width']
        image_size = (height, width)
        
        # Create combined mask for all defects
        combined_mask = np.zeros(image_size, dtype=np.uint8)
        
        # Process each object/defect
        defect_count = 0
        for obj in annotation['objects']:
            if obj['geometryType'] == 'bitmap':
                bitmap_data = obj['bitmap']['data']
                origin = obj['bitmap']['origin']
                
                # Decode bitmap
                defect_mask = self.decode_bitmap(bitmap_data, origin, image_size)
                
                # Add to combined mask
                combined_mask = np.maximum(combined_mask, defect_mask)
                defect_count += 1
        
        metadata = {
            'image_size': image_size,
            'defect_count': defect_count,
            'class_titles': [obj['classTitle'] for obj in annotation['objects']],
            'total_objects': len(annotation['objects'])
        }
        
        return combined_mask, metadata
    
    def convert_dataset(self, image_extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> Dict[str, Any]:
        """
        Convert entire dataset from JSON annotations to image masks
        
        Args:
            image_extensions: List of image file extensions to process
            
        Returns:
            Dictionary with conversion statistics
        """
        # Get all annotation files
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.json')]
        annotation_files.sort()
        
        print(f"Found {len(annotation_files)} annotation files")
        print(f"Looking for images in: {self.image_dir}")
        print(f"Looking for annotations in: {self.annotation_dir}")
        
        # Debug: List some files
        if len(annotation_files) > 0:
            print(f"Sample annotation files: {annotation_files[:3]}")
        
        # Check if image directory exists and list some files
        if os.path.exists(self.image_dir):
            image_files = [f for f in os.listdir(self.image_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
            print(f"Found {len(image_files)} image files")
            if len(image_files) > 0:
                print(f"Sample image files: {image_files[:3]}")
        else:
            print(f"ERROR: Image directory does not exist: {self.image_dir}")
            return {'total_files': 0, 'successful_conversions': 0, 'failed_conversions': 0, 'total_defects': 0, 'defect_types': []}
        
        # Statistics
        stats = {
            'total_files': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_defects': 0,
            'defect_types': set()
        }
        
        # Get all image files for matching
        all_image_files = [f for f in os.listdir(self.image_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        for ann_file in annotation_files:
            try:
                # Get corresponding image file
                # Handle both .json and .jpg.json extensions
                if ann_file.endswith('.jpg.json'):
                    base_name = ann_file.replace('.jpg.json', '')
                    image_file = base_name + '.jpg'
                else:
                    base_name = ann_file.replace('.json', '')
                    image_file = None
                    
                    # Try to find matching image
                    for ext in image_extensions:
                        potential_image = base_name + ext
                        if os.path.exists(os.path.join(self.image_dir, potential_image)):
                            image_file = potential_image
                            break
                
                if image_file is None:
                    if stats['failed_conversions'] < 5:  # Only show first 5 warnings
                        print(f"Warning: No corresponding image found for {ann_file}")
                    stats['failed_conversions'] += 1
                    continue
                
                # Process annotation
                mask, metadata = self.process_annotation(os.path.join(self.annotation_dir, ann_file))
                
                # Save mask
                mask_filename = base_name + '_mask.png'
                mask_path = os.path.join(self.output_mask_dir, mask_filename)
                cv2.imwrite(mask_path, mask * 255)  # Convert to 0-255 range
                
                # Update statistics
                stats['total_files'] += 1
                stats['successful_conversions'] += 1
                stats['total_defects'] += metadata['defect_count']
                stats['defect_types'].update(metadata['class_titles'])
                
                if stats['successful_conversions'] % 100 == 0:
                    print(f"Processed {stats['successful_conversions']} files...")
                
            except Exception as e:
                print(f"Error processing {ann_file}: {str(e)}")
                stats['failed_conversions'] += 1
        
        # Convert set to list for JSON serialization
        stats['defect_types'] = list(stats['defect_types'])
        
        print(f"\nConversion completed:")
        print(f"Total files: {stats['total_files']}")
        print(f"Successful: {stats['successful_conversions']}")
        print(f"Failed: {stats['failed_conversions']}")
        print(f"Total defects: {stats['total_defects']}")
        print(f"Defect types: {stats['defect_types']}")
        
        return stats
    
    def visualize_sample(self, image_filename: str, num_samples: int = 4):
        """
        Visualize sample images with their masks
        
        Args:
            image_filename: Name of image file to visualize
            num_samples: Number of samples to show
        """
        # Get all image files
        image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        
        # Select samples
        sample_files = image_files[:num_samples]
        
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i, img_file in enumerate(sample_files):
            # Load original image
            img_path = os.path.join(self.image_dir, img_file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load corresponding mask
            base_name = img_file.split('.')[0]
            mask_file = base_name + '_mask.png'
            mask_path = os.path.join(self.output_mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Display image
            axes[0, i].imshow(image)
            axes[0, i].set_title(f'Image: {img_file}')
            axes[0, i].axis('off')
            
            # Display mask
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f'Mask: {mask_file}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_data_splits(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Dict[str, List[str]]:
        """
        Create train/validation/test splits for the converted dataset
        
        Args:
            test_size: Fraction of data for testing
            val_size: Fraction of remaining data for validation
            random_state: Random seed
            
        Returns:
            Dictionary with train/val/test image and mask paths
        """
        from sklearn.model_selection import train_test_split
        
        # Get all image files
        image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        
        # Verify corresponding masks exist
        valid_pairs = []
        for img_file in image_files:
            base_name = img_file.split('.')[0]
            mask_file = base_name + '_mask.png'
            mask_path = os.path.join(self.output_mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                valid_pairs.append((img_file, mask_file))
        
        print(f"Found {len(valid_pairs)} valid image-mask pairs")
        
        # Split data
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
                'images': [os.path.join(self.image_dir, f[0]) for f in train_files],
                'masks': [os.path.join(self.output_mask_dir, f[1]) for f in train_files]
            },
            'val': {
                'images': [os.path.join(self.image_dir, f[0]) for f in val_files],
                'masks': [os.path.join(self.output_mask_dir, f[1]) for f in val_files]
            },
            'test': {
                'images': [os.path.join(self.image_dir, f[0]) for f in test_files],
                'masks': [os.path.join(self.output_mask_dir, f[1]) for f in test_files]
            }
        }
        
        return splits


def main():
    """
    Example usage of SeverstalAdapter
    """
    # Example paths - adjust according to your dataset structure
    image_dir = "/path/to/severstal/images"
    annotation_dir = "/path/to/severstal/annotations"
    output_mask_dir = "/path/to/severstal/masks"
    
    # Create adapter
    adapter = SeverstalAdapter(image_dir, annotation_dir, output_mask_dir)
    
    # Convert dataset
    stats = adapter.convert_dataset()
    
    # Visualize samples
    adapter.visualize_sample("sample.jpg", num_samples=4)
    
    # Create data splits
    splits = adapter.create_data_splits()
    
    print("Dataset conversion completed successfully!")


if __name__ == "__main__":
    main()
