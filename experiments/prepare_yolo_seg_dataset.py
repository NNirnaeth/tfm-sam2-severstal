#!/usr/bin/env python3
"""
Prepare YOLO segmentation dataset from Severstal PNG bitmap annotations - CORRECTED VERSION
Converts PNG bitmap compressed in JSON to YOLO segmentation format

CORRECTED: Now properly uses the 'origin' field from Supervisely bitmap format
- Fixes critical bug where bitmaps were positioned incorrectly  
- Uses cv2 for PNG decoding (same as working detection scripts)
- Properly positions bitmap fragments within the full image using origin coordinates
"""

import os
import json
import base64
import zlib
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import argparse
from tqdm import tqdm

# Check required dependencies
try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    print("Warning: shapely not available, using fallback methods")
    SHAPELY_AVAILABLE = False

def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
    
    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def decode_png_bitmap(annotation_data, img_width, img_height):
    """Decode PNG bitmap from Severstal annotation with proper positioning - CORRECTED VERSION"""
    try:
        # Decode bitmap from base64 and zlib (same as corrected detection script)
        decoded_data = base64.b64decode(annotation_data)
        decompressed_data = zlib.decompress(decoded_data)
        
        # Use cv2 to decode PNG (same as working scripts)
        nparr = np.frombuffer(decompressed_data, np.uint8)
        bitmap_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if bitmap_img is None:
            print("    Warning: Failed to decode bitmap with cv2")
            return None
        
        # Handle different image formats
        if len(bitmap_img.shape) == 3:
            bitmap_img = bitmap_img[:, :, 0]  # Take first channel
        
        # Ensure binary mask
        binary_mask = (bitmap_img > 0).astype(np.uint8)
        
        return binary_mask
        
    except Exception as e:
        print(f"Error decoding annotation: {e}")
        return None

def mask_to_yolo_seg_simple(mask, img_width, img_height):
    """Simple fallback method for converting mask to YOLO segmentation"""
    try:
        # Binarize mask
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Find contours with CHAIN_APPROX_SIMPLE for fewer points
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_segments = []
        
        for contour in contours:
            try:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < 50:
                    continue
                
                # Ensure proper shape
                contour = np.array(contour, dtype=np.int32)
                if len(contour.shape) == 3:
                    contour = contour.reshape(-1, 2)
                
                if len(contour) < 3:
                    continue
                
                # Aggressive simplification: keep only every 3rd point for very complex contours
                if len(contour) > 50:
                    step = len(contour) // 50
                    contour = contour[::step]
                elif len(contour) > 30:
                    step = len(contour) // 30
                    contour = contour[::step]
                elif len(contour) > 20:
                    step = len(contour) // 20
                    contour = contour[::step]
                
                # Ensure we have at least 3 points
                if len(contour) < 3:
                    continue
                
                # Convert to normalized coordinates with reduced precision
                segment = []
                for point in contour:
                    x, y = point[0], point[1]
                    x_norm = max(0.0, min(1.0, x / img_width))
                    y_norm = max(0.0, min(1.0, y / img_height))
                    # Round to 4 decimal places to avoid tiny differences
                    x_norm = round(x_norm, 4)
                    y_norm = round(y_norm, 4)
                    segment.extend([x_norm, y_norm])
                
                if len(segment) >= 6:
                    yolo_segments.append(segment)
                    
            except Exception:
                continue
        
        return yolo_segments
        
    except Exception:
        return []

def mask_to_yolo_seg(mask, img_width, img_height):
    """Convert binary mask to YOLO segmentation format with clean contours"""
    # Try the complex method first
    try:
        result = mask_to_yolo_seg_complex(mask, img_width, img_height)
        if result:
            return result
    except Exception:
        pass
    
    # Fallback to simple method
    return mask_to_yolo_seg_simple(mask, img_width, img_height)

def mask_to_yolo_seg_complex(mask, img_width, img_height):
    """Complex method for converting mask to YOLO segmentation with aggressive Douglas-Peucker"""
    import cv2
    import numpy as np
    
    # Check if shapely is available
    if not SHAPELY_AVAILABLE:
        return []
    
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
    except ImportError:
        return []
    
    # Binarize mask (0/1)
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Remove small particles (area < 50 pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours (external only, ignore holes)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_segments = []
    
    for contour in contours:
        try:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 50:  # Minimum area threshold
                continue
                
            # Ensure contour is numpy array with correct shape
            contour = np.array(contour, dtype=np.int32)
            if len(contour.shape) == 3:
                contour = contour.reshape(-1, 2)
            
            if len(contour) < 3:
                continue
                
            # Validate contour data
            if not np.all(np.isfinite(contour)):
                continue
                
            # Create a mask for just this contour to calculate IoU properly
            contour_mask = np.zeros_like(binary_mask)
            cv2.fillPoly(contour_mask, [contour], 1)
            
            # Aggressive resampling: keep fewer points
            if len(contour) > 100:
                step = len(contour) // 100
                contour_resampled = contour[::step]
            elif len(contour) > 50:
                step = len(contour) // 50
                contour_resampled = contour[::step]
            elif len(contour) > 25:
                step = len(contour) // 25
                contour_resampled = contour[::step]
            else:
                contour_resampled = contour.copy()
            
            if len(contour_resampled) < 3:
                continue
                
            # Ensure contour_resampled is a valid numpy array
            contour_resampled = np.array(contour_resampled, dtype=np.int32)
            if len(contour_resampled.shape) == 3:
                contour_resampled = contour_resampled.reshape(-1, 2)
            
            # Validate resampled contour
            if not np.all(np.isfinite(contour_resampled)) or len(contour_resampled) < 3:
                continue
                
            # More aggressive Douglas-Peucker simplification
            epsilon_start = 0.02  # Start with 2% of perimeter (more aggressive)
            epsilon_max = 0.08    # Max 8% of perimeter (more aggressive)
            
            best_polygon = None
            best_iou = 0
            best_epsilon = epsilon_start
            
            # Binary search for optimal epsilon
            for attempt in range(6):  # More attempts for better optimization
                try:
                    epsilon = epsilon_start + (epsilon_max - epsilon_start) * attempt / 5
                    
                    # Calculate arc length with proper error handling
                    arc_length = cv2.arcLength(contour_resampled, True)
                    if not np.isfinite(arc_length) or arc_length <= 0:
                        continue
                        
                    epsilon_abs = epsilon * arc_length
                    
                    approx = cv2.approxPolyDP(contour_resampled, epsilon_abs, True)
                    
                    if len(approx) < 3:
                        continue
                        
                    # Convert to polygon and validate
                    try:
                        # Convert to shapely polygon for validation
                        coords = approx.reshape(-1, 2)
                        if len(coords) < 3:
                            continue
                            
                        polygon = Polygon(coords)
                        
                        if not polygon.is_valid:
                            # Try to fix invalid polygon
                            polygon = make_valid(polygon)
                            if polygon.is_empty:
                                continue
                        
                        # Calculate IoU between original contour and simplified polygon (per-contour IoU)
                        # Create simplified mask for this specific contour
                        simplified_mask = np.zeros_like(binary_mask)
                        cv2.fillPoly(simplified_mask, [approx], 1)
                        
                        # Calculate IoU between original contour mask and simplified mask
                        intersection = np.logical_and(contour_mask, simplified_mask).sum()
                        union = np.logical_or(contour_mask, simplified_mask).sum()
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_polygon = approx
                            best_epsilon = epsilon
                            
                    except Exception as e:
                        continue
                        
                except Exception as e:
                    continue
            
            # Use best polygon if IoU is acceptable (lowered threshold for more aggressive simplification)
            if best_polygon is not None and best_iou >= 0.90:  # Lowered threshold for more aggressive simplification
                # Limit number of points more aggressively (max 100 instead of 200)
                if len(best_polygon) > 100:
                    # Uniform decimation
                    step = len(best_polygon) // 100
                    best_polygon = best_polygon[::step]
                
                # Remove duplicate points
                unique_points = []
                for point in best_polygon:
                    try:
                        if len(point.shape) > 1:
                            point_tuple = tuple(point[0])
                        else:
                            point_tuple = tuple(point)
                        if point_tuple not in unique_points:
                            unique_points.append(point_tuple)
                    except Exception:
                        continue
                
                if len(unique_points) >= 3:
                    # Convert to normalized coordinates with reduced precision
                    segment = []
                    for x, y in unique_points:
                        try:
                            x_norm = max(0.0, min(1.0, x / img_width))
                            y_norm = max(0.0, min(1.0, y / img_height))
                            # Round to 4 decimal places to avoid tiny differences
                            x_norm = round(x_norm, 4)
                            y_norm = round(y_norm, 4)
                            segment.extend([x_norm, y_norm])
                        except Exception:
                            continue
                    
                    if len(segment) >= 6:  # At least 3 points (6 coordinates)
                        yolo_segments.append(segment)
                        
        except Exception as e:
            # Skip this contour if any error occurs
            continue
    
    return yolo_segments

def analyze_polygon_quality(labels_dir, images_dir, sample_size=100):
    """Analyze polygon quality and provide statistics"""
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    if len(label_files) > sample_size:
        label_files = random.sample(label_files, sample_size)
    
    stats = {
        'num_points': [],
        'iou_scores': [],
        'area_ratios': [],
        'perimeter_ratios': []
    }
    
    print(f"Analyzing {len(label_files)} label files...")
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        img_file = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(images_dir, img_file)
        
        if not os.path.exists(img_path):
            continue
            
        # Load image and get dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Load labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 7:  # class + at least 3 points
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    # Count points
                    num_points = len(coords) // 2
                    stats['num_points'].append(num_points)
    
    # Print statistics
    print("\n=== POLYGON QUALITY STATISTICS ===")
    print(f"Total instances analyzed: {len(stats['num_points'])}")
    print(f"Points per instance:")
    print(f"  Mean: {np.mean(stats['num_points']):.1f}")
    print(f"  Median: {np.median(stats['num_points']):.1f}")
    print(f"  P95: {np.percentile(stats['num_points'], 95):.1f}")
    print(f"  Max: {np.max(stats['num_points'])}")
    print(f"  Min: {np.min(stats['num_points'])}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stats['num_points'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.median(stats['num_points']), color='red', linestyle='--', label=f'Median: {np.median(stats["num_points"]):.1f}')
    plt.xlabel('Number of Points per Polygon')
    plt.ylabel('Frequency')
    plt.title('Distribution of Polygon Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(labels_dir), 'polygon_quality_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nQuality analysis plot saved to: {plot_path}")
    
    return stats

def visualize_sample_conversions(labels_dir, images_dir, num_samples=5):
    """Visualize sample conversions from bitmap to polygon"""
    import random
    import matplotlib.pyplot as plt
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    if len(label_files) > num_samples:
        label_files = random.sample(label_files, num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, label_file in enumerate(label_files):
        label_path = os.path.join(labels_dir, label_file)
        img_file = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(images_dir, img_file)
        
        if not os.path.exists(img_path):
            continue
            
        # Load image
        img = np.array(Image.open(img_path))
        
        # Load labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Create polygon visualization
        img_with_polygons = img.copy()
        
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 7:
                    coords = [float(x) for x in parts[1:]]
                    points = []
                    
                    for j in range(0, len(coords), 2):
                        x = int(coords[j] * img.shape[1])
                        y = int(coords[j+1] * img.shape[0])
                        points.append([x, y])
                    
                    if len(points) >= 3:
                        points = np.array(points, dtype=np.int32)
                        cv2.polylines(img_with_polygons, [points], True, (0, 255, 0), 2)
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original: {img_file}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img_with_polygons)
        axes[i, 1].set_title(f'Polygons: {len(lines)} instances')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(os.path.dirname(labels_dir), 'sample_conversions.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Sample conversions visualization saved to: {viz_path}")
    
    return viz_path

def process_image_annotations(img_path, ann_path, output_dir, img_id):
    """Process single image and its annotations"""
    try:
        # Load image
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Load annotations
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        
        # Create YOLO label file
        label_filename = f"{img_id}.txt"
        label_path = os.path.join(output_dir, label_filename)
        
        yolo_lines = []
        
        for obj in annotations.get('objects', []):
            try:
                # Check if the object is any type of defect (defect_1, defect_2, defect_3, etc.)
                if 'defect' in obj.get('classTitle', '').lower():
                    # Get bitmap data and origin
                    bitmap_data = obj.get('bitmap', {}).get('data')
                    bitmap_origin = obj.get('bitmap', {}).get('origin', [0, 0])
                    
                    if bitmap_data:
                        # Decode mask
                        mask = decode_png_bitmap(bitmap_data, img_width, img_height)
                        if mask is not None:
                            # Create full-size mask and place the bitmap at correct origin
                            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            
                            # Get bitmap dimensions
                            mask_h, mask_w = mask.shape
                            x0, y0 = bitmap_origin
                            
                            # Ensure coordinates are within image bounds
                            x0 = max(0, min(x0, img_width - 1))
                            y0 = max(0, min(y0, img_height - 1))
                            x1 = min(x0 + mask_w, img_width)
                            y1 = min(y0 + mask_h, img_height)
                            
                            # Adjust mask dimensions if needed
                            actual_mask_h = y1 - y0
                            actual_mask_w = x1 - x0
                            
                            if actual_mask_h > 0 and actual_mask_w > 0:
                                # Place the mask at correct position
                                full_mask[y0:y1, x0:x1] = mask[:actual_mask_h, :actual_mask_w]
                                
                                # Convert to YOLO format
                                segments = mask_to_yolo_seg(full_mask, img_width, img_height)
                                
                                for segment in segments:
                                    if len(segment) >= 6:  # At least 3 points
                                        # YOLO format: class_id + normalized coordinates
                                        line = f"0 {' '.join([f'{coord:.6f}' for coord in segment])}"
                                        yolo_lines.append(line)
            except Exception as e:
                # Skip this object if there's an error
                continue
        
        # Write label file
        if yolo_lines:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
        
        return len(yolo_lines)
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Prepare YOLO segmentation dataset from Severstal')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/ptp/sam2/datasets/Data/splits',
                       help='Path to Severstal data directory')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/ptp/sam2/datasets/yolo_segmentation',
                       help='Output directory for YOLO segmentation dataset')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again")
        return
    
    # Create output directories
    train_output = os.path.join(args.output_dir, 'labels', 'train')
    val_output = os.path.join(args.output_dir, 'labels', 'val')
    train_img_output = os.path.join(args.output_dir, 'images', 'train')
    val_img_output = os.path.join(args.output_dir, 'images', 'val')
    
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    os.makedirs(train_img_output, exist_ok=True)
    os.makedirs(val_img_output, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    train_img_dir = os.path.join(args.data_dir, 'train_split', 'img')
    train_ann_dir = os.path.join(args.data_dir, 'train_split', 'ann')
    
    print(f"Training image directory: {train_img_dir}")
    print(f"Training annotation directory: {train_ann_dir}")
    
    if not os.path.exists(train_img_dir):
        print(f"ERROR: Training image directory not found: {train_img_dir}")
        return
    
    if not os.path.exists(train_ann_dir):
        print(f"ERROR: Training annotation directory not found: {train_ann_dir}")
        return
    
    train_images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    print(f"Found {len(train_images)} training images")
    train_count = 0
    train_errors = 0
    
    for img_file in tqdm(train_images, desc="Training"):
        try:
            img_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(train_img_dir, img_file)
            # Fixed: annotation files have .jpg.json extension as copied by 00_prepare_data.py
            ann_path = os.path.join(train_ann_dir, f"{img_id}.jpg.json")
            
            if os.path.exists(ann_path):
                # Copy image
                import shutil
                shutil.copy2(img_path, os.path.join(train_img_output, img_file))
                
                # Process annotations
                count = process_image_annotations(img_path, ann_path, train_output, img_id)
                train_count += count
            else:
                print(f"Warning: Annotation not found for {img_file}")
        except Exception as e:
            print(f"Error processing training image {img_file}: {e}")
            train_errors += 1
            continue
    
    print(f"Training completed: {train_count} annotations, {train_errors} errors")
    
    # Process validation data
    print("Processing validation data...")
    val_img_dir = os.path.join(args.data_dir, 'val_split', 'img')
    val_ann_dir = os.path.join(args.data_dir, 'val_split', 'ann')
    
    print(f"Validation image directory: {val_img_dir}")
    print(f"Validation annotation directory: {val_ann_dir}")
    
    if not os.path.exists(val_img_dir):
        print(f"ERROR: Validation image directory not found: {val_img_dir}")
        return
    
    if not os.path.exists(val_ann_dir):
        print(f"ERROR: Validation annotation directory not found: {val_ann_dir}")
        return
    
    val_images = [f for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    print(f"Found {len(val_images)} validation images")
    val_count = 0
    val_errors = 0
    
    for img_file in tqdm(val_images, desc="Validation"):
        try:
            img_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(val_img_dir, img_file)
            # Fixed: annotation files have .jpg.json extension as copied by 00_prepare_data.py
            ann_path = os.path.join(val_ann_dir, f"{img_id}.jpg.json")
            
            if os.path.exists(ann_path):
                # Copy image
                import shutil
                shutil.copy2(img_path, os.path.join(val_img_output, img_file))
                
                # Process annotations
                count = process_image_annotations(img_path, ann_path, val_output, img_id)
                val_count += count
            else:
                print(f"Warning: Annotation not found for {img_file}")
        except Exception as e:
            print(f"Error processing validation image {img_file}: {e}")
            val_errors += 1
            continue
    
    print(f"Validation completed: {val_count} annotations, {val_errors} errors")
    
    # Create YAML configuration
    yaml_content = f"""path: {args.output_dir}  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Val images (relative to 'path')

nc: 1  # Number of classes
names: ['defect']  # Class names
"""
    
    yaml_path = os.path.join(args.output_dir, 'yolo_segmentation.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset preparation completed!")
    print(f"Training: {len(train_images)} images, {train_count} annotations")
    print(f"Validation: {len(val_images)} images, {val_count} annotations")
    print(f"Output directory: {args.output_dir}")
    print(f"YAML config: {yaml_path}")
    
    # Quality analysis and visualization
    print("\n=== PERFORMING QUALITY ANALYSIS ===")
    
    # Analyze training data quality
    train_labels_dir = os.path.join(args.output_dir, 'labels', 'train')
    train_images_dir = os.path.join(args.output_dir, 'images', 'train')
    
    if os.path.exists(train_labels_dir) and os.path.exists(train_images_dir):
        print("\nAnalyzing training data polygon quality...")
        try:
            train_stats = analyze_polygon_quality(train_labels_dir, train_images_dir, sample_size=200)
            
            # Visualize sample conversions
            print("\nCreating sample conversion visualizations...")
            viz_path = visualize_sample_conversions(train_labels_dir, train_images_dir, num_samples=10)
            
            print(f"\n Quality analysis completed!")
            print(f" Statistics: {len(train_stats['num_points'])} instances analyzed")
            print(f" Visualization: {viz_path}")
            
        except Exception as e:
            print(f" Quality analysis failed: {e}")
            print("Continuing without quality analysis...")
    else:
        print(" Training directories not found, skipping quality analysis")

if __name__ == "__main__":
    main()
