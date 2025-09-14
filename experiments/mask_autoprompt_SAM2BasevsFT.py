#!/usr/bin/env python3
"""
Professional SAM2 Auto-prompt Visualization Tool

This script provides efficient parallel processing for visualizing SAM2 model predictions
using auto-prompt mode. It supports both base and fine-tuned models with automatic
configuration detection.

Features:
- Parallel processing with configurable workers
- Automatic model size detection from checkpoint paths
- Professional error handling and logging
- Comprehensive visualization with TP/FP/FN analysis
- JSON summary with detailed statistics

Usage:
    python mask_autoprompt_SAM2BasevsFT.py --images /path/to/images --annotations /path/to/ann
    python mask_autoprompt_SAM2BasevsFT.py --checkpoint /path/to/checkpoint.torch --images /path/to/images --annotations /path/to/ann
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import base64
import zlib
import io
import argparse
import random
import time
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add paths
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.append(str(project_root / "libs" / "sam2base"))
sys.path.append(str(project_root / "new_src" / "utils"))

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2
except ImportError as e:
    print(f"Error importing SAM2 modules: {e}")
    print("Please ensure SAM2 is properly installed and paths are correct")
    sys.exit(1)

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Setup professional logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'sam2_visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class SAM2Visualizer:
    """Professional SAM2 visualization class with parallel processing"""
    
    def __init__(self, checkpoint_path=None, model_size=None, confidence_threshold=None, 
                 images_dir=None, annotations_dir=None, output_dir=None, num_workers=4):
        """
        Initialize SAM2 Visualizer
        
        Args:
            checkpoint_path: Path to SAM2 checkpoint (None for base model)
            model_size: Model size ('small' or 'large', auto-detected if None)
            confidence_threshold: Confidence threshold (auto-set if None)
            images_dir: Directory containing test images
            annotations_dir: Directory containing test annotations
            output_dir: Output directory for visualizations
            num_workers: Number of parallel workers
        """
        self.checkpoint_path = checkpoint_path
        self.model_size = model_size or self._detect_model_size()
        self.confidence_threshold = confidence_threshold or self._get_confidence_threshold()
        self.images_dir = Path(images_dir) if images_dir else None
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.num_workers = min(num_workers, cpu_count())
        
        self.model = None
        self.predictor = None
        self.model_name = self._get_model_name()
        
        logger.info(f"Initialized SAM2 Visualizer: {self.model_name}")
        logger.info(f"Model size: {self.model_size}, Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Workers: {self.num_workers}")
    
    def _detect_model_size(self):
        """Detect model size from checkpoint path"""
        if self.checkpoint_path is None:
            return "large"  # Default for base models
        
        checkpoint_name = Path(self.checkpoint_path).name.lower()
        parent_dir = Path(self.checkpoint_path).parent.name.lower()
        
        if any(keyword in checkpoint_name for keyword in ["small", "s_"]):
            return "small"
        elif any(keyword in checkpoint_name for keyword in ["large", "l_"]):
            return "large"
        elif any(keyword in parent_dir for keyword in ["small", "s_"]):
            return "small"
        elif any(keyword in parent_dir for keyword in ["large", "l_"]):
            return "large"
        else:
            logger.warning(f"Could not detect model size from {self.checkpoint_path}, defaulting to 'large'")
            return "large"
    
    def _get_confidence_threshold(self):
        """Get appropriate confidence threshold based on model type"""
        if self.checkpoint_path is None:
            return 0.3  # Lower threshold for base models
        else:
            return 0.7  # Higher threshold for fine-tuned models
    
    def _get_model_name(self):
        """Get descriptive model name"""
        model_type = "Base" if self.checkpoint_path is None else "Fine-tuned"
        return f"SAM2 {self.model_size.title()} {model_type}"
    
    def _validate_paths(self):
        """Validate all required paths exist"""
        if self.images_dir is None or not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        if self.annotations_dir is None or not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")
        
        if self.checkpoint_path is not None and not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("All paths validated successfully")
    
    def _load_model(self):
        """Load SAM2 model and predictor"""
        try:
            # Change to sam2base directory for proper config resolution
            original_dir = os.getcwd()
            sam2base_dir = project_root / "libs" / "sam2base"
            os.chdir(str(sam2base_dir))
            
            # Config file based on model size
            config_file = f"configs/sam2/sam2_hiera_{self.model_size[0]}.yaml"
            
            logger.info(f"Loading {self.model_name} from config: {config_file}")
            
            # Load model
            if self.checkpoint_path is None:
                logger.info("Loading base model (pretrained)")
                self.model = build_sam2(config_file, ckpt_path=None, device="cpu")
            else:
                logger.info(f"Loading fine-tuned model from {self.checkpoint_path}")
                self.model = build_sam2(config_file, ckpt_path=self.checkpoint_path, device="cpu")
            
            # Move to GPU and create predictor
            self.model.to("cuda")
            self.model.eval()
            self.predictor = SAM2ImagePredictor(self.model)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        finally:
            os.chdir(original_dir)
    
    def _get_binary_gt_mask(self, ann_path, target_shape):
        """Load ground truth mask from Supervisely JSON format"""
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)

            H, W = target_shape
            full = np.zeros((H, W), dtype=np.uint8)

            for obj in data.get('objects', []):
                bmp = obj.get('bitmap', {})
                if 'data' not in bmp or 'origin' not in bmp:
                    continue

                # Decode compressed PNG
                raw = zlib.decompress(base64.b64decode(bmp['data']))
                patch = np.array(Image.open(io.BytesIO(raw)))

                # Use alpha channel if exists, otherwise first channel
                if patch.ndim == 3 and patch.shape[2] == 4:
                    patch = patch[:, :, 3]
                elif patch.ndim == 3:
                    patch = patch[:, :, 0]

                patch = (patch > 0).astype(np.uint8)

                ox, oy = map(int, bmp['origin'])
                ph, pw = patch.shape

                x1 = max(0, ox); y1 = max(0, oy)
                x2 = min(W, ox + pw); y2 = min(H, oy + ph)
                if x2 > x1 and y2 > y1:
                    full[y1:y2, x1:x2] = np.maximum(
                        full[y1:y2, x1:x2], patch[:(y2 - y1), :(x2 - x1)]
                    )

            return full.astype(bool)

        except Exception as e:
            logger.error(f"Error loading GT from {ann_path}: {e}")
            return None
    
    def _generate_auto_prompts(self, image, grid_spacing=128, box_scales=[128, 256]):
        """Generate automatic prompts (same as evaluation scripts)"""
        height, width = image.shape[:2]
        
        # Strategy 1: Grid-based point sampling
        grid_points = []
        grid_labels = []
        
        for y in range(grid_spacing, height - grid_spacing, grid_spacing):
            for x in range(grid_spacing, width - grid_spacing, grid_spacing):
                grid_points.append([x, y])
                grid_labels.append(1)  # All positive prompts
        
        # Strategy 2: Multi-scale box sweeping
        auto_boxes = []
        for scale in box_scales:
            stride = scale
            for y in range(0, height - scale, stride):
                for x in range(0, width - scale, stride):
                    if x + scale <= width and y + scale <= height:
                        auto_boxes.append([x, y, x + scale, y + scale])
        
        return {
            'points': np.array(grid_points) if grid_points else np.zeros((0, 2)),
            'point_labels': np.array(grid_labels) if grid_labels else np.zeros((0,)),
            'boxes': np.array(auto_boxes) if auto_boxes else np.zeros((0, 4))
        }
    
    def _compute_mask_iou(self, mask1, mask2):
        """Compute IoU between two binary masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-8)
    
    def _apply_nms(self, masks, scores, iou_threshold=0.65):
        """Apply Non-Maximum Suppression"""
        if len(masks) <= 1:
            return masks, scores
        
        sorted_indices = np.argsort(scores)[::-1]
        keep_indices = []
        
        for idx in sorted_indices:
            keep = True
            for kept_idx in keep_indices:
                iou = self._compute_mask_iou(masks[idx], masks[kept_idx])
                if iou > iou_threshold:
                    keep = False
                    break
            if keep:
                keep_indices.append(idx)
        
        return [masks[i] for i in keep_indices], [scores[i] for i in keep_indices]
    
    def _predict_with_auto_prompts(self, image, auto_prompts):
        """Predict using automatic prompts"""
        height, width = image.shape[:2]
        all_masks = []
        all_scores = []
        
        # Process grid points
        if len(auto_prompts['points']) > 0:
            try:
                points = auto_prompts['points']
                labels = auto_prompts['point_labels']
                
                # Process in small batches
                point_batch_size = 4
                for i in range(0, len(points), point_batch_size):
                    batch_points = points[i:i + point_batch_size]
                    batch_labels = labels[i:i + point_batch_size]
                    
                    point_masks, point_scores, _ = self.predictor.predict(
                        point_coords=batch_points,
                        point_labels=batch_labels,
                        multimask_output=True
                    )
                    
                    if len(batch_points) == 1:
                        for j, score in enumerate(point_scores):
                            if score > self.confidence_threshold:
                                mask = (point_masks[j].astype(np.float32) > 0.5).astype(bool)
                                all_masks.append(mask)
                                all_scores.append(score)
                    else:
                        best_idx = np.argmax(point_scores)
                        if point_scores[best_idx] > self.confidence_threshold:
                            mask = (point_masks[best_idx].astype(np.float32) > 0.5).astype(bool)
                            all_masks.append(mask)
                            all_scores.append(point_scores[best_idx])
            except Exception as e:
                logger.error(f"Error with point prompts: {e}")
        
        # Process boxes individually
        if len(auto_prompts['boxes']) > 0:
            try:
                boxes = auto_prompts['boxes']
                for box in boxes:
                    box_masks, box_scores, _ = self.predictor.predict(
                        box=box,
                        multimask_output=True
                    )
                    
                    for j, score in enumerate(box_scores):
                        if score > self.confidence_threshold:
                            mask = (box_masks[j].astype(np.float32) > 0.5).astype(bool)
                            all_masks.append(mask)
                            all_scores.append(score)
            except Exception as e:
                logger.error(f"Error with box prompts: {e}")
        
        if not all_masks:
            return np.zeros((height, width), dtype=bool), 0
        
        # Apply top-K filtering
        if len(all_masks) > 200:
            sorted_indices = np.argsort(all_scores)[::-1][:200]
            all_masks = [all_masks[i] for i in sorted_indices]
            all_scores = [all_scores[i] for i in sorted_indices]
        
        # Apply NMS
        filtered_masks, filtered_scores = self._apply_nms(all_masks, all_scores)
        
        if not filtered_masks:
            return np.zeros((height, width), dtype=bool), 0
        
        # Combine masks using logical OR
        combined_mask = np.zeros((height, width), dtype=bool)
        for mask in filtered_masks:
            binary_mask = mask.astype(bool)
            combined_mask = np.logical_or(combined_mask, binary_mask)
        
        return combined_mask, len(filtered_masks)
    
    def _create_visualization(self, image, gt_mask, pred_mask, img_name, iou, save_path):
        """Create professional visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_name} - Auto-prompt: {img_name}\nIoU: {iou:.3f}', 
                    fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ground Truth
        axes[0, 1].imshow(image)
        axes[0, 1].imshow(gt_mask, alpha=0.6, cmap='Greens')
        axes[0, 1].set_title('Ground Truth (Green)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Prediction
        color_map = 'Blues' if 'base' in self.model_name.lower() else 'Reds'
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(pred_mask, alpha=0.6, cmap=color_map)
        axes[1, 0].set_title(f'{self.model_name} Prediction\nIoU: {iou:.3f}', fontweight='bold')
        axes[1, 0].axis('off')
        
        # TP/FP/FN Analysis
        tp = np.logical_and(pred_mask, gt_mask)
        fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
        fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
        
        analysis_img = np.zeros_like(image)
        analysis_img[tp] = [0, 255, 0]    # Green for TP
        analysis_img[fp] = [255, 0, 0]    # Red for FP
        analysis_img[fn] = [255, 255, 0]  # Yellow for FN
        
        axes[1, 1].imshow(analysis_img)
        axes[1, 1].set_title('TP/FP/FN Analysis\nGreen=TP, Red=FP, Yellow=FN', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Add legend
        legend_elements = [
            patches.Patch(color='green', label='True Positive (TP)'),
            patches.Patch(color='red', label='False Positive (FP)'),
            patches.Patch(color='yellow', label='False Negative (FN)')
        ]
        axes[1, 1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _process_single_image(self, img_file):
        """Process a single image (for parallel processing)"""
        try:
            img_path = self.images_dir / img_file
            ann_path = self.annotations_dir / f"{img_file}.json"
            
            if not ann_path.exists():
                return None
            
            # Load image and GT
            image = cv2.imread(str(img_path))
            if image is None:
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            gt_mask = self._get_binary_gt_mask(str(ann_path), (height, width))
            if gt_mask is None:
                return None
            
            # Generate auto prompts
            auto_prompts = self._generate_auto_prompts(image)
            
            # Predict
            self.predictor.set_image(image)
            pred_mask, _ = self._predict_with_auto_prompts(image, auto_prompts)
            
            # Compute IoU
            iou = self._compute_mask_iou(pred_mask, gt_mask)
            
            # Create visualization
            img_name = Path(img_file).stem
            save_path = self.output_dir / f"{img_name}_{self.model_size}_{'base' if self.checkpoint_path is None else 'finetuned'}.png"
            
            self._create_visualization(image, gt_mask, pred_mask, img_name, iou, str(save_path))
            
            return {
                'image_name': img_name,
                'iou': iou,
                'visualization_path': str(save_path),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing {img_file}: {e}")
            return {
                'image_name': Path(img_file).stem,
                'iou': 0.0,
                'visualization_path': None,
                'success': False,
                'error': str(e)
            }
    
    def visualize(self, max_images=20, random_seed=42):
        """
        Main visualization method with parallel processing
        
        Args:
            max_images: Maximum number of images to process
            random_seed: Random seed for image selection
            
        Returns:
            dict: Summary statistics and results
        """
        logger.info(f"Starting visualization with {max_images} images, {self.num_workers} workers")
        
        # Validate paths
        self._validate_paths()
        
        # Load model
        self._load_model()
        
        # Get test images
        test_images = [f for f in self.images_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        
        if not test_images:
            raise ValueError(f"No images found in {self.images_dir}")
        
        # Select random subset
        random.seed(random_seed)
        selected_images = random.sample([f.name for f in test_images], 
                                      min(max_images, len(test_images)))
        
        logger.info(f"Selected {len(selected_images)} images for processing")
        
        # Process images in parallel using threads (PyTorch models can't be pickled for processes)
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self._process_single_image, img_file): img_file 
                for img_file in selected_images
            }
            
            # Collect results with progress bar
            with tqdm(total=len(selected_images), desc="Processing images") as pbar:
                for future in as_completed(future_to_image):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        processing_time = time.time() - start_time
        
        # Filter successful results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if failed_results:
            logger.warning(f"{len(failed_results)} images failed to process")
            for failed in failed_results:
                logger.warning(f"Failed: {failed['image_name']} - {failed.get('error', 'Unknown error')}")
        
        # Calculate statistics
        if successful_results:
            ious = [r['iou'] for r in successful_results]
            mean_iou = np.mean(ious)
            std_iou = np.std(ious)
            min_iou = np.min(ious)
            max_iou = np.max(ious)
        else:
            mean_iou = std_iou = min_iou = max_iou = 0.0
        
        # Create summary
        summary = {
            "model_info": {
                "name": self.model_name,
                "type": "base" if self.checkpoint_path is None else "finetuned",
                "size": self.model_size,
                "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
                "confidence_threshold": self.confidence_threshold
            },
            "processing_info": {
                "images_processed": len(successful_results),
                "images_failed": len(failed_results),
                "total_images_available": len(test_images),
                "selected_images": len(selected_images),
                "processing_time_seconds": processing_time,
                "images_per_second": len(successful_results) / processing_time if processing_time > 0 else 0,
                "workers_used": self.num_workers,
                "random_seed": random_seed
            },
            "performance_metrics": {
                "mean_iou": float(mean_iou),
                "std_iou": float(std_iou),
                "min_iou": float(min_iou),
                "max_iou": float(max_iou),
                "individual_ious": [float(r['iou']) for r in successful_results]
            },
            "paths": {
                "images_directory": str(self.images_dir),
                "annotations_directory": str(self.annotations_dir),
                "output_directory": str(self.output_dir)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = self.output_dir / "visualization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log results
        logger.info(f"Visualization completed in {processing_time:.2f} seconds")
        logger.info(f"Successfully processed: {len(successful_results)}/{len(selected_images)} images")
        logger.info(f"Mean IoU: {mean_iou:.3f} ± {std_iou:.3f}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return summary

def main():
    """Main function with professional argument parsing"""
    parser = argparse.ArgumentParser(
        description='Professional SAM2 Auto-prompt Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Base model visualization
  python mask_autoprompt_SAM2BasevsFT.py --images /path/to/images --annotations /path/to/ann
  
  # Fine-tuned model visualization
  python mask_autoprompt_SAM2BasevsFT.py --checkpoint /path/to/checkpoint.torch --images /path/to/images --annotations /path/to/ann
  
  # Custom configuration
  python mask_autoprompt_SAM2BasevsFT.py --checkpoint /path/to/checkpoint.torch --images /path/to/images --annotations /path/to/ann --max_images 50 --workers 8 --output_dir /path/to/output
        """
    )
    
    # Required arguments
    parser.add_argument('--images', type=str, required=True,
                       help='Path to test images directory (required)')
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to test annotations directory (required)')
    
    # Optional arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to SAM2 checkpoint (None for base model)')
    parser.add_argument('--model_size', type=str, choices=['small', 'large'], default=None,
                       help='SAM2 model size (auto-detected if not specified)')
    parser.add_argument('--max_images', type=int, default=20,
                       help='Maximum number of images to visualize (default: 20)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for image selection (default: 42)')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(log_level)
    
    try:
        # Generate output directory if not specified
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = "base" if args.checkpoint is None else "finetuned"
            args.output_dir = f"sam2_visualization_{model_type}_{timestamp}"
        
        # Create visualizer
        visualizer = SAM2Visualizer(
            checkpoint_path=args.checkpoint,
            model_size=args.model_size,
            images_dir=args.images,
            annotations_dir=args.annotations,
            output_dir=args.output_dir,
            num_workers=args.workers
        )
        
        # Run visualization
        summary = visualizer.visualize(
            max_images=args.max_images,
            random_seed=args.random_seed
        )
        
        # Print final summary
        print(f"\n{'='*80}")
        print("VISUALIZATION COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Model: {summary['model_info']['name']}")
        print(f"Images processed: {summary['processing_info']['images_processed']}")
        print(f"Mean IoU: {summary['performance_metrics']['mean_iou']:.3f} ± {summary['performance_metrics']['std_iou']:.3f}")
        print(f"Processing time: {summary['processing_info']['processing_time_seconds']:.2f} seconds")
        print(f"Speed: {summary['processing_info']['images_per_second']:.2f} images/second")
        print(f"Results saved to: {summary['paths']['output_directory']}")
        print(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()