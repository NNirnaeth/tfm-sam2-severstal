from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import os
import argparse
import glob
import json
import torch
import time
from tqdm import tqdm

def register_datasets(train_img, train_ann, val_img, val_ann, dataset_name="severstal"):
    """Register COCO datasets with given paths"""
    train_dataset = f"{dataset_name}_train"
    val_dataset = f"{dataset_name}_val"
    
    # Clear existing datasets if they exist
    if train_dataset in DatasetCatalog.list():
        DatasetCatalog.remove(train_dataset)
        MetadataCatalog.remove(train_dataset)
    if val_dataset in DatasetCatalog.list():
        DatasetCatalog.remove(val_dataset)
        MetadataCatalog.remove(val_dataset)
    
    # Register new datasets
    register_coco_instances(train_dataset, {}, train_ann, train_img)
    register_coco_instances(val_dataset, {}, val_ann, val_img)
    
    # Add metadata for visualization
    MetadataCatalog.get(train_dataset).set(thing_classes=["defect"])
    MetadataCatalog.get(val_dataset).set(thing_classes=["defect"])
    
    print(f"Registered datasets:")
    print(f"  Train: {train_dataset} -> {train_img}")
    print(f"  Val:   {val_dataset} -> {val_img}")
    
    return train_dataset, val_dataset

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    checkpoint_patterns = [
        os.path.join(output_dir, "checkpoint_*"),
        os.path.join(output_dir, "*_model_*.pth"),
        os.path.join(output_dir, "models", "*_model_*.pth"),
        os.path.join(output_dir, "**", "*.pth")
    ]
    
    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern, recursive=True))
    
    # Filter out non-checkpoint files
    checkpoint_files = []
    for checkpoint in all_checkpoints:
        filename = os.path.basename(checkpoint)
        if not any(skip in filename.lower() for skip in ['evaluation', 'prediction', 'result']):
            checkpoint_files.append(checkpoint)
    
    if not checkpoint_files:
        return None
    
    # Sort by iteration number
    def extract_iteration(checkpoint_path):
        filename = os.path.basename(checkpoint_path)
        try:
            if filename.startswith('checkpoint_'):
                return int(filename.split('_')[1])
            elif '_model_' in filename:
                parts = filename.split('_')
                for part in parts:
                    if part.isdigit() and len(part) >= 4:
                        return int(part)
                return 0
            else:
                return 0
        except (IndexError, ValueError):
            return 0
    
    latest_checkpoint = max(checkpoint_files, key=extract_iteration)
    return latest_checkpoint

def bbox_to_mask(bbox, height, width):
    """Convert bounding box to binary mask"""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Clamp coordinates to image boundaries
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    
    mask = np.zeros((height, width), dtype=np.uint8)
    if w > 0 and h > 0:
        mask[y:y+h, x:x+w] = 1
    
    return mask

def calculate_dice_metrics(model, dataset_dicts, device, max_samples=None):
    """Calculate DICE metrics by converting detections to masks"""
    print("Calculating DICE metrics...")
    
    if max_samples and len(dataset_dicts) > max_samples:
        dataset_dicts = random.sample(dataset_dicts, max_samples)
    
    dice_scores = []
    inference_times = []
    
    for idx, d in enumerate(tqdm(dataset_dicts, desc="Calculating DICE")):
        try:
            # Load image
            img = cv2.imread(d["file_name"])
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img_rgb.shape[:2]
            
            # Prepare input
            inputs = [{
                "image": torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device),
                "height": height,
                "width": width
            }]
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get predictions
            if isinstance(outputs, list) and len(outputs) > 0:
                predictions = outputs[0]
            else:
                predictions = outputs
            
            # Create ground truth mask from annotations
            gt_mask = np.zeros((height, width), dtype=np.uint8)
            for ann in d['annotations']:
                bbox = ann['bbox']  # [x, y, w, h]
                ann_mask = bbox_to_mask(bbox, height, width)
                gt_mask = np.maximum(gt_mask, ann_mask)
            
            # Create prediction mask from detections
            pred_mask = np.zeros((height, width), dtype=np.uint8)
            if 'instances' in predictions and len(predictions['instances']) > 0:
                for i in range(len(predictions['instances'])):
                    bbox = predictions['instances'].pred_boxes.tensor[i].cpu().numpy()  # [x1, y1, x2, y2]
                    # Convert to [x, y, w, h] format
                    x1, y1, x2, y2 = bbox
                    bbox_xywh = [x1, y1, x2-x1, y2-y1]
                    ann_mask = bbox_to_mask(bbox_xywh, height, width)
                    pred_mask = np.maximum(pred_mask, ann_mask)
            
            # Calculate DICE score
            intersection = np.sum(pred_mask * gt_mask)
            union = np.sum(pred_mask) + np.sum(gt_mask)
            dice = (2 * intersection) / (union + 1e-6)
            dice_scores.append(dice)
            
            # Debug: Print some examples
            if idx < 5:
                print(f"Image {idx}: GT_pixels={np.sum(gt_mask)}, Pred_pixels={np.sum(pred_mask)}, "
                      f"Intersection={intersection}, DICE={dice:.4f}")
            
        except Exception as e:
            print(f"Error calculating DICE for image {d['file_name']}: {e}")
            continue
    
    return {
        'mean_dice': np.mean(dice_scores) if dice_scores else 0.0,
        'std_dice': np.std(dice_scores) if dice_scores else 0.0,
        'avg_inference_time': np.mean(inference_times) if inference_times else 0.0,
        'total_inference_time': np.sum(inference_times) if inference_times else 0.0,
        'num_images': len(dataset_dicts)
    }

def visualize_dataset(dataset_name, num_samples=5, output_dir="./output_detectron2"):
    """Visualize dataset samples with bounding boxes"""
    print(f"Visualizing {num_samples} samples from {dataset_name}...")
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    # Randomly sample images
    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, d in enumerate(samples):
        # Load image
        img = cv2.imread(d["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualizer
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        
        # Draw annotations
        vis = visualizer.draw_dataset_dict(d)
        
        # Save visualization
        output_path = f"{output_dir}/visualization_{dataset_name}_{i}.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR))
        
        print(f"Saved: {output_path}")
        print(f"Image shape: {img.shape}")
        print(f"Number of annotations: {len(d['annotations'])}")
        
        # Print annotation details
        for j, ann in enumerate(d['annotations']):
            bbox = ann['bbox']  # [x, y, w, h]
            category_id = ann['category_id']
            print(f"  Annotation {j}: bbox={bbox}, category={category_id}")
        print("-" * 50)

class SimpleTrainer(DefaultTrainer):
    """Simple trainer without early stopping - just saves checkpoints periodically"""
    
    def __init__(self, cfg, save_iter=1000, resume_from=None):
        super().__init__(cfg)
        self.save_iter = save_iter
        self.resume_from = resume_from
        self.checkpointer = DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR)
        
        # Load training state if resuming
        if self.resume_from:
            self._load_training_state()
    
    def _load_training_state(self):
        """Load training state from checkpoint"""
        try:
            checkpoint_path = self.cfg.MODEL.WEIGHTS
            checkpoint = self.checkpointer.load(checkpoint_path)
            if checkpoint:
                print(f"Resuming from checkpoint: {checkpoint_path}")
                print(f"Resumed iteration: {checkpoint.get('iteration', 'unknown')}")
            else:
                print(f"Failed to load checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading training state: {e}")
    
    def save_checkpoint(self, checkpoint_path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_name = os.path.basename(checkpoint_path)
        self.checkpointer.save(checkpoint_name)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def run_step(self):
        # Call parent run_step
        super().run_step()
        
        # Save checkpoint every save_iter iterations
        if self.iter % self.save_iter == 0 and self.iter > 0:
            print(f"\n{'='*60}")
            print(f"SAVING CHECKPOINT AT ITERATION {self.iter}")
            print(f"{'='*60}")
            
            # Save checkpoint
            checkpoint_path = f"{self.cfg.OUTPUT_DIR}/checkpoint_{self.iter}"
            self.save_checkpoint(checkpoint_path)
            
            print(f"Checkpoint saved: {checkpoint_path}")
            print(f"{'='*60}\n")
        
        return True

class EarlyStoppingTrainer(DefaultTrainer):
    """Trainer with early stopping based on evaluation metrics"""
    
    def __init__(self, cfg, eval_interval=1000, patience=5, resume_from=None):
        super().__init__(cfg)
        self.eval_interval = eval_interval
        self.patience = patience
        self.best_metric = 0.0
        self.patience_counter = 0
        self.early_stop = False
        self.last_eval_iter = 0
        self.resume_from = resume_from
        self.checkpointer = DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR)
        
        # Load training state if resuming
        if self.resume_from:
            self._load_training_state()
    
    def _load_training_state(self):
        """Load training state from checkpoint"""
        try:
            checkpoint_path = self.cfg.MODEL.WEIGHTS
            checkpoint = self.checkpointer.load(checkpoint_path)
            if checkpoint:
                print(f"Resuming from checkpoint: {checkpoint_path}")
                print(f"Resumed iteration: {checkpoint.get('iteration', 'unknown')}")
                
                # Try to load additional training state from JSON file
                state_path = checkpoint_path + "_state.json"
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        training_state = json.load(f)
                    
                    self.best_metric = training_state.get('best_metric', 0.0)
                    self.patience_counter = training_state.get('patience_counter', 0)
                    self.last_eval_iter = training_state.get('last_eval_iter', 0)
                    
                    print(f"Resumed best metric: {self.best_metric:.4f}")
                    print(f"Resumed patience counter: {self.patience_counter}")
                    print(f"Resumed last eval iter: {self.last_eval_iter}")
                else:
                    print("No training state file found, using default values")
            else:
                print(f"Failed to load checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading training state: {e}")
    
    def save_checkpoint(self, checkpoint_path):
        """Save model checkpoint with training state"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_name = os.path.basename(checkpoint_path)
        self.checkpointer.save(checkpoint_name)
        
        # Save additional training state
        state_path = checkpoint_path + "_state.json"
        training_state = {
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'last_eval_iter': self.last_eval_iter,
            'iteration': self.iter
        }
        
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"Training state saved: {state_path}")
    
    def run_step(self):
        # Call parent run_step
        super().run_step()
        
        # Check for evaluation every eval_interval iterations
        if self.iter % self.eval_interval == 0 and self.iter > self.last_eval_iter:
            self.last_eval_iter = self.iter
            print(f"\n{'='*60}")
            print(f"EVALUATION AT ITERATION {self.iter}")
            print(f"{'='*60}")
            
            # Save checkpoint
            checkpoint_path = f"{self.cfg.OUTPUT_DIR}/checkpoint_{self.iter}"
            self.save_checkpoint(checkpoint_path)
            
            # Run evaluation using DICE metrics
            print("Running evaluation with DICE metrics...")
            device = next(self.model.parameters()).device
            dataset_dicts = DatasetCatalog.get(self.cfg.DATASETS.TEST[0])
            
            # Calculate DICE metrics
            dice_metrics = calculate_dice_metrics(self.model, dataset_dicts, device, max_samples=100)
            current_metric = dice_metrics['mean_dice']
            metric_name = "DICE"
            
            print(f"Current {metric_name}: {current_metric:.4f}")
            print(f"Best {metric_name}: {self.best_metric:.4f}")
            print(f"DICE std: {dice_metrics['std_dice']:.4f}")
            print(f"Evaluated on {dice_metrics['num_images']} images")
            
            # Check for improvement
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                print(f"✓ New best {metric_name}: {self.best_metric:.4f}")
                
                # Save best model
                best_model_path = f"{self.cfg.OUTPUT_DIR}/best_model"
                self.save_checkpoint(best_model_path)
                print(f"Best model saved: {best_model_path}")
            else:
                self.patience_counter += 1
                print(f"⚠ No improvement for {self.patience_counter}/{self.patience} evaluations")
                
                if self.patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered! No improvement for {self.patience} evaluations")
                    self.early_stop = True
                    return False  # Stop training
            
            print(f"{'='*60}\n")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Train Detectron2 model with configurable options')
    
    # Dataset configuration
    parser.add_argument('--train_img', type=str, 
                       default='datasets/severstal/splits/full/train2017/',
                       help='Path to training images directory')
    parser.add_argument('--train_ann', type=str,
                       default='datasets/severstal/splits/full/annotations/instances_train2017.json',
                       help='Path to training annotations JSON file')
    parser.add_argument('--val_img', type=str,
                       default='datasets/severstal/splits/full/val2017/',
                       help='Path to validation images directory')
    parser.add_argument('--val_ann', type=str,
                       default='datasets/severstal/splits/full/annotations/instances_val2017.json',
                       help='Path to validation annotations JSON file')
    parser.add_argument('--dataset_name', type=str, default='severstal',
                       help='Name for the dataset (used in registration)')
    
    # Model configuration
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to custom config file (if not provided, uses COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml)')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='Path to pretrained weights (if not provided, uses COCO pretrained)')
    
    # Training configuration
    parser.add_argument('--output_dir', type=str, default='./output_detectron2',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--max_iter', type=int, default=9000,
                       help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (IMS_PER_BATCH)')
    parser.add_argument('--base_lr', type=float, default=0.001,
                       help='Base learning rate')
    parser.add_argument('--lr_steps', type=int, nargs='+', default=[6000, 8000],
                       help='Learning rate decay steps')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                       help='Learning rate decay gamma')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loader workers')
    
    # Checkpoint configuration
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path to resume from')
    parser.add_argument('--save_iter', type=int, default=1000,
                       help='Save checkpoint every N iterations')
    
    # Input configuration
    parser.add_argument('--min_size_train', type=int, nargs='+', default=[256, 512, 768],
                       help='Minimum training image sizes (multi-scale)')
    parser.add_argument('--max_size_train', type=int, default=1600,
                       help='Maximum training image size')
    parser.add_argument('--min_size_test', type=int, default=256,
                       help='Minimum test image size')
    parser.add_argument('--max_size_test', type=int, default=1600,
                       help='Maximum test image size')
    
    # Evaluation configuration
    parser.add_argument('--score_thresh', type=float, default=0.5,
                       help='Score threshold for detections')
    parser.add_argument('--eval_samples', type=int, default=100,
                       help='Number of samples to use for evaluation during training')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize dataset samples before training')
    parser.add_argument('--vis_samples', type=int, default=3,
                       help='Number of samples to visualize')
    
    # Early stopping (disabled by default)
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping (disabled by default)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                       help='Evaluate every N iterations (for early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience for early stopping')
    
    # Device configuration
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda:0, cuda:1, etc.). If not specified, uses CUDA_VISIBLE_DEVICES or cuda:0')
    
    args = parser.parse_args()
    
    # Handle CUDA_VISIBLE_DEVICES and device selection
    print("=" * 60)
    print("DEVICE CONFIGURATION")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Training will use CPU (very slow!)")
        device = "cpu"
    else:
        # Get available devices
        available_devices = list(range(torch.cuda.device_count()))
        print(f"Available CUDA devices: {available_devices}")
        
        # Check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible:
            print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
            # Parse visible devices
            visible_devices = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
            print(f"Visible devices: {visible_devices}")
            
            # Use the first visible device
            if visible_devices:
                device_id = visible_devices[0]
                device = f"cuda:{device_id}"
                print(f"Using device: {device}")
            else:
                device = "cuda:0"
                print(f"Invalid CUDA_VISIBLE_DEVICES, using: {device}")
        else:
            # Use specified device or default
            if args.device:
                device = args.device
                print(f"Using specified device: {device}")
            else:
                device = "cuda:0"
                print(f"Using default device: {device}")
    
    # Verify device is available
    if device.startswith('cuda:'):
        device_id = int(device.split(':')[1])
        if device_id >= torch.cuda.device_count():
            print(f"Warning: Device {device} not available. Available devices: {available_devices}")
            device = f"cuda:{available_devices[0]}" if available_devices else "cpu"
            print(f"Falling back to: {device}")
    
    print(f"Final device: {device}")
    
    # Test device functionality
    if device.startswith('cuda:'):
        try:
            test_tensor = torch.randn(2, 3).to(device)
            print(f"✓ Device test successful: {test_tensor.device}")
        except Exception as e:
            print(f"✗ Device test failed: {e}")
            print("Falling back to CPU")
            device = "cpu"
    
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Register datasets
    print("=" * 60)
    print("REGISTERING DATASETS")
    print("=" * 60)
    train_dataset, val_dataset = register_datasets(
        args.train_img, args.train_ann, 
        args.val_img, args.val_ann, 
        args.dataset_name
    )
    
    # Visualize datasets if requested
    if args.visualize:
        print("\n" + "=" * 60)
        print("VISUALIZING DATASETS")
        print("=" * 60)
        visualize_dataset(train_dataset, args.vis_samples, args.output_dir)
        visualize_dataset(val_dataset, args.vis_samples, args.output_dir)
    
    # Load configuration
    print("\n" + "=" * 60)
    print("LOADING CONFIGURATION")
    print("=" * 60)
    cfg = get_cfg()
    
    if args.config_file:
        print(f"Loading custom config from: {args.config_file}")
        cfg.merge_from_file(args.config_file)
    else:
        print("Using default COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml config")
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Set pretrained weights
    if args.pretrained_weights:
        cfg.MODEL.WEIGHTS = args.pretrained_weights
        print(f"Using pretrained weights: {args.pretrained_weights}")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        print("Using COCO pretrained weights")
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)
    
    # DataLoader configuration
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    
    # Solver configuration
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = args.lr_steps
    cfg.SOLVER.GAMMA = args.lr_gamma
    
    # Model configuration
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 defect class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    
    # Input configuration
    cfg.INPUT.MIN_SIZE_TRAIN = tuple(args.min_size_train)
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size_train
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST = args.max_size_test
    
    # Output directory
    cfg.OUTPUT_DIR = args.output_dir
    
    # Set device for Detectron2
    if device.startswith('cuda:'):
        device_id = int(device.split(':')[1])
        torch.cuda.set_device(device_id)
        print(f"Set CUDA device to: {device_id}")
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device: {device}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Base LR: {args.base_lr}")
    print(f"  LR steps: {args.lr_steps}")
    print(f"  Save every: {args.save_iter} iterations")
    print(f"  Early stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
    
    # Determine checkpoint to resume from
    resume_checkpoint = None
    if args.resume or args.checkpoint:
        if args.checkpoint:
            resume_checkpoint = args.checkpoint
            print(f"Resuming from specified checkpoint: {resume_checkpoint}")
        else:
            resume_checkpoint = find_latest_checkpoint(args.output_dir)
            if resume_checkpoint:
                print(f"Resuming from latest checkpoint: {resume_checkpoint}")
            else:
                print("No checkpoint found for resume, starting from scratch")
    else:
        print("Starting training from scratch")
    
    # Set up checkpoint path for Detectron2 resume mechanism
    if resume_checkpoint:
        if not os.path.isabs(resume_checkpoint):
            resume_checkpoint = os.path.abspath(resume_checkpoint)
        
        if not os.path.exists(resume_checkpoint):
            print(f"Warning: Checkpoint file not found: {resume_checkpoint}")
            print("Starting training from scratch instead")
            resume_checkpoint = None
        else:
            cfg.MODEL.WEIGHTS = resume_checkpoint
            print(f"Set MODEL.WEIGHTS to: {cfg.MODEL.WEIGHTS}")
    
    # Create trainer
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    if args.early_stopping:
        # Use EarlyStoppingTrainer
        trainer = EarlyStoppingTrainer(
            cfg, 
            eval_interval=args.eval_interval, 
            patience=args.patience, 
            resume_from=resume_checkpoint
        )
    else:
        # Use simple trainer without early stopping
        trainer = SimpleTrainer(cfg, save_iter=args.save_iter, resume_from=resume_checkpoint)
    
    # Configure resume behavior
    if resume_checkpoint:
        trainer.resume_or_load(resume=True)
        print("Training resumed from checkpoint")
    else:
        trainer.resume_or_load(resume=False)
        print("Training started from scratch")
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save final checkpoint
        final_checkpoint = f"{args.output_dir}/checkpoint_interrupted_{trainer.iter}"
        trainer.save_checkpoint(final_checkpoint)
        print(f"Final checkpoint saved: {final_checkpoint}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    if hasattr(trainer, 'best_metric'):
        print(f"Best metric achieved: {trainer.best_metric:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
