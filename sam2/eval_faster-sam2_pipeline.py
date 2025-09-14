#!/usr/bin/env python3
"""
Evaluate Detectron2 + SAM2 Pipeline on Segmentation Datasets

This script provides a unified evaluation pipeline that:
1. Uses Detectron2 for bounding box detection
2. Uses SAM2 for mask generation from bbox prompts
3. Evaluates final masks against ground truth
4. Supports multiple dataset formats (Supervisely bitmap, COCO)

Workflow:
1. Load image and ground truth annotation
2. Run Detectron2 inference → bounding boxes
3. Use bboxes as prompts for SAM2 → masks
4. Calculate segmentation metrics (IoU, Dice, etc.)
5. Save results and visualizations

Usage:
    python eval_d2_sam2_pipeline.py --d2_model path/to/d2.pth --sam2_model path/to/sam2.pth --img_dir path/to/images --ann_dir path/to/annotations
"""

import os
import sys
import json
import torch
import numpy as np
import time
import cv2
import random
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
import base64
import zlib
from PIL import Image
from io import BytesIO
from sklearn.metrics import average_precision_score
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import queue
warnings.filterwarnings('ignore')

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_test_loader

# SAM2 imports
sys.path.append('libs/sam2base')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# LoRA imports
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA support will be limited.")

# LoRA implementation (same as eval_lora_sam2_full_dataset.py)
class LoRALayer(torch.nn.Module):
    """LoRA layer implementation"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout)
        
        # Initialize B with zeros for stable training
        torch.nn.init.zeros_(self.lora_B)
        
        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        # Ensure LoRA matrices are on the same device as input
        device = x.device
        lora_A = self.lora_A.to(device)
        lora_B = self.lora_B.to(device)
        
        # Apply LoRA: output = B @ A @ x * scaling
        if x.dim() > 2:
            # For batched inputs, reshape to 2D for matrix multiplication
            batch_shape = x.shape[:-1]
            x_2d = x.reshape(-1, x.shape[-1])
            lora_output = self.dropout(lora_B @ (lora_A @ x_2d.T)).T
            lora_output = lora_output.reshape(*batch_shape, -1) * self.scaling
        else:
            # For 2D inputs, use direct matrix multiplication
            lora_output = self.dropout(lora_B @ (lora_A @ x.T)).T * self.scaling
        return lora_output

class LoRALinear(torch.nn.Module):
    """Linear layer with LoRA adaptation"""
    
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features, 
            linear_layer.out_features, 
            rank, alpha, dropout
        )
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original output + LoRA adaptation
        original_output = self.linear(x)
        lora_output = self.lora(x)
        return original_output + lora_output

def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.05):
    """Apply LoRA adapters to SAM2 model"""
    lora_params = []
    
    # Apply LoRA only to mask decoder attention projections
    if hasattr(model, 'sam_mask_decoder'):
        decoder = model.sam_mask_decoder
        for layer_name, layer in decoder.named_modules():
            # Only apply to attention layers with q_proj attribute
            if 'attn' in layer_name and hasattr(layer, 'q_proj'):
                try:
                    # Apply LoRA to Q, K, V, O projections
                    layer.q_proj = LoRALinear(layer.q_proj, rank, alpha, dropout)
                    layer.k_proj = LoRALinear(layer.k_proj, rank, alpha, dropout)
                    layer.v_proj = LoRALinear(layer.v_proj, rank, alpha, dropout)
                    layer.out_proj = LoRALinear(layer.out_proj, rank, alpha, dropout)
                    
                    # Collect LoRA parameters
                    lora_params.extend(list(layer.q_proj.lora.parameters()))
                    lora_params.extend(list(layer.k_proj.lora.parameters()))
                    lora_params.extend(list(layer.v_proj.lora.parameters()))
                    lora_params.extend(list(layer.out_proj.lora.parameters()))
                    
                except Exception as e:
                    print(f"Warning: Could not apply LoRA to {layer_name}: {e}")
                    continue
    
    return model, lora_params

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.segmentation_metrics import SegmentationMetrics


def set_seed(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable for better performance


def diagnose_checkpoint(checkpoint_path):
    """Diagnose checkpoint to determine if it's LoRA or full model"""
    print(f"Diagnosing checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        print(f"Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check for LoRA indicators
            lora_keys = [k for k in checkpoint.keys() if 'lora' in k.lower() or 'adapter' in k.lower()]
            if lora_keys:
                print(f"✅ LoRA checkpoint detected (keys: {lora_keys[:3]}...)")
                return 'lora'
            
            # Check for SAM2 structure
            sam2_keys = [k for k in checkpoint.keys() if any(prefix in k for prefix in ['sam2.', 'image_encoder.', 'prompt_encoder.', 'mask_decoder.'])]
            if sam2_keys:
                print(f"✅ Full SAM2 checkpoint detected (keys: {sam2_keys[:3]}...)")
                return 'full'
            
            # Check for standard checkpoint structure
            if 'model' in checkpoint or 'state_dict' in checkpoint:
                print(f"✅ Standard checkpoint structure detected")
                return 'standard'
        
        print(f"❓ Unknown checkpoint structure")
        return 'unknown'
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return 'error'


def convert_lora_to_full_checkpoint(base_model_path, lora_path, output_path, model_type="sam2_hiera_l"):
    """Convert LoRA checkpoint to full model checkpoint using direct approach"""
    print(f"Converting LoRA to full checkpoint...")
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {lora_path}")
    print(f"Output: {output_path}")
    
    try:
        print("Loading base model...")
        config_file = f"configs/sam2/{model_type}.yaml"
        base_model = build_sam2(config_file=config_file, ckpt_path=base_model_path, device='cpu')
        
        print("Applying LoRA structure...")
        sam2_model, _ = apply_lora_to_model(base_model, rank=8, alpha=16, dropout=0.05)
        
        print("Loading LoRA checkpoint...")
        lora_checkpoint = torch.load(lora_path, map_location='cpu', weights_only=True)
        
        print(f"LoRA checkpoint keys: {list(lora_checkpoint.keys())[:5]}...")
        
        # Apply LoRA weights
        missing_keys, unexpected_keys = sam2_model.load_state_dict(lora_checkpoint, strict=False)
        
        if missing_keys:
            print(f"Missing keys (expected): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        print("Merging LoRA with base model...")
        # The model now has LoRA integrated, save the full state dict
        torch.save(sam2_model.state_dict(), output_path)
        print(f"✅ Full checkpoint saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ LoRA conversion failed: {e}")
        raise e


def optimize_gpu_settings():
    """Optimize GPU settings for maximum performance"""
    if torch.cuda.is_available():
        # Enable memory efficient attention if available
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Set memory fraction to use most of GPU memory
        torch.cuda.set_per_process_memory_fraction(0.90)  # Slightly reduced to avoid OOM
        
        # Enable mixed precision
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        print(f"GPU optimization enabled. Available GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_image_optimized(img_path):
    """Optimized image loading with caching"""
    try:
        # Use PIL for faster loading, then convert to numpy
        with Image.open(img_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Convert to numpy array
            image = np.array(img)
        return image
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def load_annotation_optimized(ann_path, img_shape):
    """Optimized annotation loading"""
    try:
        return load_bitmap_annotation(ann_path, img_shape)
    except Exception as e:
        print(f"Error loading annotation {ann_path}: {e}")
        return None


def preload_batch_data(batch_paths, num_workers=2):
    """Optimized batch data loading with reduced workers"""
    def load_single_item(item):
        img_path, ann_path = item
        image = load_image_optimized(img_path)
        if image is not None:
            gt_mask = load_annotation_optimized(ann_path, image.shape[:2])
            if gt_mask is not None:
                return (image, gt_mask, img_path, ann_path)
        return None
    
    # Use fewer workers to reduce memory overhead
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_single_item, batch_paths))
    
    # Filter out None results
    return [item for item in results if item is not None]


class VisualizationSaver:
    """Async visualization saver to avoid blocking inference"""
    
    def __init__(self, max_workers=2):
        self.queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        """Worker thread that processes visualization queue"""
        while self.running:
            try:
                # Get task from queue with timeout
                task = self.queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Execute visualization save
                image, gt_mask, bboxes, pred_mask, vis_path, image_name = task
                save_visualization(image, gt_mask, bboxes, pred_mask, vis_path, image_name)
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving visualization: {e}")
                self.queue.task_done()
    
    def save_async(self, image, gt_mask, bboxes, pred_mask, vis_path, image_name):
        """Add visualization to save queue"""
        if self.running:
            self.queue.put((image, gt_mask, bboxes, pred_mask, vis_path, image_name))
    
    def shutdown(self):
        """Shutdown the visualization saver"""
        self.running = False
        self.queue.put(None)  # Signal shutdown
        self.worker_thread.join(timeout=2.0)  # Shorter timeout
        if self.worker_thread.is_alive():
            print("Warning: Visualization thread still running, forcing shutdown")
        self.executor.shutdown(wait=False)  # Don't wait for executor


def load_detectron2_model(model_ckpt_path, config_file=None, device='cuda'):
    """Load Detectron2 model from checkpoint"""
    print(f"Loading Detectron2 model from: {model_ckpt_path}")
    
    # Load configuration
    cfg = get_cfg()
    if config_file:
        cfg.merge_from_file(config_file)
    else:
        # Use default Faster R-CNN config
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Update configuration
    cfg.MODEL.WEIGHTS = model_ckpt_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 defect class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Build and load model
    model = build_model(cfg)
    checkpoint = torch.load(model_ckpt_path, map_location=device, weights_only=False)
    
    # Try different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Detectron2 model loaded successfully")
    return model, cfg


def load_sam2_model_with_lora(base_model_path, lora_path, model_type="sam2_hiera_l", device='cuda'):
    """Load SAM2 model with LoRA adaptation (using direct loading approach)"""
    print(f"Loading SAM2 base model from: {base_model_path}")
    print(f"Loading LoRA adapter from: {lora_path}")
    print(f"Model type: {model_type}")
    
    # Use the correct config file path
    config_file = f"configs/sam2/{model_type}.yaml"
    
    try:
        # Load base model
        base_model = build_sam2(config_file=config_file, ckpt_path=base_model_path, device=device)
        print("Base SAM2 model loaded successfully")
        
        # Apply LoRA structure manually (same as eval_lora_sam2_full_dataset.py)
        print("Applying LoRA structure...")
        sam2_model, _ = apply_lora_to_model(base_model, rank=8, alpha=16, dropout=0.05)
        
        # Load LoRA checkpoint directly (no PEFT needed)
        print(f"Loading LoRA checkpoint: {lora_path}")
        sd = torch.load(lora_path, map_location=device, weights_only=True)
        missing_keys, unexpected_keys = sam2_model.load_state_dict(sd, strict=False)
        
        if missing_keys:
            print(f"Missing keys (expected for base model): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        sam2_model.eval()
        print("LoRA adapter applied successfully")
        
        predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2 model with LoRA loaded successfully")
        return sam2_model, predictor
        
    except Exception as e:
        print(f"Error loading SAM2 with LoRA: {e}")
        raise e


def load_sam2_model(checkpoint_path, model_type="sam2_hiera_l", device='cuda', base_model_path=None, lora_path=None):
    """Load SAM2 model from checkpoint with automatic LoRA detection"""
    print(f"Loading SAM2 model from: {checkpoint_path}")
    print(f"Model type: {model_type}")
    
    # If LoRA paths are provided, use LoRA loading
    if lora_path and base_model_path:
        return load_sam2_model_with_lora(base_model_path, lora_path, model_type, device)
    
    # Diagnose checkpoint type
    checkpoint_type = diagnose_checkpoint(checkpoint_path)
    
    if checkpoint_type == 'lora':
        print("⚠️  LoRA checkpoint detected but no base model provided!")
        print("   Please provide --sam2_base_model and use --sam2_model for LoRA path")
        print("   Falling back to standard loading (may not work correctly)...")
    
    # Use the correct config file path
    config_file = f"configs/sam2/{model_type}.yaml"
    
    try:
        # First try the standard build_sam2 approach
        sam2 = build_sam2(config_file=config_file, ckpt_path=checkpoint_path, device=device)
        predictor = SAM2ImagePredictor(sam2)
        print(f"SAM2 model loaded successfully")
        return sam2, predictor
    except KeyError as e:
        if "'model'" in str(e):
            print("Checkpoint doesn't have 'model' key, trying to load fine-tuned checkpoint directly...")
            # Handle fine-tuned checkpoint that doesn't have 'model' key
            try:
                # Load the model without checkpoint first
                sam2 = build_sam2(config_file=config_file, ckpt_path=None, device=device)
                
                # Load the fine-tuned state dict directly
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                
                # If checkpoint is a dict with state_dict key, use that
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                # If checkpoint is the state dict itself
                elif isinstance(checkpoint, dict) and any(key.startswith(('sam2.', 'image_encoder.', 'prompt_encoder.', 'mask_decoder.')) for key in checkpoint.keys()):
                    state_dict = checkpoint
                else:
                    # Try to find the model weights in the checkpoint
                    state_dict = checkpoint
                
                # Load the state dict
                sam2.load_state_dict(state_dict, strict=False)
                sam2.eval()
                
                predictor = SAM2ImagePredictor(sam2)
                print(f"SAM2 fine-tuned model loaded successfully")
                return sam2, predictor
                
            except Exception as e2:
                print(f"Error loading fine-tuned checkpoint: {e2}")
                raise e2
        else:
            raise e


def load_bitmap_annotation(ann_path, target_shape):
    """Load bitmap annotation from Supervisely format JSON"""
    try:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        if 'objects' not in data or not data['objects']:
            return np.zeros(target_shape, dtype=np.uint8)
        
        # Get image dimensions
        height = data['size']['height']
        width = data['size']['width']
        
        # Create empty mask
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process all objects with proper bitmap positioning
        for obj in data['objects']:
            if 'bitmap' not in obj or 'data' not in obj['bitmap']:
                continue
            
            bmp = obj['bitmap']
            
            # Decompress bitmap data (PNG compressed)
            compressed_data = base64.b64decode(bmp['data'])
            decompressed_data = zlib.decompress(compressed_data)
            
            # Load PNG image from bytes
            png_image = Image.open(BytesIO(decompressed_data))
            
            # Use ALPHA channel if exists, otherwise convert to L (grayscale)
            if png_image.mode == 'RGBA':
                crop = np.array(png_image.split()[-1])  # Alpha channel
            else:
                crop = np.array(png_image.convert('L'))  # Grayscale
            
            # Convert to binary mask
            crop_bin = (crop > 0).astype(np.uint8)
            
            # Get origin coordinates from bitmap
            x0, y0 = bmp.get('origin', [0, 0])
            h, w = crop_bin.shape[:2]
            
            # Calculate end coordinates (clamped to image boundaries)
            x1, y1 = min(x0 + w, width), min(y0 + h, height)
            
            # Place the bitmap at the correct position
            if x1 > x0 and y1 > y0:
                full_mask[y0:y1, x0:x1] = np.maximum(
                    full_mask[y0:y1, x0:x1],
                    crop_bin[:y1-y0, :x1-x0]
                )
        
        # Resize to target shape if needed
        if full_mask.shape != target_shape:
            mask_pil = Image.fromarray(full_mask * 255)
            mask_pil = mask_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            full_mask = (np.array(mask_pil) > 0).astype(np.uint8)
        
        return full_mask
        
    except Exception as e:
        print(f"Error loading GT from {ann_path}: {e}")
        return np.zeros(target_shape, dtype=np.uint8)


def load_test_data(img_dir, ann_dir, max_samples=None):
    """Load test data from image and annotation directories"""
    print(f"Loading test data from: {img_dir}")
    
    test_data = []
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        
        # Try different annotation formats
        ann_path = None
        for ext in ['.json', '.jpg.json']:
            potential_ann = os.path.join(ann_dir, img_file + ext)
            if os.path.exists(potential_ann):
                ann_path = potential_ann
                break
        
        if ann_path and os.path.exists(ann_path):
            test_data.append((img_path, ann_path))
        else:
            print(f"Warning: No annotation found for {img_file}")
    
    print(f"Found {len(test_data)} valid image-annotation pairs")
    
    if max_samples and len(test_data) > max_samples:
        print(f"Limiting to {max_samples} samples")
        test_data = random.sample(test_data, max_samples)
    
    return test_data


def run_detectron2_inference_batch(model, images, device):
    """Run Detectron2 inference on batch of images"""
    inputs = []
    
    for image in images:
        height, width = image.shape[:2]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device, non_blocking=True)
        inputs.append({
            "image": image_tensor,
            "height": height,
            "width": width
        })
    
    # Run inference with mixed precision
    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model(inputs)
    
    # Extract predictions for each image
    all_bboxes = []
    for output in outputs:
        bboxes = []
        if 'instances' in output and len(output['instances']) > 0:
            instances = output['instances']
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            
            for box, score in zip(boxes, scores):
                if score > 0.5:
                    bboxes.append(box.tolist())
        all_bboxes.append(bboxes)
    
    return all_bboxes


def run_detectron2_inference(model, image, device):
    """Run Detectron2 inference on single image (for compatibility)"""
    return run_detectron2_inference_batch(model, [image], device)[0]


def run_sam2_inference(predictor, image, bboxes, prompt_type='bbox'):
    """Optimized SAM2 inference using bboxes as prompts"""
    if not bboxes:
        return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)
    
    # Set image once per image (not per bbox)
    predictor.set_image(image)
    
    # Process all bboxes for this image efficiently
    if prompt_type == 'bbox':
        # Convert all bboxes to numpy array for batch processing
        all_boxes = np.array(bboxes)
        
        # Process all bboxes at once (much faster than individual processing)
        masks_pred, scores_pred, _ = predictor.predict(
            box=all_boxes,
            multimask_output=False  # Disable multimask for speed
        )
        
        # Handle single vs multiple bboxes
        if len(bboxes) == 1:
            masks_pred = [masks_pred]
            scores_pred = [scores_pred]
        
        # Merge masks efficiently
        merged_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        merged_prob = np.zeros(image.shape[:2], dtype=np.float32)
        
        for mask, score in zip(masks_pred, scores_pred):
            merged_mask = np.maximum(merged_mask, mask.astype(np.uint8))
            merged_prob = np.maximum(merged_prob, mask.astype(np.float32) * score)
        
        return merged_mask, merged_prob
    
    else:
        # Fallback for other prompt types (less optimized)
        masks = []
        scores = []
        
        for bbox in bboxes:
            if prompt_type == 'center_point':
                x1, y1, x2, y2 = bbox
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                input_point = np.array([[center_x, center_y]])
                input_label = np.array([1])
                masks_pred, scores_pred, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )
            elif prompt_type == 'random_points':
                x1, y1, x2, y2 = bbox
                points = []
                for _ in range(3):
                    px = random.uniform(x1, x2)
                    py = random.uniform(y1, y2)
                    points.append([px, py])
                input_point = np.array(points)
                input_label = np.array([1] * 3)
                masks_pred, scores_pred, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
            
            masks.append(masks_pred)
            scores.append(scores_pred)
        
        # Merge masks
        if masks:
            merged_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            merged_prob = np.zeros(image.shape[:2], dtype=np.float32)
            
            for mask, score in zip(masks, scores):
                merged_mask = np.maximum(merged_mask, mask.astype(np.uint8))
                merged_prob = np.maximum(merged_prob, mask.astype(np.float32) * score)
            
            return merged_mask, merged_prob
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)


def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """Calculate segmentation metrics"""
    # Convert to binary
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    
    # Calculate metrics
    intersection = np.sum(pred_binary * gt_mask)
    union = np.sum(pred_binary) + np.sum(gt_mask) - intersection
    
    # IoU
    iou = intersection / (union + 1e-8)
    
    # Dice
    dice = (2 * intersection) / (np.sum(pred_binary) + np.sum(gt_mask) + 1e-8)
    
    # Precision, Recall, F1
    tp = intersection
    fp = np.sum(pred_binary) - intersection
    fn = np.sum(gt_mask) - intersection
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_visualization(image, gt_mask, bboxes, pred_mask, output_path, image_name):
    """Save visualization with original, GT, bboxes, and prediction"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(gt_mask, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Bounding boxes
    axes[1, 0].imshow(image)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='blue', facecolor='none')
        axes[1, 0].add_patch(rect)
        axes[1, 0].text(x1, y1-5, f'{i+1}', fontsize=8, color='blue',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    axes[1, 0].set_title(f'Detected Boxes ({len(bboxes)})')
    axes[1, 0].axis('off')
    
    # Prediction
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(pred_mask, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
    axes[1, 1].set_title('SAM2 Prediction')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_pipeline(d2_model, sam2_predictor, test_data, device, 
                     prompt_type='bbox', threshold=0.5, save_visualizations=False, 
                     output_dir=None, max_samples=None, batch_size=4, num_workers=4):
    """Evaluate the D2+SAM2 pipeline with batch processing"""
    print(f"Evaluating pipeline with {len(test_data)} images...")
    print(f"Prompt type: {prompt_type}")
    print(f"Threshold: {threshold}")
    print(f"Batch size: {batch_size}")
    
    all_metrics = []
    inference_times = []
    
    # Create output directory for visualizations
    vis_saver = None
    if save_visualizations and output_dir:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        vis_saver = VisualizationSaver(max_workers=2)
        print("Async visualization saver initialized")
    
    # Process in batches for better GPU utilization
    
    for batch_idx in tqdm(range(0, len(test_data), batch_size), desc="Processing batches"):
        batch_data = test_data[batch_idx:batch_idx + batch_size]
        
        # Pre-load batch data using multiprocessing
        batch_items = preload_batch_data(batch_data, num_workers)
        
        if not batch_items:
            continue
            
        # Extract images and metadata for batch processing
        batch_images = [item[0] for item in batch_items]
        batch_gt_masks = [item[1] for item in batch_items]
        batch_paths = [(item[2], item[3]) for item in batch_items]
        
        # Step 1: Batch Detectron2 inference
        batch_start_time = time.time()
        batch_bboxes = run_detectron2_inference_batch(d2_model, batch_images, device)
        
        # Step 2: Process each image with SAM2 (SAM2 doesn't support true batching)
        for idx, (image, gt_mask, bboxes, (img_path, ann_path)) in enumerate(zip(batch_images, batch_gt_masks, batch_bboxes, batch_paths)):
            try:
                # SAM2 inference
                pred_mask, pred_prob = run_sam2_inference(sam2_predictor, image, bboxes, prompt_type)
                
                # Calculate metrics
                metrics = calculate_metrics(pred_mask, gt_mask, threshold)
                all_metrics.append(metrics)
                # Removed all_predictions and all_targets to save memory (AUPRC not needed)
                
                # Save visualization asynchronously (only first 10 for speed)
                if save_visualizations and output_dir and (batch_idx + idx) < 10:
                    image_name = os.path.basename(img_path).split('.')[0]
                    vis_path = os.path.join(vis_dir, f"{image_name}_pipeline.jpg")
                    vis_saver.save_async(image, gt_mask, bboxes, pred_mask, vis_path, image_name)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Record batch time
        batch_time = time.time() - batch_start_time
        inference_times.append(batch_time / len(batch_items))
        
        # Clear GPU cache less frequently to reduce overhead
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # AUPRC removed for performance - only using IoU, Dice, Precision, Recall, F1
        
        avg_metrics['avg_inference_time'] = np.mean(inference_times)
        avg_metrics['total_inference_time'] = np.sum(inference_times)
        avg_metrics['num_images'] = len(all_metrics)
        
        # Shutdown visualization saver
        if vis_saver:
            print("Shutting down visualization saver...")
            vis_saver.shutdown()
        
        return avg_metrics, all_metrics
    else:
        # Shutdown visualization saver even if no metrics
        if vis_saver:
            vis_saver.shutdown()
        return None, []


def save_results(metrics, output_dir, model_name):
    """Save evaluation results to CSV"""
    if not metrics:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results row
    results_row = {
        'model': model_name,
        'test_mIoU': metrics['iou'],
        'test_Dice': metrics['dice'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        # 'AUPRC': metrics['auprc'],  # Removed for performance
        'avg_inference_time': metrics['avg_inference_time'],
        'total_inference_time': metrics['total_inference_time'],
        'num_images': metrics['num_images'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"results_{model_name}.csv")
    df = pd.DataFrame([results_row])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return csv_path


def print_results(metrics):
    """Print evaluation results"""
    if not metrics:
        print("No results to display")
        return
    
    print("\n" + "="*60)
    print("D2+SAM2 PIPELINE EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nSegmentation Metrics:")
    print(f"  mIoU:           {metrics['iou']:.4f}")
    print(f"  mDice:          {metrics['dice']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1:             {metrics['f1']:.4f}")
    # print(f"  AUPRC:          {metrics['auprc']:.4f}")  # Removed for performance
    
    print(f"\nPerformance:")
    print(f"  Number of images:   {metrics['num_images']}")
    print(f"  Avg inference time: {metrics['avg_inference_time']:.4f}s")
    print(f"  Total time:         {metrics['total_inference_time']:.2f}s")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Detectron2 + SAM2 Pipeline')
    
    # Model paths
    parser.add_argument('--d2_model', type=str, required=True,
                       help='Path to Detectron2 model checkpoint')
    parser.add_argument('--sam2_model', type=str, required=True,
                       help='Path to SAM2 model checkpoint (or LoRA adapter if using --sam2_base_model)')
    parser.add_argument('--sam2_base_model', type=str, default=None,
                       help='Path to SAM2 base model checkpoint (required for LoRA)')
    parser.add_argument('--convert_lora', action='store_true',
                       help='Convert LoRA to full checkpoint and exit')
    parser.add_argument('--output_checkpoint', type=str, default=None,
                       help='Output path for converted checkpoint (used with --convert_lora)')
    parser.add_argument('--d2_config', type=str, default=None,
                       help='Path to Detectron2 config file (optional)')
    parser.add_argument('--sam2_model_type', type=str, default='sam2_hiera_l',
                       choices=['sam2_hiera_l', 'sam2_hiera_b', 'sam2_hiera_t'],
                       help='SAM2 model type')
    
    # Data paths
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Path to test images directory')
    parser.add_argument('--ann_dir', type=str, required=True,
                       help='Path to test annotations directory')
    
    # Evaluation settings
    parser.add_argument('--prompt_type', type=str, default='bbox',
                       choices=['bbox', 'center_point', 'random_points'],
                       help='SAM2 prompt type')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary threshold for mask evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./d2_sam2_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default='d2_sam2_pipeline',
                       help='Model name for output files')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save prediction visualizations')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Handle LoRA conversion mode
    if args.convert_lora:
        if not args.sam2_base_model:
            print("Error: --sam2_base_model is required for LoRA conversion")
            return
        
        if not args.output_checkpoint:
            # Generate output path
            base_name = os.path.splitext(os.path.basename(args.sam2_model))[0]
            args.output_checkpoint = f"{base_name}_merged.torch"
        
        try:
            convert_lora_to_full_checkpoint(
                base_model_path=args.sam2_base_model,
                lora_path=args.sam2_model,
                output_path=args.output_checkpoint,
                model_type=args.sam2_model_type
            )
            print(f"\n✅ LoRA conversion completed!")
            print(f"   You can now use the merged checkpoint: {args.output_checkpoint}")
            return
        except Exception as e:
            print(f"❌ LoRA conversion failed: {e}")
            return
    
    # Set seed
    set_seed(args.seed)
    
    # Optimize GPU settings
    optimize_gpu_settings()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use optimized number of workers
    print(f"Using {args.num_workers} workers for data loading")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    d2_model, d2_cfg = load_detectron2_model(args.d2_model, args.d2_config, device)
    
    # Load SAM2 model with LoRA support
    if args.sam2_base_model:
        print("Using LoRA mode: Loading base model + LoRA adapter")
        sam2_model, sam2_predictor = load_sam2_model(
            checkpoint_path=args.sam2_model,  # This will be the LoRA path
            model_type=args.sam2_model_type,
            device=device,
            base_model_path=args.sam2_base_model,
            lora_path=args.sam2_model
        )
    else:
        print("Using standard mode: Loading full model checkpoint")
        sam2_model, sam2_predictor = load_sam2_model(
            checkpoint_path=args.sam2_model,
            model_type=args.sam2_model_type,
            device=device
        )
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(args.img_dir, args.ann_dir, args.max_samples)
    
    if not test_data:
        print("No test data found!")
        return
    
    # Evaluate pipeline
    print(f"\nEvaluating {args.model_name}...")
    metrics, per_image_metrics = evaluate_pipeline(
        d2_model=d2_model,
        sam2_predictor=sam2_predictor,
        test_data=test_data,
        device=device,
        prompt_type=args.prompt_type,
        threshold=args.threshold,
        save_visualizations=args.save_visualizations,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Print and save results
    print_results(metrics)
    if metrics:
        csv_path = save_results(metrics, args.output_dir, args.model_name)
        
        # Save detailed results as JSON
        json_path = os.path.join(args.output_dir, f"detailed_results_{args.model_name}.json")
        results_json = {
            'summary_metrics': metrics,
            'per_image_metrics': per_image_metrics,
            'config': vars(args)
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {csv_path}")
        print(f"Detailed results saved to: {json_path}")
        
        if args.save_visualizations:
            vis_dir = os.path.join(args.output_dir, "visualizations")
            print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
