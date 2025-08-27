#!/usr/bin/env python3
"""
Evaluate LoRA SAM2 Large model on Severstal test split (2000 images).

This script evaluates LoRA-adapted SAM2 models trained with train_lora_sam2_full_dataset.py
following the same evaluation pipeline and metrics as other models for TFM comparison.

Test data: datasets/Data/splits/test_split (2000 images)
Evaluation metrics: mIoU, IoU@50, IoU@75, IoU@90, IoU@95, Dice, Precision, Recall, F1
Results saved in CSV format under data/results/ for comparative analysis.

"""

import os
import sys
import json
import torch
import numpy as np
import time
import random
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import cv2
from PIL import Image
import base64
import zlib
import io
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Fix imports for sam2 module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "libs", "sam2base"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Add src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from utils import decode_bitmap_to_mask

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LoRALayer(torch.nn.Module):
    """LoRA layer implementation (same as training script)"""
    
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
    """Linear layer with LoRA adaptation (same as training script)"""
    
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
    """Apply LoRA adapters to SAM2 model (same as training script)"""
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

def load_lora_model(base_model_path, config_path, lora_checkpoint_path, device="cuda"):
    """Load SAM2 model with LoRA adapters"""
    print(f"Loading base SAM2 model: {base_model_path}")
    
    # Build base model using relative config name (Hydra expects this format)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    sam2_model = build_sam2(config_name, base_model_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Apply LoRA structure (this creates the LoRA layers)
    print("Applying LoRA structure...")
    sam2_model, lora_params = apply_lora_to_model(sam2_model)
    
    # Load LoRA checkpoint
    print(f"Loading LoRA checkpoint: {lora_checkpoint_path}")
    lora_state_dict = torch.load(lora_checkpoint_path, map_location=device)
    
    # Load LoRA parameters
    missing_keys, unexpected_keys = sam2_model.load_state_dict(lora_state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys (expected for base model): {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    print(f"LoRA model loaded successfully")
    return predictor

def get_binary_gt_mask(ann_path, target_shape):
    """Load and combine all defect objects into a single binary mask from PNG bitmap"""
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
            png_image = Image.open(io.BytesIO(decompressed_data))
            
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

def generate_point_prompts(binary_mask, num_points=10):
    """Generate point prompts from binary mask (same as training)"""
    # Erode mask to get interior points
    eroded = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    
    # Find coordinates of positive pixels
    coords = np.argwhere(eroded > 0)
    
    if len(coords) == 0:
        # Fallback to original mask if erosion removes everything
        coords = np.argwhere(binary_mask > 0)
    
    if len(coords) == 0:
        return np.zeros((0, 1, 2)), np.zeros((0, 1))
    
    # Shuffle and select up to num_points
    np.random.shuffle(coords)
    points = coords[:min(num_points, len(coords))]
    
    # Convert to SAM2 format: (y, x) -> (x, y) and add batch dimension
    points = points[:, [1, 0]]  # Swap x, y
    points = np.expand_dims(points, axis=1)  # Add batch dimension
    
    # Create labels (all positive)
    labels = np.ones((len(points), 1), dtype=np.int32)
    
    return points, labels

def load_image_and_mask(img_path, ann_path):
    """Load image and corresponding binary mask (same preprocessing as training)"""
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load annotation
    mask = get_binary_gt_mask(ann_path, (image.shape[0], image.shape[1]))
    
    # Apply same resizing as training
    r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
    image_resized = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
    mask_resized = cv2.resize(mask, (image_resized.shape[1], image_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return image_resized, mask_resized

def calculate_metrics(pred_mask, gt_mask):
    """Calculate all evaluation metrics"""
    # Ensure binary masks
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Intersection and Union
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    # IoU
    iou = intersection / (union + 1e-8)
    
    # Pixel-level metrics
    tp = intersection
    fp = np.sum(pred_binary) - intersection
    fn = np.sum(gt_binary) - intersection
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Dice Coefficient
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'dice': float(dice)
    }

def calculate_iou_at_thresholds(pred_prob, gt_mask, thresholds=[0.5, 0.75, 0.9, 0.95]):
    """Calculate IoU at different thresholds using probability map"""
    results = {}
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    for th in thresholds:
        # Apply threshold to probability map
        pred_thresholded = (pred_prob > th).astype(np.uint8)
        
        # Calculate IoU at this threshold
        intersection = np.sum(pred_thresholded * gt_binary)
        union = np.sum(pred_thresholded) + np.sum(gt_binary) - intersection
        iou_th = intersection / (union + 1e-8)
        
        results[f'iou_{int(th*100)}'] = float(iou_th)
    
    return results

def evaluate_lora_sam2(predictor, test_data, threshold=0.5, save_examples=False, output_dir=None):
    """Evaluate LoRA SAM2 model on test data"""
    predictor.model.eval()
    
    all_metrics = []
    all_iou_thresholds = []
    inference_times = []
    
    # For AUPRC calculation (sample to save memory)
    sample_size = min(200, len(test_data))
    sample_indices = random.sample(range(len(test_data)), sample_size)
    sample_preds = []
    sample_targets = []
    
    print(f"Evaluating LoRA SAM2 on {len(test_data)} test images...")
    
    with torch.no_grad():
        for idx, (img_path, ann_path) in enumerate(tqdm(test_data, desc="Evaluating")):
            try:
                # Load image and mask
                image, mask = load_image_and_mask(img_path, ann_path)
                
                # Generate point prompts from ground truth mask
                input_points, input_labels = generate_point_prompts(mask)
                
                if len(input_points) == 0:
                    # Skip images without valid prompts
                    continue
                
                # Measure inference time
                start_time = time.time()
                
                # Set image and predict
                predictor.set_image(image)
                
                # Prepare prompts (same as training)
                input_point_tensor = torch.tensor(input_points, dtype=torch.float32).cuda()
                input_label_tensor = torch.tensor(input_labels, dtype=torch.int64).cuda()
                
                mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                    input_point_tensor, input_label_tensor, box=None, mask_logits=None, normalize_coords=True
                )
                
                if unnorm_coords is None or labels is None:
                    continue
                
                # Get embeddings and predict
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )
                
                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [feat[-1].unsqueeze(0) for feat in predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Post-process masks
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                
                # Get probability map (use sigmoid)
                prd_prob = torch.sigmoid(prd_masks[:, 0])
                
                # Combine multi-instance predictions if needed
                if prd_prob.dim() == 3:
                    prd_prob = prd_prob.max(dim=0).values
                
                # Convert to numpy and resize to match ground truth
                pred_prob_np = prd_prob.cpu().numpy()
                if pred_prob_np.shape != mask.shape:
                    pred_prob_np = cv2.resize(pred_prob_np, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # Calculate metrics
                metrics = calculate_metrics(pred_prob_np, mask)
                iou_thresholds = calculate_iou_at_thresholds(pred_prob_np, mask)
                
                # Combine all metrics
                combined_metrics = {**metrics, **iou_thresholds}
                all_metrics.append(combined_metrics)
                all_iou_thresholds.append(iou_thresholds)
                
                # Store sample for AUPRC
                if idx in sample_indices:
                    sample_preds.append(pred_prob_np.flatten())
                    sample_targets.append(mask.flatten())
                
                # Save examples if requested
                if save_examples and output_dir and idx < 20:
                    save_prediction_example(pred_prob_np, mask, img_path, output_dir, idx)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Calculate AUPRC from sample
        if sample_preds and sample_targets:
            all_preds_flat = np.concatenate(sample_preds)
            all_targets_flat = np.concatenate(sample_targets)
            auprc = average_precision_score(all_targets_flat, all_preds_flat)
            avg_metrics['auprc'] = auprc
        else:
            avg_metrics['auprc'] = 0.0
        
        # Add performance metrics
        avg_metrics['avg_inference_time'] = np.mean(inference_times)
        avg_metrics['total_inference_time'] = np.sum(inference_times)
        
        return avg_metrics, all_metrics
    else:
        return None, []

def save_prediction_example(pred, target, img_path, output_dir, idx):
    """Save prediction example for visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to match prediction
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img_resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    axes[0].imshow(img_resized)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(target, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Save
    filename = f"lora_sam2_pred_{idx:04d}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def save_results_csv(results, model_info, output_dir="new_src/evaluation/evaluation_results/sam2_lora"):
    """Save evaluation results to CSV following TFM format"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"exp_lora_sam2_{model_info['lr']}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for CSV
    data = {
        'exp_id': f"exp_lora_sam2_{model_info['lr']}_{timestamp}",
        'model': 'SAM2-Large-LoRA',
        'subset_size': 'full_dataset',
        'variant': f"lora_r{model_info['rank']}_a{model_info['alpha']}_lr{model_info['lr']}",
        'prompt_type': 'points',
        'img_size': '1024xAuto',
        'batch_size': 1,
        'steps': model_info.get('steps', 'N/A'),
        'lr': model_info['lr'],
        'wd': model_info.get('weight_decay', 'N/A'),
        'seed': 42,
        'val_mIoU': 'N/A',
        'val_Dice': 'N/A',
        'test_mIoU': results['iou'],
        'test_Dice': results['dice'],
        'IoU@50': results.get('iou_50', 'N/A'),
        'IoU@75': results.get('iou_75', 'N/A'),
        'IoU@90': results.get('iou_90', 'N/A'),
        'IoU@95': results.get('iou_95', 'N/A'),
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1': results['f1'],
        'AUPRC': results.get('auprc', 'N/A'),
        'avg_inference_time': results.get('avg_inference_time', 'N/A'),
        'ckpt_path': model_info['checkpoint_path'],
        'timestamp': timestamp
    }
    
    # Create DataFrame and save
    df = pd.DataFrame([data])
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA SAM2 Large on Severstal test split')
    
    # Model configuration
    parser.add_argument('--lora_checkpoint', type=str, required=True,
                       help='Path to LoRA checkpoint file')
    parser.add_argument('--base_model', type=str, 
                       default='models/sam2_base_models/sam2_hiera_large.pt',
                       help='Path to base SAM2 model')
    parser.add_argument('--config', type=str,
                       default='configs/sam2/sam2_hiera_l.yaml',
                       help='Path to SAM2 config file')
    
    # Data paths
    parser.add_argument('--test_dir', type=str,
                       default='datasets/Data/splits/test_split',
                       help='Path to test data directory')
    
    # Evaluation settings
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary threshold for predictions')
    parser.add_argument('--save_examples', action='store_true',
                       help='Save prediction examples')
    parser.add_argument('--output_dir', type=str, default='new_src/evaluation/evaluation_results/sam2_lora',
                       help='Output directory for results')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("LoRA SAM2 Large Evaluation Script")
    print("=" * 50)
    print(f"LoRA checkpoint: {args.lora_checkpoint}")
    print(f"Base model: {args.base_model}")
    print(f"Config: {args.config}")
    print(f"Test data: {args.test_dir}")
    print(f"Threshold: {args.threshold}")
    print("-" * 50)
    
    # Get project root directory (go up from new_src/evaluation/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Convert relative paths to absolute paths from project root
    if not os.path.isabs(args.base_model):
        args.base_model = os.path.join(project_root, args.base_model)
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    if not os.path.isabs(args.test_dir):
        args.test_dir = os.path.join(project_root, args.test_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    # Check if files exist
    if not os.path.exists(args.lora_checkpoint):
        raise ValueError(f"LoRA checkpoint not found: {args.lora_checkpoint}")
    if not os.path.exists(args.base_model):
        raise ValueError(f"Base model not found: {args.base_model}")
    if not os.path.exists(args.config):
        raise ValueError(f"Config file not found: {args.config}")
    
    # Load test data
    print("Loading test data...")
    test_img_dir = os.path.join(args.test_dir, "img")
    test_ann_dir = os.path.join(args.test_dir, "ann")
    
    if not os.path.exists(test_img_dir) or not os.path.exists(test_ann_dir):
        raise ValueError(f"Test data structure not found. Expected: {test_img_dir} and {test_ann_dir}")
    
    # Get all image files
    image_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} test images")
    
    test_data = []
    for img_file in image_files:
        img_path = os.path.join(test_img_dir, img_file)
        ann_file = img_file + '.json'
        ann_path = os.path.join(test_ann_dir, ann_file)
        
        if os.path.exists(ann_path):
            test_data.append((img_path, ann_path))
        else:
            print(f"Warning: No annotation found for {img_file}")
    
    print(f"Successfully paired {len(test_data)} image-annotation pairs")
    
    # Limit number of images if specified
    if args.max_images and len(test_data) > args.max_images:
        print(f"Limiting evaluation to {args.max_images} images (randomly selected)")
        random.seed(args.seed)
        test_data = random.sample(test_data, args.max_images)
        print(f"Selected {len(test_data)} images for evaluation")
    
    if len(test_data) == 0:
        raise ValueError("No valid test data found")
    
    # Load LoRA model
    print("Loading LoRA SAM2 model...")
    predictor = load_lora_model(args.base_model, args.config, args.lora_checkpoint, device)
    
    # Extract model info from checkpoint path for CSV
    checkpoint_name = os.path.basename(args.lora_checkpoint)
    model_info = {
        'checkpoint_path': args.lora_checkpoint,
        'rank': 8,  # Default values, could be extracted from checkpoint
        'alpha': 16,
        'lr': 'unknown',  # Could be extracted from checkpoint directory name
        'steps': 'unknown',
        'weight_decay': 'unknown'
    }
    
    # Try to extract info from checkpoint path
    if 'r8' in checkpoint_name or 'r16' in checkpoint_name:
        if 'r8' in checkpoint_name:
            model_info['rank'] = 8
        elif 'r16' in checkpoint_name:
            model_info['rank'] = 16
    
    if 'lr' in checkpoint_name:
        # Extract learning rate from filename
        import re
        lr_match = re.search(r'lr([0-9e\-\.]+)', checkpoint_name)
        if lr_match:
            model_info['lr'] = lr_match.group(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate model
    print(f"\nEvaluating LoRA SAM2 Large...")
    results, detailed_results = evaluate_lora_sam2(
        predictor=predictor,
        test_data=test_data,
        threshold=args.threshold,
        save_examples=args.save_examples,
        output_dir=args.output_dir if args.save_examples else None
    )
    
    if results is None:
        print("Evaluation failed!")
        return
    
    # Print results
    print(f"\nEvaluation Results for LoRA SAM2 Large:")
    print("=" * 50)
    print(f"Basic Metrics (threshold={args.threshold}):")
    print(f"  mIoU: {results['iou']:.4f}")
    print(f"  mDice: {results['dice']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    print(f"\nIoU at Different Thresholds:")
    if 'iou_50' in results:
        print(f"  IoU@50: {results['iou_50']:.4f}")
    if 'iou_75' in results:
        print(f"  IoU@75: {results['iou_75']:.4f}")
    if 'iou_90' in results:
        print(f"  IoU@90: {results['iou_90']:.4f}")
    if 'iou_95' in results:
        print(f"  IoU@95: {results['iou_95']:.4f}")
    
    print(f"\nExtra Metrics:")
    if 'auprc' in results:
        print(f"  AUPRC: {results['auprc']:.4f}")
    
    print(f"\nPerformance:")
    if 'avg_inference_time' in results:
        print(f"  Average inference time: {results['avg_inference_time']:.4f}s")
    if 'total_inference_time' in results:
        print(f"  Total inference time: {results['total_inference_time']:.2f}s")
    
    # Save results to CSV
            csv_path = save_results_csv(results, model_info, "new_src/evaluation/evaluation_results/sam2_lora")
    
    # Save detailed results as JSON
    json_path = os.path.join(args.output_dir, f"detailed_results_lora_sam2.json")
    results_json = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_json[key] = value.tolist()
        else:
            results_json[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {csv_path}")
    print(f"Detailed results saved to: {json_path}")
    if args.save_examples:
        print(f"Prediction examples saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
