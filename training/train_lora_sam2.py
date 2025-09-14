#!/usr/bin/env python3
"""
Train SAM2 Large model with LoRA adapters on Severstal dataset.

This script implements LoRA (Low-Rank Adaptation) for efficient fine-tuning of SAM2:
- Applies LoRA adapters only to attention projections (Q, K, V, O) in image encoder and mask decoder
- Freezes all base model weights except LoRA adapters
- Uses recommended LoRA hyperparameters: r=8, alpha=16, dropout=0.05
- Implements proper training pipeline with validation, checkpointing, and metrics logging

Supports different dataset sizes:
- Full dataset: datasets/Data/splits/train_split (full training set)
- Subsets: datasets/Data/splits/subsets/{500,1000,2000}_subset
- Validation data: datasets/Data/splits/val_split (internal validation)
- Test data: datasets/Data/splits/test_split (final evaluation)

"""

import os
import sys
import numpy as np
import torch
import argparse
import json
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# Fix imports for sam2 module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "libs", "sam2base"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Add src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from utils import read_batch
from metrics_logger import MetricsLogger

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
np.random.seed(42)


class LoRALayer(torch.nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer implementation.
    
    This layer adds low-rank adaptation to existing linear layers by:
    - Adding two low-rank matrices A and B
    - Computing: output = original_output + (B @ A) @ input * (alpha / r)
    """
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices - ensure dimensions match the layer
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout)
        
        # Initialize B with zeros for stable training
        torch.nn.init.zeros_(self.lora_B)
        
        # Store dimensions for debugging
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        # Ensure LoRA matrices are on the same device as input
        device = x.device
        lora_A = self.lora_A.to(device)
        lora_B = self.lora_B.to(device)
        
        # Apply LoRA: output = B @ A @ x * scaling
        # Handle batched inputs properly
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
    """
    Linear layer with LoRA adaptation.
    
    This module wraps an existing linear layer and adds LoRA adaptation
    while keeping the original weights frozen.
    """
    
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
    """
    Apply LoRA adapters to SAM2 model.
    
    Args:
        model: SAM2 model
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout rate
    
    Returns:
        model: Model with LoRA adapters applied
        lora_params: List of LoRA parameters for training
    """
    lora_params = []
    
    # First, let's explore all layers to understand the model structure
    print("Exploring SAM2 model structure...")
    if hasattr(model, 'sam_mask_decoder'):
        decoder = model.sam_mask_decoder
        print(f"Mask decoder has {len(list(decoder.named_modules()))} modules")
        
        # Print all modules and their attributes
        for layer_name, layer in decoder.named_modules():
            if hasattr(layer, 'in_features') or hasattr(layer, 'out_features'):
                in_feat = getattr(layer, 'in_features', 'N/A')
                out_feat = getattr(layer, 'out_features', 'N/A')
                print(f"  {layer_name}: in_features={in_feat}, out_features={out_feat}")
            elif hasattr(layer, 'q_proj'):
                in_feat = layer.q_proj.in_features
                out_feat = layer.q_proj.out_features
                print(f"  {layer_name}: q_proj in_features={in_feat}, out_features={out_feat}")
    
    # Apply LoRA only to mask decoder attention projections (safer approach)
    if hasattr(model, 'sam_mask_decoder'):
        decoder = model.sam_mask_decoder
        for layer_name, layer in decoder.named_modules():
            # Only apply to attention layers with q_proj attribute
            if 'attn' in layer_name and hasattr(layer, 'q_proj'):
                try:
                    # Debug: print layer dimensions
                    print(f"Layer {layer_name}: in_features={layer.q_proj.in_features}, out_features={layer.q_proj.out_features}")
                    
                    # Validate dimensions before applying LoRA
                    if layer.q_proj.in_features <= 0 or layer.q_proj.out_features <= 0:
                        print(f"Warning: Invalid dimensions for {layer_name}, skipping")
                        continue
                    
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
                    
                    print(f"Applied LoRA to {layer_name}")
                except Exception as e:
                    print(f"Warning: Could not apply LoRA to {layer_name}: {e}")
                    continue
    
    # Skip image encoder and MLP layers for now to avoid dimension mismatches
    print(f"Applied LoRA to {len(lora_params)//4} attention layers")
    
    return model, lora_params


def prepare_dataset(dataset_size="full"):
    """
    Prepare dataset based on specified size.
    
    Args:
        dataset_size: "full", "500", "1000", or "2000"
    
    Returns:
        train_data, val_data: Lists of image-annotation pairs
    """
    def load_pairs(directory):
        """Load image-annotation pairs from a directory"""
        img_dir = os.path.join(directory, "img")
        ann_dir = os.path.join(directory, "ann")
        
        if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
            print(f"Error: Directory not found: {directory}")
            return []
        
        paired = []
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        for img_file in img_files:
            ann_file = img_file + '.json'
            ann_path = os.path.join(ann_dir, ann_file)
            img_path = os.path.join(img_dir, img_file)
            
            if os.path.exists(ann_path):
                paired.append({"image": img_path, "annotation": ann_path})
        
        return paired
    
    # Get project root for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Determine training data path based on dataset size
    if dataset_size == "full":
        train_dir = os.path.join(project_root, "datasets/Data/splits/train_split")
    else:
        train_dir = os.path.join(project_root, f"datasets/Data/splits/subsets/{dataset_size}_subset")
    
    # Validation data is always the same
    val_dir = os.path.join(project_root, "datasets/Data/splits/val_split")
    
    # Load training and validation data
    train_data = load_pairs(train_dir)
    val_data = load_pairs(val_dir)
    
    print(f"Dataset loaded:")
    print(f"  Training images: {len(train_data)} (from {train_dir})")
    print(f"  Validation images: {len(val_data)} (from {val_dir})")
    
    return train_data, val_data


def evaluate_on_validation_set(predictor, val_data, annotation_format="bitmap"):
    """
    Evaluate the model on the internal validation set.
    Returns mean IoU and Dice coefficient across all validation images.
    """
    predictor.model.eval()
    
    all_ious = []
    all_dices = []
    
    with torch.no_grad():
        for entry in val_data:
            try:
                # Load image and annotation
                image, mask, input_point, num_masks = read_batch(
                    [entry], annotation_format=annotation_format, visualize_data=False
                )
                
                if image is None or mask is None or num_masks == 0:
                    continue
                
                # Generate point prompts from ground truth mask
                input_label = np.ones((input_point.shape[0], 1), dtype=np.int32)
                if input_point.size == 0 or input_label.size == 0:
                    continue
                
                # Set image and predict
                predictor.set_image(image)
                mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                    input_point, input_label, box=None, mask_logits=None, normalize_coords=True
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
                
                # Post-process masks
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                
                # Calculate metrics
                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                prd_prob = torch.sigmoid(prd_masks[:, 0])
                
                # Combine multi-instance predictions
                if prd_prob.dim() == 3:
                    prd_prob = prd_prob.max(dim=0).values
                
                # Binary prediction
                pred_binary = (prd_prob > 0.5).float()
                
                # Ensure masks have same shape
                if gt_mask.dim() > 2:
                    gt_mask = gt_mask.squeeze(0)
                if pred_binary.dim() > 2:
                    pred_binary = pred_binary.squeeze(0)
                
                # Resize prediction to match ground truth (safer approach)
                if gt_mask.shape != pred_binary.shape:
                    pred_binary = torch.nn.functional.interpolate(
                        pred_binary.unsqueeze(0).unsqueeze(0), 
                        size=gt_mask.shape, 
                        mode='nearest'
                    ).squeeze()
                
                # Calculate IoU and Dice
                gt_flat = gt_mask.flatten()
                pred_flat = pred_binary.flatten()
                
                inter = (gt_flat * pred_flat).sum()
                union = gt_flat.sum() + pred_flat.sum() - inter
                iou = inter / (union + 1e-6)
                iou = torch.clamp(iou, 0.0, 1.0)
                
                dice = (2.0 * inter) / (gt_flat.sum() + pred_flat.sum() + 1e-6)
                dice = torch.clamp(dice, 0.0, 1.0)
                
                all_ious.append(iou.item())
                all_dices.append(dice.item())
                
            except Exception as e:
                print(f"Error processing validation sample: {e}")
                continue
    
    predictor.model.train()
    
    if len(all_ious) > 0:
        mean_iou = np.mean(all_ious)
        mean_dice = np.mean(all_dices)
        return mean_iou, mean_dice
    else:
        return 0.0, 0.0


def train_sam2_lora(learning_rate=1e-3, weight_decay=0.01, steps=10000, 
                   batch_size=1, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
                   dataset_size="full"):
    """
    Train SAM2 Large model with LoRA adapters on specified dataset.
    
    Args:
        learning_rate: Learning rate for LoRA training (higher than traditional FT)
        weight_decay: Weight decay for optimizer
        steps: Number of training steps
        batch_size: Batch size (effective batch size with accumulation)
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA scaling factor (alpha)
        lora_dropout: LoRA dropout rate
        dataset_size: "full", "500", "1000", or "2000"
    """
    
    print(f"\n{'='*80}")
    print(f"Training SAM2 LARGE with LoRA on {dataset_size} dataset")
    print(f"{'='*80}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"LoRA Alpha: {lora_alpha}")
    print(f"LoRA Dropout: {lora_dropout}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Training Steps: {steps}")
    print(f"Effective Batch Size: {batch_size * 4} (batch_size={batch_size}, accumulation=4)")
    
    # Get project root for output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Configuration - use absolute paths from project root
    base_model = os.path.join(project_root, "models/base/sam2/sam2_hiera_large.pt")
    config = "sam2_hiera_l"  # SAM2 expects just the config name, not the full path
    
    # Check if base model exists
    if not os.path.exists(base_model):
        print(f"Error: Base model not found: {base_model}")
        return None
    
    # Prepare dataset
    train_data, val_data = prepare_dataset(dataset_size)
    
    if len(train_data) == 0:
        print(f"Error: No training data found")
        return None
    
    # Build model
    print(f"Building SAM2 large model...")
    sam2_model = build_sam2(config, base_model, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Apply LoRA adapters
    print(f"Applying LoRA adapters (rank={lora_rank}, alpha={lora_alpha})...")
    sam2_model, lora_params = apply_lora_to_model(
        sam2_model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
    )
    
    print(f"LoRA parameters: {len(lora_params)}")
    print(f"Total trainable parameters: {sum(p.numel() for p in lora_params if p.requires_grad)}")
    
    # Set training mode
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    
    # Optimizer (only for LoRA parameters)
    optimizer = torch.optim.AdamW(
        lora_params, 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Cosine scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=1000,
        T_mult=2,
        eta_min=learning_rate * 0.01
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training configuration
    accumulation_steps = 4
    best_val_iou = 0.0
    mean_iou = 0.0
    
    # Early stopping configuration
    early_stopping_patience = 7  # Higher patience for LoRA training
    early_stopping_counter = 0
    early_stopping_min_delta = 1e-4
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"sam2_large_lora_r{lora_rank}_a{lora_alpha}_lr{str(learning_rate).replace('e-', 'e')}_{dataset_size}_{timestamp}"
    ckpt_dir = os.path.join(project_root, f"new_src/training/training_results/sam2_lora/{model_name}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Metrics logger
    logger = MetricsLogger(os.path.join(project_root, f"logs/training_metrics_lora_{dataset_size}_{timestamp}.csv"))
    
    # Training loop
    print(f"\nStarting LoRA training...")
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    
    for step in range(1, steps + 1):
        try:
            with torch.amp.autocast('cuda'):
                # Read batch
                image, mask, input_point, num_masks = read_batch(
                    train_data, annotation_format="bitmap", visualize_data=False
                )
                
                if image is None or mask is None or num_masks == 0:
                    continue
                
                # Prepare labels and convert to tensors on CUDA
                input_label = np.ones((input_point.shape[0], 1), dtype=np.int32)
                if input_point.size == 0 or input_label.size == 0:
                    continue
                
                # Convert numpy arrays to tensors and move to CUDA
                input_point_tensor = torch.tensor(input_point, dtype=torch.float32).cuda()
                input_label_tensor = torch.tensor(input_label, dtype=torch.int64).cuda()  # SAM2 expects Long
                
                # Set image and prepare prompts
                predictor.set_image(image)
                mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                    input_point_tensor, input_label_tensor, box=None, mask_logits=None, normalize_coords=True
                )
                
                if unnorm_coords is None or labels is None:
                    continue
                
                # Get embeddings
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )
                
                # Predict masks
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
                
                # Post-process masks
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                
                # Calculate loss
                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                prd_prob = torch.sigmoid(low_res_masks[:, 0])
                
                # logits [B, 3, 256, 256] -> coge la primera máscara por prompt
                prd_logits = low_res_masks[:, 0]                   # [B, 256, 256]

                # gt a [B, 256, 256]
                gt_mask = torch.tensor(mask.astype(np.float32), device=prd_logits.device)  # [H, W] o [1,H,W]
                if gt_mask.dim() == 2:
                    gt_mask = gt_mask.unsqueeze(0)                 # [1, H, W]
                gt_mask = torch.nn.functional.interpolate(
                    gt_mask.unsqueeze(1),                          # [1,1,H,W]
                    size=prd_logits.shape[-2:], mode='nearest'
                ).squeeze(1)                                       # [1, 256, 256]
                if gt_mask.shape[0] != prd_logits.shape[0]:
                    gt_mask = gt_mask.repeat(prd_logits.shape[0], 1, 1)  # [B, 256, 256]

                # BCE con logits (no aplicar sigmoid antes)
                seg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    prd_logits, gt_mask, reduction='mean'
                )
                
                # IoU por muestra (usar sigmoid aquí)
                prd_prob = torch.sigmoid(prd_logits)               # [B, 256, 256]
                pred_bin = (prd_prob > 0.5).float()                # [B, 256, 256]

                gt_flat   = gt_mask.view(gt_mask.size(0), -1)      # [B, 65536]
                pred_flat = pred_bin.view(pred_bin.size(0), -1)    # [B, 65536]
                inter = (gt_flat * pred_flat).sum(dim=1)
                union = gt_flat.sum(dim=1) + pred_flat.sum(dim=1) - inter
                iou   = (inter / (union + 1e-6)).clamp(0, 1)
                
                # Score loss
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                
                # Combined loss
                loss = seg_loss + score_loss * 0.05
                loss = loss / accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                
                # Optimizer step
                if step % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                # Update running IoU
                batch_mean_iou = np.mean(iou.cpu().detach().numpy())
                mean_iou = mean_iou * 0.99 + 0.01 * batch_mean_iou
                
                # Log progress
                if step % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Step {step:6d}/{steps}: IoU={mean_iou:.4f}, Loss={loss.item():.4f}, LR={current_lr:.2e}")
                
                # Validation and checkpointing every 500 steps
                if step % 500 == 0:
                    print(f"\nRunning validation at step {step}...")
                    val_iou, val_dice = evaluate_on_validation_set(predictor, val_data, "bitmap")
                    print(f"Validation IoU: {val_iou:.4f}, Dice: {val_dice:.4f} (Best IoU: {best_val_iou:.4f})")
                    
                    # Save best model
                    if val_iou > best_val_iou:
                        best_val_iou = val_iou
                        
                        # Save LoRA adapters only
                        lora_state_dict = {}
                        for name, param in predictor.model.named_parameters():
                            if 'lora' in name:
                                lora_state_dict[name] = param.data.clone()
                        
                        best_ckpt_path = os.path.join(ckpt_dir, f"best_lora_step{step}_iou{val_iou:.4f}_dice{val_dice:.4f}.torch")
                        torch.save(lora_state_dict, best_ckpt_path)
                        print(f"  New best LoRA adapters saved: {best_ckpt_path}")
                        
                        # Save merged model for inference
                        merged_ckpt_path = os.path.join(ckpt_dir, f"best_merged_step{step}_iou{val_iou:.4f}_dice{val_dice:.4f}.torch")
                        torch.save(predictor.model.state_dict(), merged_ckpt_path)
                        print(f"  New best merged model saved: {merged_ckpt_path}")
                        
                        # Reset early stopping counter
                        early_stopping_counter = 0
                        
                        # Remove previous best checkpoints (keep only top 3)
                        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("best_")]
                        if len(ckpt_files) > 6:  # 3 LoRA + 3 merged
                            ckpt_files.sort(key=lambda x: float(x.split("iou")[1].split("_")[0]), reverse=True)
                            for old_ckpt in ckpt_files[6:]:
                                os.remove(os.path.join(ckpt_dir, old_ckpt))
                                print(f"  Removed old checkpoint: {old_ckpt}")
                    else:
                        # Check if improvement is below threshold
                        if val_iou <= best_val_iou - early_stopping_min_delta:
                            early_stopping_counter += 1
                            print(f"  Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                        else:
                            # Small improvement, reset counter
                            early_stopping_counter = 0
                    
                    # Check early stopping condition
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} validations.")
                        print(f"Best validation IoU: {best_val_iou:.4f} at step {step}")
                        break
                    
                    # Save latest checkpoint
                    latest_lora_path = os.path.join(ckpt_dir, f"latest_lora_step{step}.torch")
                    lora_state_dict = {}
                    for name, param in predictor.model.named_parameters():
                        if 'lora' in name:
                            lora_state_dict[name] = param.data.clone()
                    torch.save(lora_state_dict, latest_lora_path)
                    
                    # Log metrics
                    logger.log(
                        mode="train",
                        dataset=f"severstal_{dataset_size}",
                        steps=step,
                        model_name=model_name,
                        iou=mean_iou,
                        iou50=None,
                        iou75=None,
                        iou95=None
                    )
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final model
    final_lora_path = os.path.join(ckpt_dir, f"final_lora_step{steps}.torch")
    lora_state_dict = {}
    for name, param in predictor.model.named_parameters():
        if 'lora' in name:
            lora_state_dict[name] = param.data.clone()
    torch.save(lora_state_dict, final_lora_path)
    
    # Save training summary
    summary = {
        "model_size": "large",
        "tuning_method": "lora",
        "dataset_size": dataset_size,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "total_steps": steps,
        "actual_steps": step,
        "best_val_iou": best_val_iou,
        "final_train_iou": mean_iou,
        "early_stopping_triggered": early_stopping_counter >= early_stopping_patience,
        "early_stopping_patience": early_stopping_patience,
        "timestamp": timestamp,
        "checkpoint_dir": ckpt_dir,
        "note": f"LoRA training with attention projections only on {dataset_size} dataset. Early stopping with patience=7."
    }
    
    summary_path = os.path.join(ckpt_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nLoRA training completed!")
    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping triggered at step {step}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Final training IoU: {mean_iou:.4f}")
    print(f"LoRA adapters saved to: {ckpt_dir}")
    print(f"Training summary: {summary_path}")
    
    return ckpt_dir


def main():
    parser = argparse.ArgumentParser(description="Train SAM2 Large with LoRA on Severstal dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate for LoRA training")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for optimizer")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (effective batch = batch_size * 4)")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA scaling factor (alpha)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout rate")
    parser.add_argument("--dataset_size", type=str, 
                       choices=["full", "500", "1000", "2000"],
                       default="full",
                       help="Dataset size: full, 500, 1000, or 2000 images")
    
    args = parser.parse_args()
    
    print("SAM2 Large LoRA Training Script")
    print("=" * 50)
    print(f"Training SAM2 large with LoRA on {args.dataset_size} dataset")
    if args.dataset_size == "full":
        print(f"Training data: datasets/Data/splits/train_split")
    else:
        print(f"Training data: datasets/Data/splits/subsets/{args.dataset_size}_subset")
    print(f"Validation data: datasets/Data/splits/val_split")
    print(f"Training steps: {args.steps}")
    print(f"Effective batch size: {args.batch_size * 4}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"LoRA dropout: {args.lora_dropout}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    
    # Create output directories
    os.makedirs("new_src/training/training_results/sam2_lora", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Train with LoRA
    try:
        result = train_sam2_lora(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            steps=args.steps,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            dataset_size=args.dataset_size
        )
        
        if result:
            print(f"\nLoRA training completed successfully!")
            print(f"Checkpoints saved to: {result}")
        else:
            print(f"\nLoRA training failed!")
            
    except Exception as e:
        print(f"Error during LoRA training: {e}")
        raise


if __name__ == "__main__":
    main()
