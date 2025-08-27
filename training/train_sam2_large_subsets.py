#!/usr/bin/env python3
"""
Train SAM2 Large model on different subset sizes from datasets/Data/splits/subsets
using the same training approach as train_sam2_small_large_full_dataset.py

Training data: datasets/Data/splits/subsets/{size}_subset (training images)
Validation data: datasets/Data/splits/val_split (internal validation)

This script creates models for each subset size to compare performance against full dataset training.
"""

import os
import sys
import numpy as np
import torch
import argparse
import json
from datetime import datetime
from tqdm import tqdm

# Fix imports for sam2 module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "libs", "sam2base"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Add src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from utils import read_batch
from metrics_logger import MetricsLogger

def read_batch_corrected(data, annotation_format='bitmap', visualize_data=False):
    """
    Corrected version of read_batch that maintains consistent dimensions
    between images and masks for SAM2 training.
    """
    import random
    import cv2
    import numpy as np
    import json
    import base64
    import zlib
    
    ent = random.choice(data)
    img_path = ent["image"]
    Img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if Img is None:
        print(f"Error reading image {img_path}")
        return None, None, None, 0

    if annotation_format == 'bitmap':
        try:
            with open(ent["annotation"], "r") as f:
                json_data = json.load(f)

            height = json_data["size"]["height"]
            width = json_data["size"]["width"]
            ann_map = np.zeros((height, width), dtype=np.uint8)

            for obj in json_data.get("objects", []):
                if obj.get("geometryType") != "bitmap":
                    continue
                bitmap_data = obj["bitmap"]["data"]
                origin_x, origin_y = obj["bitmap"]["origin"]

                # Decode bitmap data
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

        except Exception as e:
            print(f"[Error decoding bitmap] {ent['annotation']}: {e}")
            return None, None, None, 0

    else:
        print(f"Unsupported annotation format: {annotation_format}")
        return None, None, None, 0

    # Convert BGR to RGB
    Img = Img[..., ::-1]
    
    # Don't resize - let SAM2 handle the resizing internally
    img = Img
    ann_map = ann_map

    # Create binary mask
    binary_mask = (ann_map > 0).astype(np.uint8)
    
    # Sample points from the mask
    eroded = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    coords = np.argwhere(eroded > 0)
    
    if len(coords) > 0:
        np.random.shuffle(coords)
        points = coords[:10]  # Take up to 10 points
        # Convert from (y, x) to (x, y) format and ensure positive strides
        points = points[:, [1, 0]].copy()  # Swap y,x to x,y and make copy
        points = np.expand_dims(points, axis=1)
    else:
        points = np.zeros((0, 1, 2))

    # Add batch dimension to mask
    binary_mask = np.expand_dims(binary_mask, axis=0)

    n_instances = len(np.unique(ann_map)) - 1  # ignore background 0
    return img, binary_mask, points, n_instances


def prepare_subset_with_validation(subset_size):
    """
    Prepare subset data using the correct directory structure.
    Uses {subset_size}_subset for training and val_split for internal validation.
    """
    # Use the correct directory structure from datasets/Data/splits/subsets
    train_dir = f"datasets/Data/splits/subsets/{subset_size}_subset"
    val_dir = "datasets/Data/splits/val_split"
    
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
    
    # Load training and validation data
    train_data = load_pairs(train_dir)
    val_data = load_pairs(val_dir)
    
    print(f"Subset {subset_size} dataset loaded:")
    print(f"  Training images: {len(train_data)} (from {train_dir})")
    print(f"  Validation images: {len(val_data)} (from {val_dir})")
    
    return train_data, val_data


def evaluate_on_validation_set(predictor, val_data, annotation_format="bitmap"):
    """
    Evaluate the model on the internal validation set.
    Returns mean IoU and Dice coefficient across all validation images.
    
    Note: Uses bitmap annotations in Supervisely JSON format for consistency
    with the TFM evaluation pipeline.
    """
    predictor.model.eval()
    
    all_ious = []
    all_dices = []
    
    with torch.no_grad():
        for entry in val_data:
            try:
                # Load image and annotation using original function
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
                
                # Post-process masks using SAM2's internal dimensions
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                
                # Calculate metrics
                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                # Probabilities per-instance: [N, H, W]
                prd_prob = torch.sigmoid(prd_masks[:, 0])
                # Combine multi-instance predictions using max across instances
                if prd_prob.dim() == 3:
                    prd_prob = prd_prob.max(dim=0).values  # [H, W]
                # Binary prediction
                pred_binary = (prd_prob > 0.5).float()
                
                # Ensure masks are 2D (remove batch dimension if present)
                if gt_mask.dim() > 2:
                    gt_mask = gt_mask.squeeze(0)  # Remove batch dimension
                if pred_binary.dim() > 2:
                    pred_binary = pred_binary.squeeze(0)  # Remove batch dimension
                
                # Ensure both masks have the same shape
                if gt_mask.shape != pred_binary.shape:
                    print(f"  Shape mismatch after processing - gt: {gt_mask.shape}, pred: {pred_binary.shape}")
                    # Resize prediction to match ground truth
                    pred_binary = torch.nn.functional.interpolate(
                        pred_binary.unsqueeze(0).unsqueeze(0), 
                        size=gt_mask.shape, 
                        mode='nearest'
                    ).squeeze()
                
                # Flatten masks for metric calculation
                gt_flat = gt_mask.flatten()
                pred_flat = pred_binary.flatten()
                
                # IoU calculation with proper dimension handling
                inter = (gt_flat * pred_flat).sum()
                union = gt_flat.sum() + pred_flat.sum() - inter
                iou = inter / (union + 1e-6)
                
                # Clamp IoU to valid range [0, 1]
                iou = torch.clamp(iou, 0.0, 1.0)
                
                # Dice coefficient calculation
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


def train_sam2_large_on_subset(subset_size, learning_rate, weight_decay, steps, batch_size=1):
    """
    Train SAM2 large model on a specific subset size.
    
    Args:
        subset_size: size of the subset (100, 200, 500, 1000, 2000, 4000)
        learning_rate: learning rate for training
        weight_decay: weight decay for optimizer
        steps: number of training steps
        batch_size: batch size (effective batch size with accumulation)
    
    Note: Uses bitmap annotations and consistent metrics for TFM evaluation.
    """
    
    print(f"\n{'='*80}")
    print(f"Training SAM2 LARGE on Subset {subset_size}")
    print(f"{'='*80}")
    print(f"Subset Size: {subset_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Training Steps: {steps}")
    print(f"Effective Batch Size: {batch_size * 4} (batch_size={batch_size}, accumulation=4)")
    
    # Configuration
    base_model = "models/sam2_base_models/sam2_hiera_large.pt"
    config = "configs/sam2/sam2_hiera_l.yaml"
    
    # Check if base model exists
    if not os.path.exists(base_model):
        print(f"Error: Base model not found: {base_model}")
        return None
    
    # Prepare dataset with internal validation
    train_data, val_data = prepare_subset_with_validation(subset_size)
    
    if len(train_data) == 0:
        print(f"Error: No training data found for subset {subset_size}")
        return None
    
    # Build model
    print(f"Building SAM2 large model...")
    sam2_model = build_sam2(config, base_model, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Set training mode
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        predictor.model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Cosine annealing scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=1000,  # Restart every 1000 steps
        T_mult=2,  # Double the restart interval each time
        eta_min=learning_rate * 0.01  # Minimum LR
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training configuration
    accumulation_steps = 4
    best_val_iou = 0.0
    mean_iou = 0.0
    
    # Early stopping configuration
    early_stopping_patience = 3
    early_stopping_counter = 0
    early_stopping_min_delta = 1e-4  # Minimum improvement threshold
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"sam2_large_subset_{subset_size}_lr{str(learning_rate).replace('e-', 'e')}_{timestamp}"
    ckpt_dir = f"new_src/training/training_results/sam2_subsets/{model_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Metrics logger
    logger = MetricsLogger(f"logs/training_metrics_subset_{subset_size}.csv")
    
    # Training loop
    print(f"\nStarting training on subset {subset_size}...")
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    
    for step in range(1, steps + 1):
        try:
            with torch.amp.autocast('cuda'):
                # Read batch using original function (proven to work)
                image, mask, input_point, num_masks = read_batch(
                    train_data, annotation_format="bitmap", visualize_data=False
                )
                
                if image is None or mask is None or num_masks == 0:
                    continue
                
                # Prepare labels
                input_label = np.ones((input_point.shape[0], 1), dtype=np.int32)
                if input_point.size == 0 or input_label.size == 0:
                    continue
                
                # Set image and prepare prompts
                predictor.set_image(image)
                mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                    input_point, input_label, box=None, mask_logits=None, normalize_coords=True
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
                
                # Post-process masks using SAM2's internal dimensions
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                
                # Calculate loss (consistent with src/train.py)
                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                # Probabilities per-instance: [N, H, W]
                prd_prob = torch.sigmoid(prd_masks[:, 0])
                # Combine multi-instance predictions using max across instances
                if prd_prob.dim() == 3:
                    prd_prob = prd_prob.max(dim=0).values  # [H, W]
                prd_mask = prd_prob
                
                # Ensure both masks have the same dimensions for loss calculation
                if gt_mask.dim() == 2:
                    gt_mask = gt_mask.unsqueeze(0)  # Add batch dimension [1, H, W]
                if prd_mask.dim() == 2:
                    prd_mask = prd_mask.unsqueeze(0)  # Add batch dimension [1, H, W]
                
                # Segmentation loss (Binary Cross Entropy)
                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) - 
                           (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
                
                # IoU loss - CORREGIDO: Con clamping y protección contra división por cero
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
                iou = inter / (union + 1e-6)  # Protección contra división por cero
                iou = torch.clamp(iou, 0.0, 1.0)  # Clamping para asegurar IoU <= 1.0
                
                # Score loss (predicted vs actual IoU)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                
                # Combined loss
                loss = seg_loss + score_loss * 0.05
                loss = loss / accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                if step % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    predictor.model.zero_grad()
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
                        best_ckpt_path = os.path.join(ckpt_dir, f"best_step{step}_iou{val_iou:.4f}_dice{val_dice:.4f}.torch")
                        torch.save(predictor.model.state_dict(), best_ckpt_path)
                        print(f"  New best model saved: {best_ckpt_path}")
                        
                        # Reset early stopping counter on improvement
                        early_stopping_counter = 0
                        
                        # Remove previous best checkpoints (keep only top 3)
                        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("best_step")]
                        if len(ckpt_files) > 3:
                            ckpt_files.sort(key=lambda x: float(x.split("iou")[1].split("_")[0]), reverse=True)
                            for old_ckpt in ckpt_files[3:]:
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
                    latest_ckpt_path = os.path.join(ckpt_dir, f"latest_step{step}.torch")
                    torch.save(predictor.model.state_dict(), latest_ckpt_path)
                    
                    # Log metrics (consistent with TFM metrics)
                    logger.log(
                        mode="train",
                        dataset=f"severstal_subset_{subset_size}",
                        steps=step,
                        model_name=model_name,
                        iou=mean_iou,
                        iou50=None,  # Will be calculated during evaluation
                        iou75=None,  # Will be calculated during evaluation
                        iou95=None   # Will be calculated during evaluation
                    )
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            continue
    
    # Save final model
    final_ckpt_path = os.path.join(ckpt_dir, f"final_step{steps}.torch")
    torch.save(predictor.model.state_dict(), final_ckpt_path)
    
    # Save training summary
    summary = {
        "subset_size": subset_size,
        "model_size": "large",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "total_steps": steps,
        "actual_steps": step,  # May be less due to early stopping
        "best_val_iou": best_val_iou,
        "final_train_iou": mean_iou,
        "early_stopping_triggered": early_stopping_counter >= early_stopping_patience,
        "early_stopping_patience": early_stopping_patience,
        "timestamp": timestamp,
        "checkpoint_dir": ckpt_dir,
        "note": "Subset training with same metrics and parameters as full dataset for fair comparison. Early stopping with patience=3."
    }
    
    summary_path = os.path.join(ckpt_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining on subset {subset_size} completed!")
    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping triggered at step {step}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Final training IoU: {mean_iou:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Training summary: {summary_path}")
    print(f"Early stopping: patience={early_stopping_patience}, triggered={early_stopping_counter >= early_stopping_patience}")
    
    return ckpt_dir


def main():
    parser = argparse.ArgumentParser(description="Train SAM2 Large on different subset sizes")
    parser.add_argument("--subset_sizes", nargs="+", 
                       default=["100", "200", "500", "1000", "2000", "4000"],
                       help="List of subset sizes to train on")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay for optimizer")
    parser.add_argument("--steps", type=int, default=5000,
                       help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (effective batch = batch_size * 4)")
    
    args = parser.parse_args()
    
    print("SAM2 Large Subsets Training Script")
    print("=" * 50)
    print(f"Training SAM2 large on subsets: {args.subset_sizes}")
    print(f"Training data: datasets/Data/splits/subsets/*_subset")
    print(f"Validation data: datasets/Data/splits/val_split")
    print(f"Training steps: {args.steps}")
    print(f"Effective batch size: {args.batch_size * 4}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    
    # Create output directories
    os.makedirs("new_src/training/training_results/sam2_subsets", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    results = {}
    
    # Train on each subset
    for subset_size in args.subset_sizes:
        print(f"\n{'='*80}")
        print(f"TRAINING ON SUBSET {subset_size}")
        print(f"{'='*80}")
        
        try:
            result = train_sam2_large_on_subset(
                subset_size=subset_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                steps=args.steps,
                batch_size=args.batch_size
            )
            results[subset_size] = {
                'success': True,
                'checkpoint_dir': result
            }
        except Exception as e:
            print(f"Error training on subset {subset_size}: {e}")
            results[subset_size] = {
                'success': False,
                'error': str(e)
            }
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    for subset_size, result in results.items():
        if result['success']:
            print(f"Subset {subset_size}: SUCCESS - {result['checkpoint_dir']}")
        else:
            print(f"Subset {subset_size}: FAILED - {result['error']}")
    
    # Save overall results
    results_file = f"new_src/training/training_results/sam2_subsets/sam2_large_subsets_training_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    os.makedirs("results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nOverall results saved to: {results_file}")
    print(f"Models are ready for evaluation and comparison with full dataset training!")


if __name__ == "__main__":
    main()
