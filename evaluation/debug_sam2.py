#!/usr/bin/env python3
"""
Debug SAM2 mask generation specifically
"""

import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# Add new_src to path for utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import SAM2 utilities
sys.path.append('libs/sam2base')
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("✅ SAM2 imports successful")
except Exception as e:
    print(f"❌ SAM2 import failed: {e}")
    sys.exit(1)

def debug_sam2_pipeline():
    """Debug the SAM2 part of the pipeline"""
    
    # Paths
    sam2_checkpoint = "/home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_large_full_dataset_lr0.0001_20250829_1646/best_step11000_iou0.5124_dice0.6461.torch"
    test_images_dir = "/home/ptp/sam2/datasets/yolo_detection_fixed/images/test_split"
    yolo_labels_dir = "/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection/predict_test_corrected/labels"
    
    print(f"🔍 DEBUGGING SAM2 PIPELINE")
    print(f"Checkpoint: {sam2_checkpoint}")
    print(f"Checkpoint exists: {os.path.exists(sam2_checkpoint)}")
    
    # Load SAM2 model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config_file = "configs/sam2/sam2_hiera_l.yaml"
        
        print(f"Loading SAM2 model...")
        sam2 = build_sam2(config_file=config_file, ckpt_path=sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2)
        print(f"✅ SAM2 model loaded successfully on {device}")
    except Exception as e:
        print(f"❌ Failed to load SAM2 model: {e}")
        return
    
    # Get first image
    image_files = list(Path(test_images_dir).glob("*.jpg"))[:1]
    label_files = list(Path(yolo_labels_dir).glob("*.txt"))[:1]
    
    for image_file, label_file in zip(image_files, label_files):
        print(f"\n📷 Testing image: {image_file.name}")
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"❌ Could not load image")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image_rgb.shape[:2]
        print(f"Image dimensions: {img_width}x{img_height}")
        
        # Load YOLO predictions
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        bboxes = []
        for line in lines[:3]:  # Test first 3 boxes
            parts = line.strip().split()
            if len(parts) >= 5:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5]) if len(parts) >= 6 else 1.0
                
                # Convert to pixel coordinates
                cx, cy, w, h = x_center*img_width, y_center*img_height, width*img_width, height*img_height
                x1 = max(0, int(round(cx - w/2)))
                y1 = max(0, int(round(cy - h/2)))
                x2 = min(img_width-1, int(round(cx + w/2)))
                y2 = min(img_height-1, int(round(cy + h/2)))
                
                if x2 > x1 and y2 > y1:
                    bbox = [x1, y1, x2, y2]
                    bboxes.append(bbox)
                    print(f"  Box: {bbox} (conf: {confidence:.3f})")
        
        print(f"Total valid boxes: {len(bboxes)}")
        
        if not bboxes:
            print("❌ No valid bboxes found")
            continue
        
        # Test SAM2 prediction
        try:
            print(f"🎯 Testing SAM2 predictions...")
            predictor.set_image(image_rgb)
            print(f"✅ Image set in predictor")
            
            for i, bbox in enumerate(bboxes):
                print(f"\n  Testing box {i+1}: {bbox}")
                
                # Generate mask with SAM2
                input_box = np.array(bbox)
                masks_pred, scores_pred, logits_pred = predictor.predict(
                    box=input_box,
                    multimask_output=True
                )
                
                print(f"    Masks shape: {masks_pred.shape}")
                print(f"    Scores: {scores_pred}")
                print(f"    Logits shape: {logits_pred.shape if logits_pred is not None else 'None'}")
                
                # Select best mask
                best_idx = np.argmax(scores_pred)
                best_mask = masks_pred[best_idx]
                best_score = scores_pred[best_idx]
                
                print(f"    Best mask: idx={best_idx}, score={best_score:.3f}")
                print(f"    Best mask shape: {best_mask.shape}")
                print(f"    Best mask sum: {best_mask.sum()}")
                print(f"    Best mask unique: {np.unique(best_mask)}")
                
                if best_mask.sum() == 0:
                    print(f"    ⚠️ WARNING: Empty mask generated!")
                else:
                    print(f"    ✅ Valid mask generated")
                
                # Test probability conversion
                if logits_pred is not None:
                    best_logits = logits_pred[best_idx]
                    print(f"    Logits shape: {best_logits.shape}")
                    print(f"    Logits range: [{best_logits.min():.3f}, {best_logits.max():.3f}]")
                    
                    # Convert to probabilities
                    prob = torch.sigmoid(torch.from_numpy(best_logits)).numpy()
                    print(f"    Prob range: [{prob.min():.3f}, {prob.max():.3f}]")
                    
                    # Test thresholding
                    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        thresh_mask = (prob > thresh).astype(np.uint8)
                        print(f"      Threshold {thresh}: {thresh_mask.sum()} pixels")
        
        except Exception as e:
            print(f"❌ SAM2 prediction failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_sam2_pipeline()
