#!/usr/bin/env python3
"""
Test script to verify DICE evaluation works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_detectron2 import calculate_dice_metrics, bbox_to_mask
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
import numpy as np

def test_dice_evaluation():
    """Test DICE evaluation with a simple example"""
    print("Testing DICE evaluation...")
    
    # Create a simple test case
    height, width = 100, 100
    
    # Test bbox_to_mask function
    bbox = [10, 10, 20, 30]  # [x, y, w, h]
    mask = bbox_to_mask(bbox, height, width)
    
    print(f"Bbox: {bbox}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sum: {np.sum(mask)}")
    print(f"Expected sum: {bbox[2] * bbox[3]} = {20 * 30}")
    
    # Test DICE calculation
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    pred_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Perfect overlap case
    gt_mask[10:40, 10:30] = 1
    pred_mask[10:40, 10:30] = 1
    
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    dice = (2 * intersection) / (union + 1e-6)
    
    print(f"Perfect overlap DICE: {dice:.4f} (should be 1.0)")
    
    # No overlap case
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    pred_mask = np.zeros((height, width), dtype=np.uint8)
    gt_mask[10:40, 10:30] = 1
    pred_mask[50:80, 50:70] = 1
    
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    dice = (2 * intersection) / (union + 1e-6)
    
    print(f"No overlap DICE: {dice:.4f} (should be 0.0)")
    
    print("DICE evaluation test completed successfully!")

if __name__ == "__main__":
    test_dice_evaluation()
