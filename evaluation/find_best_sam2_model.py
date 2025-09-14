#!/usr/bin/env python3
"""
Find the best SAM2 model by analyzing training results
"""

import os
import re
from pathlib import Path

def extract_metrics_from_filename(filename):
    """Extract IoU and Dice metrics from filename"""
    # Pattern: best_stepXXXX_iou0.XXXX_dice0.XXXX.torch
    pattern = r'best_step(\d+)_iou0\.(\d+)_dice0\.(\d+)\.torch'
    match = re.search(pattern, filename)
    
    if match:
        step = int(match.group(1))
        iou = float(match.group(2)) / 10000  # Convert from 0.XXXX to 0.XXXX
        dice = float(match.group(3)) / 10000
        return step, iou, dice
    return None, None, None

def find_best_models():
    """Find all SAM2 models and rank them by performance"""
    
    base_dir = Path("/home/ptp/sam2/new_src/training/training_results")
    models = []
    
    # Search in sam2_full_dataset
    full_dataset_dir = base_dir / "sam2_full_dataset"
    if full_dataset_dir.exists():
        for model_dir in full_dataset_dir.iterdir():
            if model_dir.is_dir():
                for file in model_dir.glob("best_step*_iou*_dice*.torch"):
                    step, iou, dice = extract_metrics_from_filename(file.name)
                    if step is not None:
                        models.append({
                            'path': str(file),
                            'type': 'full_dataset',
                            'size': 'large' if 'large' in model_dir.name else 'small',
                            'step': step,
                            'iou': iou,
                            'dice': dice,
                            'model_dir': str(model_dir)
                        })
    
    # Search in sam2_lora
    lora_dir = base_dir / "sam2_lora"
    if lora_dir.exists():
        for model_dir in lora_dir.iterdir():
            if model_dir.is_dir():
                for file in model_dir.glob("best_lora_step*_iou*_dice*.torch"):
                    step, iou, dice = extract_metrics_from_filename(file.name)
                    if step is not None:
                        models.append({
                            'path': str(file),
                            'type': 'lora',
                            'size': 'large',  # All LoRA models are large
                            'step': step,
                            'iou': iou,
                            'dice': dice,
                            'model_dir': str(model_dir)
                        })
    
    # Sort by IoU (primary) and Dice (secondary)
    models.sort(key=lambda x: (x['iou'], x['dice']), reverse=True)
    
    return models

def main():
    print("🔍 Analyzing SAM2 training results...")
    print("=" * 80)
    
    models = find_best_models()
    
    if not models:
        print("❌ No SAM2 models found!")
        return
    
    print(f"📊 Found {len(models)} SAM2 models")
    print()
    
    # Show top 10 models
    print("🏆 TOP 10 SAM2 MODELS (ranked by IoU):")
    print("-" * 80)
    print(f"{'Rank':<4} {'Type':<12} {'Size':<6} {'Step':<6} {'IoU':<8} {'Dice':<8} {'Path'}")
    print("-" * 80)
    
    for i, model in enumerate(models[:10], 1):
        print(f"{i:<4} {model['type']:<12} {model['size']:<6} {model['step']:<6} "
              f"{model['iou']:<8.4f} {model['dice']:<8.4f} {model['path']}")
    
    print()
    
    # Best model overall
    best = models[0]
    print("🥇 BEST MODEL OVERALL:")
    print(f"   Type: {best['type']}")
    print(f"   Size: {best['size']}")
    print(f"   Step: {best['step']}")
    print(f"   IoU:  {best['iou']:.4f}")
    print(f"   Dice: {best['dice']:.4f}")
    print(f"   Path: {best['path']}")
    print()
    
    # Best by type
    print("🏅 BEST BY TYPE:")
    
    # Best full dataset
    full_dataset_models = [m for m in models if m['type'] == 'full_dataset']
    if full_dataset_models:
        best_full = full_dataset_models[0]
        print(f"   Full Dataset: {best_full['size']} - IoU: {best_full['iou']:.4f}, Dice: {best_full['dice']:.4f}")
        print(f"   Path: {best_full['path']}")
    
    # Best LoRA
    lora_models = [m for m in models if m['type'] == 'lora']
    if lora_models:
        best_lora = lora_models[0]
        print(f"   LoRA: {best_lora['size']} - IoU: {best_lora['iou']:.4f}, Dice: {best_lora['dice']:.4f}")
        print(f"   Path: {best_lora['path']}")
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print(f"   🎯 For YOLO+SAM2 pipeline: Use {best['path']}")
    print(f"   📈 Best performance: IoU {best['iou']:.4f}, Dice {best['dice']:.4f}")
    
    if best['type'] == 'full_dataset':
        print("   ✅ Full dataset training - best for end-to-end performance")
    elif best['type'] == 'lora':
        print("   ⚡ LoRA training - faster inference, good performance")
    
    print()
    print("🔧 To use this model in the pipeline:")
    print(f"   --sam2_checkpoint \"{best['path']}\"")

if __name__ == "__main__":
    main()
