#!/usr/bin/env python3
"""
Example script showing how to train SAM2 with LoRA on different dataset sizes.

This script demonstrates the usage of train_lora_sam2.py with various configurations.
"""

import subprocess
import sys
import os

def run_training_example(dataset_size, steps=1000, lr=1e-3):
    """Run training with specified parameters"""
    
    print(f"\n{'='*60}")
    print(f"Training SAM2 with LoRA on {dataset_size} dataset")
    print(f"{'='*60}")
    
    cmd = [
        "python", "new_src/training/train_lora_sam2.py",
        "--dataset_size", dataset_size,
        "--steps", str(steps),
        "--learning_rate", str(lr),
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--batch_size", "1"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Starting training...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print("STDERR:", e.stderr[-500:])  # Last 500 chars
        return False

def main():
    """Run training examples for different dataset sizes"""
    
    print("SAM2 LoRA Training Examples")
    print("=" * 50)
    print("This script demonstrates training SAM2 with LoRA on different dataset sizes.")
    print("Available dataset sizes: full, 500, 1000, 2000")
    print("\nNote: This is a demonstration script. For actual training, use:")
    print("python new_src/training/train_lora_sam2.py --dataset_size <size> --steps <steps>")
    
    # Example configurations
    examples = [
        {"dataset_size": "500", "steps": 1000, "lr": 1e-3},
        {"dataset_size": "1000", "steps": 2000, "lr": 1e-3},
        {"dataset_size": "2000", "steps": 3000, "lr": 1e-3},
        {"dataset_size": "full", "steps": 5000, "lr": 1e-3},
    ]
    
    print(f"\nExample configurations:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['dataset_size']} images: {example['steps']} steps, lr={example['lr']}")
    
    print(f"\nTo run actual training, use commands like:")
    print(f"python new_src/training/train_lora_sam2.py --dataset_size 500 --steps 1000")
    print(f"python new_src/training/train_lora_sam2.py --dataset_size full --steps 10000")
    
    # Uncomment the following lines to run actual training (be careful with resources!)
    # print(f"\nRunning training examples...")
    # for example in examples:
    #     success = run_training_example(**example)
    #     if not success:
    #         print(f"Failed to train on {example['dataset_size']} dataset")
    #         break

if __name__ == "__main__":
    main()

