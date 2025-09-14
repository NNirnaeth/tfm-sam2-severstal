# SAM2 Evaluation Script Usage Examples

The updated `eval_lora_sam2_full_dataset.py` script now supports:
- **Model Types**: Base SAM2 (original Meta) or LoRA fine-tuned models
- **Model Sizes**: Small or Large SAM2 architectures
- **Evaluation Modes**: Auto-prompt, GT points (1-3), or 30 GT points

## Model Types

### Base SAM2 Models (Original Meta)
These are the original SAM2 models from Meta without any fine-tuning:
- `--model_type base`
- Uses the original `.pt` files from Meta
- No LoRA checkpoint required

### LoRA Fine-tuned Models
These are SAM2 models fine-tuned with LoRA adapters:
- `--model_type lora` (default)
- Requires `--lora_checkpoint` parameter
- Uses LoRA adapters on top of base SAM2

## Basic Usage

### 1. Base SAM2 Large with auto-prompt
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --model_type base \
    --model_size large \
    --evaluation_mode auto_prompt
```

### 2. LoRA SAM2 Large with GT points
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --model_type lora \
    --lora_checkpoint /path/to/lora_checkpoint.torch \
    --model_size large \
    --evaluation_mode gt_points \
    --num_gt_points 3
```

### 3. Base SAM2 Small with 30 GT points
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --model_type base \
    --model_size small \
    --evaluation_mode 30_gt_points
```

## Model Size Selection

### SAM2 Large (default)
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --lora_checkpoint /path/to/lora_large_checkpoint.torch \
    --model_size large \
    --evaluation_mode gt_points
```

### SAM2 Small
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --lora_checkpoint /path/to/lora_small_checkpoint.torch \
    --model_size small \
    --evaluation_mode gt_points
```

## Complete Examples

### Example 1: Base SAM2 Large with auto-prompt (original Meta model)
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --model_type base \
    --model_size large \
    --evaluation_mode auto_prompt \
    --save_examples
```

### Example 2: LoRA SAM2 Large with 3 GT points
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --model_type lora \
    --lora_checkpoint /home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_large_full_dataset_lr0.0001_20250829_1646/best_step11000_iou0.5124_dice0.6461.torch \
    --model_size large \
    --evaluation_mode gt_points \
    --num_gt_points 3 \
    --save_examples
```

### Example 3: Base SAM2 Small with 30 GT points
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --model_type base \
    --model_size small \
    --evaluation_mode 30_gt_points \
    --save_examples
```

### Example 4: LoRA SAM2 Small with auto-prompt
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --model_type lora \
    --lora_checkpoint /home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_small_full_dataset_lr0.0001_20250829_1601/best_step4000_iou0.4302_dice0.5687.torch \
    --model_size small \
    --evaluation_mode auto_prompt \
    --save_examples
```

### Example 5: Base SAM2 Large with 1 GT point
```bash
python new_src/evaluation/eval_lora_sam2_full_dataset.py \
    --model_type base \
    --model_size large \
    --evaluation_mode gt_points \
    --num_gt_points 1 \
    --save_examples
```

## Available Options

- `--model_type`: Choose between `base` (original Meta SAM2) or `lora` (fine-tuned with LoRA)
- `--model_size`: Choose between `small` or `large` SAM2 model
- `--evaluation_mode`: Choose between `auto_prompt`, `gt_points`, or `30_gt_points`
- `--num_gt_points`: Number of GT points (1-3) when using `gt_points` mode
- `--lora_checkpoint`: Path to LoRA checkpoint (required only for `lora` model_type)
- `--base_model`: Path to base SAM2 model (.pt file from Meta)
- `--save_examples`: Save prediction examples for visualization
- `--max_images`: Limit evaluation to a specific number of images (for testing)

## Output

The script will generate:
- CSV results file with comprehensive metrics
- JSON file with detailed results
- Optional visualization examples (if `--save_examples` is used)

Results are saved in the `new_src/evaluation/evaluation_results/sam2_lora/` directory by default.
