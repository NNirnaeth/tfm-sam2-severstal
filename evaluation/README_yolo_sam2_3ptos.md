# YOLO+SAM2+3Points Evaluation Script

## Overview

The `eval_yolo+sam2+3ptos.py` script evaluates a combined YOLO detection + SAM2 segmentation pipeline using 1-3 GT points as additional prompts. This simulates human-in-the-loop interaction where users provide point clicks to refine segmentation.

## Key Features

- **Dual Prompting**: Combines YOLO bounding boxes with 1-3 GT-based point prompts
- **Smart Fallback**: When YOLO fails to detect, falls back to points-only prompting
- **GT Point Generation**: 
  - p1+: Centroid of main GT mask
  - p2+ (optional): Pixel with maximum distance to border
  - p3- (optional): Negative point 5-10px outside border
- **Comprehensive Metrics**: IoU, Dice, Precision, Recall, F1 with separate reporting for each method
- **Post-processing**: Optional morphological operations and hole filling
- **CSV Reporting**: Both summary and per-image results following TFM format

## Usage

### Basic Usage

```bash
python3 eval_yolo+sam2+3ptos.py \
    --yolo_path /path/to/yolo/model.pt \
    --sam2_checkpoint /path/to/sam2/checkpoint.torch \
    --gt_annotations /path/to/supervisely/annotations \
    --test_images_dir /path/to/test/images \
    --output_dir /path/to/results \
    --num_gt_points 2
```

### Using Pre-computed YOLO Predictions

```bash
python3 eval_yolo+sam2+3ptos.py \
    --yolo_path /path/to/yolo/predictions_dir \
    --sam2_checkpoint /path/to/sam2/checkpoint.torch \
    --gt_annotations /path/to/supervisely/annotations \
    --test_images_dir /path/to/test/images \
    --output_dir /path/to/results \
    --num_gt_points 2
```

### Full Example with Options

```bash
python3 eval_yolo+sam2+3ptos.py \
    --yolo_path training_results/yolo_detection/best.pt \
    --sam2_checkpoint training_results/sam2_large/best_step2000.torch \
    --sam2_model_type sam2_hiera_l \
    --gt_annotations ../../datasets/severstal/annotations_test \
    --test_images_dir ../../datasets/severstal/images_test \
    --output_dir evaluation_results/yolo_sam2_gt_points \
    --num_gt_points 2 \
    --conf_threshold 0.25 \
    --iou_threshold 0.70 \
    --fill_holes \
    --morphology \
    --batch_size 50
```

## Arguments

### Required Arguments
- `--yolo_path`: Path to YOLO model (.pt) or predictions directory (.txt files)
- `--sam2_checkpoint`: Path to SAM2 fine-tuned checkpoint
- `--gt_annotations`: Path to ground truth annotations (Supervisely format JSON)
- `--test_images_dir`: Directory containing test images

### Optional Arguments
- `--sam2_model_type`: SAM2 model type (default: sam2_hiera_l)
- `--output_dir`: Output directory for results (default: evaluation_results/yolo_sam2_gt_points)
- `--num_gt_points`: Number of GT points to generate 1-3 (default: 2, center + max distance)
- `--conf_threshold`: YOLO confidence threshold (default: 0.25, works better with points)
- `--iou_threshold`: YOLO IoU threshold for NMS (default: 0.70)
- `--fill_holes`: Apply hole filling post-processing
- `--morphology`: Apply morphological post-processing
- `--batch_size`: Number of images per batch (default: 50)

## GT Point Generation Strategy

The script generates 1-3 points from the ground truth mask:

1. **Point 1 (p1+)**: Centroid of the largest connected component
2. **Point 2 (p2+)**: Pixel with maximum distance to mask border (if num_points >= 2)
3. **Point 3 (p3-)**: Negative point 5-10 pixels outside the mask border (if num_points >= 3)

## Evaluation Methods

The script evaluates two scenarios:

### Box+Points Method
- Uses YOLO bounding boxes + GT points as prompts
- Applied when YOLO successfully detects objects
- Provides the main performance metric

### Points-Only Fallback
- Uses only GT points when YOLO fails to detect
- Prevents bias in recall calculations
- Reported separately to understand fallback performance

## Output Files

### Summary CSV
Contains aggregated metrics:
- Overall performance (all images)
- Box+Points performance (when YOLO detects)
- Points-only performance (fallback cases)
- Fallback rate and configuration details

### Per-Image CSV
Contains detailed results for each image:
- Method used (box_points vs points_only)
- Number of GT points and YOLO boxes
- Individual metrics (IoU, Dice, Precision, Recall, F1)
- Fallback usage flag

## Example Results Interpretation

```
📊 EVALUATION RESULTS:
Total images processed: 2000
Box+Points method: 1650 images
Points-only method: 350 images
Fallback rate: 17.5%

🎯 OVERALL PERFORMANCE:
  IoU:       0.7245
  Dice:      0.8156
  Precision: 0.8934
  Recall:    0.7523
  F1:        0.8167

📦 BOX+POINTS PERFORMANCE:
  IoU:       0.7456
  Dice:      0.8267
  F1:        0.8234

🎯 POINTS-ONLY PERFORMANCE:
  IoU:       0.6234
  Dice:      0.7456
  F1:        0.7823
```

## Best Practices

1. **Find Best Checkpoint**: Use `find_best_sam2_model.py` to locate the best SAM2 checkpoint
2. **Validate Paths**: Ensure all input paths exist before running
3. **Memory Management**: Use appropriate batch size for your GPU memory
4. **Comparison Studies**: Run with different numbers of GT points (1, 2, 3) to analyze the impact
5. **Fallback Analysis**: Monitor fallback rate - high rates may indicate YOLO detection issues

## Integration with TFM Pipeline

This script follows the TFM project structure and conventions:
- Uses the same metrics calculation system
- Saves results in CSV format compatible with other evaluation scripts
- Follows the same post-processing and configuration patterns
- Integrates with existing SAM2 checkpoint management

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure SAM2 libraries are properly installed and paths are set
2. **CUDA OOM**: Reduce batch size or use smaller SAM2 model
3. **Missing GT Points**: Some images may not generate valid GT points (empty masks)
4. **YOLO Detection Issues**: Check confidence thresholds if fallback rate is too high

### Performance Optimization
- Use pre-computed YOLO predictions to avoid redundant detection
- Enable post-processing only if needed
- Use appropriate batch sizes for memory efficiency
- Consider using SAM2-small for faster inference if acceptable performance

## Related Scripts

- `eval_yolo+sam2_full_dataset.py`: Standard YOLO+SAM2 evaluation without GT points
- `find_best_sam2_model.py`: Find the best SAM2 checkpoint
- `eval_sam2_*_full_dataset.py`: SAM2-only evaluation scripts
- `run_yolo_sam2_3ptos_example.sh`: Example usage script
