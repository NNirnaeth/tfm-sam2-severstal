# YOLO+SAM2 Pipeline Evaluation

## Overview
This pipeline evaluates the complete YOLO+SAM2 system on the Severstal test dataset:
1. **YOLO Detection**: Generates bounding boxes for defects
2. **Box Preprocessing**: Optimizes boxes for SAM2 (merge duplicates, widen slender boxes)
3. **SAM2 Segmentation**: Uses boxes as prompts to generate precise masks
4. **Evaluation**: Compares predicted masks against ground truth

## Key Features

### Box Preprocessing for SAM2
The pipeline includes intelligent preprocessing of YOLO boxes to improve SAM2 performance:

- **Duplicate Merging**: Removes overlapping boxes (IoU > 0.85) to reduce redundancy
- **Slender Box Widening**: Expands very narrow boxes (width < 8px) to provide context for SAM2
- **Top-K Limiting**: Keeps only the top 20 highest-confidence boxes per image
- **Confidence Preservation**: Maintains original confidence scores for analysis

### Ground Truth Reconstruction
Proper handling of Supervisely format annotations:

- **Origin-based Positioning**: Uses bitmap.origin coordinates to place patches correctly
- **No Resizing Artifacts**: Avoids mask displacement from resizing operations
- **Accurate Union**: Properly merges multiple annotation patches per image

### Threshold Management
Prevents overfitting on test set:

- **Pre-calculated Thresholds**: Use validation-optimized thresholds when available
- **Test Set Protection**: Warns when optimizing threshold on test data
- **Reproducible Results**: Consistent evaluation across experiments

### Parameters
- `--iou_merge 0.85`: IoU threshold for merging duplicate boxes
- `--topk_boxes 20`: Maximum boxes per image after preprocessing
- `--min_w_px 8`: Minimum width for widening slender boxes
- `--pad_px 6`: Padding when widening boxes
- `--opt_threshold 0.5`: Pre-calculated optimal threshold from validation (recommended)

## Usage

### Basic Evaluation
```bash
python new_src/evaluation/eval_yolo+sam2_full_dataset.py \
    --yolo_labels_dir /path/to/yolo/labels \
    --sam2_checkpoint /path/to/sam2/model.pt \
    --gt_annotations /path/to/gt/annotations \
    --test_images_dir /path/to/test/images \
    --output_dir /path/to/output
```

### With Custom Preprocessing
```bash
python new_src/evaluation/eval_yolo+sam2_full_dataset.py \
    --yolo_labels_dir /path/to/yolo/labels \
    --sam2_checkpoint /path/to/sam2/model.pt \
    --gt_annotations /path/to/gt/annotations \
    --test_images_dir /path/to/test/images \
    --output_dir /path/to/output \
    --iou_merge 0.80 \
    --topk_boxes 15 \
    --min_w_px 10 \
    --pad_px 8 \
    --opt_threshold 0.45
```

### With Validation Threshold (Recommended)
```bash
python new_src/evaluation/eval_yolo+sam2_full_dataset.py \
    --yolo_labels_dir /path/to/yolo/labels \
    --sam2_checkpoint /path/to/sam2/model.pt \
    --gt_annotations /path/to/gt/annotations \
    --test_images_dir /path/to/test/images \
    --output_dir /path/to/output \
    --opt_threshold 0.52  # From validation set optimization
```

## Output Files

### Results CSV
- **Main results**: `{timestamp}_yolo_sam2_pipeline_yolo_sam2_eval.csv`
- **Per-image results**: `{timestamp}_yolo_sam2_pipeline_per_image_results.csv` (includes image names)

### Metrics Included
- **Segmentation**: IoU, Dice, Precision, Recall, F1
- **Threshold Analysis**: IoU@50, IoU@75, IoU@90, AUPRC
- **Preprocessing**: Parameters used for box optimization
- **Post-processing**: Fill holes, morphology flags

## Severstal-Specific Optimizations

### Handling Slender Defects
The pipeline is optimized for Severstal's characteristic defects:
- **Longitudinal scratches**: Very tall, narrow boxes (h≈1.0, w≈0.01-0.05)
- **Context provision**: Widening gives SAM2 more context for accurate segmentation
- **Duplicate removal**: Prevents over-segmentation of long defects

### Performance Benefits
- **Reduced redundancy**: Merging duplicates reduces computational load
- **Better context**: Widened boxes improve SAM2 mask quality
- **Controlled complexity**: Top-K limiting prevents processing too many boxes

## Ablation Studies
The pipeline supports ablation studies by varying preprocessing parameters:
- **With/without widening**: Compare `min_w_px` settings
- **Different merge thresholds**: Test `iou_merge` values (0.7-0.9)
- **Top-K analysis**: Evaluate impact of `topk_boxes` (10-30)

## Integration with TFM Framework
- **Consistent metrics**: Uses same evaluation framework as other experiments
- **CSV format**: Compatible with TFM results analysis
- **Parameter tracking**: All preprocessing parameters saved for reproducibility
- **Timestamp-based naming**: Easy experiment tracking and comparison
