# YOLO+SAM2 Pipeline Schema

## 🔄 Pipeline Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Test Images   │    │  YOLO Predictions│    │  Ground Truth   │    │  SAM2 Model     │
│   (2000 .jpg)   │    │   (2000 .txt)    │    │  (2000 .json)   │    │  (Base/Fine)    │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                       │                      │
          │                      │                       │                      │
          ▼                      ▼                       ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           MAIN EVALUATION LOOP                                          │
│  For each image in test set (2000 images):                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  1. LOAD DATA                                                                           │
│  • Load image (H×W)                                                                     │
│  • Load YOLO bboxes (normalized cx,cy,w,h,conf)                                        │
│  • Load GT annotations (Supervisely JSON with bitmap+origin)                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  2. BOX PREPROCESSING                                                                   │
│  • Convert normalized → pixel coordinates                                               │
│  • Merge duplicate boxes (IoU > 0.85)                                                  │
│  • Widen slender boxes (width < 8px) for SAM2 context                                  │
│  • Keep top-K boxes (20) by confidence                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  3. SAM2 SEGMENTATION                                                                   │
│  • Set image in SAM2 predictor                                                          │
│  • For each preprocessed bbox:                                                          │
│    - Use bbox as prompt                                                                 │
│    - Generate multiple masks (multimask_output=True)                                    │
│    - Select best mask by score                                                          │
│    - Get probability map from logits                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  4. MASK MERGING                                                                        │
│  • Merge multiple masks per image                                                       │
│  • Take maximum probability per pixel                                                   │
│  • Apply threshold (0.5 or optimal from validation)                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  5. POST-PROCESSING                                                                     │
│  • Fill holes (optional)                                                                │
│  • Morphological operations (optional)                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  6. GROUND TRUTH RECONSTRUCTION                                                        │
│  • Load bitmap patches from JSON                                                       │
│  • Use origin coordinates to place patches correctly                                   │
│  • Build union mask from all patches                                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  7. EVALUATION                                                                          │
│  • Calculate metrics: IoU, Dice, Precision, Recall, F1                                 │
│  • Store predictions and targets for threshold optimization                            │
│  • Store per-image results                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  8. THRESHOLD OPTIMIZATION                                                              │
│  • Use pre-calculated threshold from validation (recommended)                          │
│  • OR optimize on test set (not recommended - overfitting)                             │
│  • Calculate final metrics with optimal threshold                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  9. RESULTS SAVING                                                                      │
│  • Save main results CSV (overall metrics)                                             │
│  • Save per-image results CSV (with image names)                                       │
│  • Include all parameters for reproducibility                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Components

### Input Data
- **Images**: 2000 test images (1024×256 or similar)
- **YOLO Predictions**: Bounding boxes with confidence scores
- **Ground Truth**: Supervisely format with bitmap patches + origin coordinates

### Box Preprocessing (Critical for Severstal)
- **Duplicate Merging**: Removes overlapping detections
- **Slender Box Widening**: Expands narrow boxes (defects are often thin lines)
- **Top-K Selection**: Limits computational load

### SAM2 Processing
- **Box Prompts**: Uses YOLO bboxes as prompts (no points)
- **Multimask Output**: Selects best mask from multiple candidates
- **Probability Maps**: Maintains uncertainty information

### Ground Truth Handling
- **Origin-based Positioning**: Places bitmap patches at correct coordinates
- **No Resizing**: Avoids mask displacement artifacts
- **Union Construction**: Merges multiple annotation patches

### Evaluation
- **Multiple Metrics**: IoU, Dice, Precision, Recall, F1
- **Threshold Analysis**: IoU@50, IoU@75, IoU@90, AUPRC
- **Per-image Tracking**: Individual results with image names

## 🔧 Configuration Options

### Box Preprocessing
- `iou_merge`: IoU threshold for merging duplicates (default: 0.85)
- `topk_boxes`: Maximum boxes per image (default: 20)
- `min_w_px`: Minimum width for widening (default: 8px)
- `pad_px`: Padding when widening (default: 6px)

### Post-processing
- `fill_holes`: Fill holes in masks
- `morphology`: Apply morphological operations

### Threshold Management
- `opt_threshold`: Pre-calculated threshold from validation (recommended)
- Without flag: Optimize on test set (causes overfitting)

## 📊 Expected Output

### Main Results CSV
- Overall metrics across all images
- Preprocessing parameters used
- Threshold information

### Per-image Results CSV
- Individual image metrics
- Image names for traceability
- TP/FP/FN counts

## ⚠️ Important Notes

1. **Ground Truth Format**: Must be Supervisely with proper origin coordinates
2. **Threshold Overfitting**: Always use validation-optimized thresholds
3. **Box Preprocessing**: Critical for Severstal's slender defects
4. **Model Selection**: Base models vs fine-tuned models available









