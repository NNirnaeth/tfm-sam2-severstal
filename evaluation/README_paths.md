# YOLO+SAM2 Pipeline - Paths Configuration

## ✅ Verified Paths

All paths have been verified and are ready for evaluation:

### Input Data
- **YOLO Predictions**: `/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_detection/predict_test_corrected/labels`
  - 2000 .txt files with YOLO format predictions
  - Generated from YOLO detection evaluation

- **Ground Truth Annotations**: `/home/ptp/sam2/datasets/Data/splits/test_split/ann`
  - 2000 .json files in Supervisely format
  - Contains bitmap data with origin coordinates

- **Test Images**: `/home/ptp/sam2/datasets/yolo_detection_fixed/images/test_split`
  - 2000 .jpg images for evaluation

### SAM2 Models Available (Ranked by Performance)
- **🥇 Best Model (Large)**: `/home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_large_full_dataset_lr0.0001_20250829_1646/best_step11000_iou0.5124_dice0.6461.torch`
  - IoU: 0.5124, Dice: 0.6461
  - **RECOMMENDED** - Best overall performance

- **🥈 Large 10K Steps**: `/home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_large_full_dataset_lr0.0001_20250829_1646/best_step10000_iou0.5076_dice0.6410.torch`
  - IoU: 0.5076, Dice: 0.6410
  - Very close to best performance

- **🥉 Large 9.5K Steps**: `/home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_large_full_dataset_lr0.0001_20250829_1646/best_step9500_iou0.5012_dice0.6341.torch`
  - IoU: 0.5012, Dice: 0.6341
  - Good performance

- **Small Model (Best)**: `/home/ptp/sam2/new_src/training/training_results/sam2_full_dataset/sam2_small_full_dataset_lr0.0001_20250829_1601/best_step4000_iou0.4302_dice0.5687.torch`
  - IoU: 0.4302, Dice: 0.5687
  - Faster inference, lower performance

### Output Directory
- **Results**: `/home/ptp/sam2/new_src/evaluation/evaluation_results/yolo_sam2_pipeline`
  - Will be created automatically
  - Contains CSV results and per-image metrics

## 🚀 Quick Start

### 1. Check Paths
```bash
./new_src/evaluation/check_paths.sh
```

### 2. Run Single Model Evaluation
```bash
./new_src/evaluation/run_yolo_sam2_eval_example.sh
```

### 3. Run Multiple Models
```bash
# Run all models
./new_src/evaluation/run_yolo_sam2_eval_models.sh

# Run specific model
./new_src/evaluation/run_yolo_sam2_eval_models.sh large_best
```

## 📊 Expected Results

The pipeline will generate:
- **Main results CSV**: Overall metrics (IoU, Dice, Precision, Recall, F1)
- **Per-image CSV**: Individual image results with names
- **Threshold analysis**: Optimal threshold from validation (if provided)
- **Preprocessing stats**: Box merging and widening statistics

## 🔧 Configuration Options

### Box Preprocessing
- `--iou_merge 0.85`: Merge overlapping boxes
- `--topk_boxes 20`: Limit boxes per image
- `--min_w_px 8`: Minimum width for widening
- `--pad_px 6`: Padding when widening

### Post-processing
- `--fill_holes`: Fill holes in masks
- `--morphology`: Apply morphological operations

### Threshold Management
- `--opt_threshold 0.5`: Use pre-calculated threshold (recommended)
- Without this flag: Optimize on test set (not recommended)

## ⚠️ Important Notes

1. **Ground Truth Format**: Uses Supervisely format with proper origin positioning
2. **Threshold Overfitting**: Always use `--opt_threshold` from validation
3. **Box Preprocessing**: Optimized for Severstal's slender defects
4. **Model Selection**: Large model recommended for best results

## 📈 Performance Expectations

Based on training results:
- **Large Model**: IoU ~0.51, Dice ~0.65
- **Small Model**: IoU ~0.43, Dice ~0.57
- **Pipeline**: Should improve over individual components
