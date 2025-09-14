# TFM Graphics Generator

## Overview
Comprehensive graphics script for TFM thesis defense that generates all requested visualizations from evaluation results.

## Usage
```bash
cd /home/ptp/sam2/new_src/graphics
python graphics_tfm.py
```

## Output
The script generates 10 visualizations in `/home/ptp/sam2/new_src/graphics/tfm_output/`:

### 1. Global Model Comparison (`1_global_model_comparison.png`)
- Bar charts comparing mIoU and mDice across all models (SAM2, UNet, YOLO)
- Shows which architecture performs best

### 2. Efficiency Tradeoff (`2_efficiency_tradeoff.png`)
- Scatter plot: mIoU vs Inference Time
- Visualizes accuracy vs efficiency tradeoffs

### 3. Few-Shot Learning Curves (`3_few_shot_learning_curves.png`)
- Learning curves showing performance vs training subset size
- Demonstrates performance saturation around 1000-2000 images

### 4. Fine-tuning Impact (`4_finetuning_impact.png`)
- Comparison of Zero-shot vs LoRA vs Full fine-tuning strategies
- Shows how training strategy affects SAM2 performance

### 5. YOLO vs YOLO+SAM2 (`5_yolo_vs_yolo_sam2.png`)
- Bar chart comparing YOLO-Seg vs YOLO+SAM2 pipeline
- Demonstrates value added by SAM2 refinement

### 6. Learning Rate Sensitivity (`6_lr_sensitivity.png`)
- Line charts showing performance vs learning rate for UNet and YOLO
- Shows UNet stability vs YOLO sensitivity

### 7. Precision-Recall Curves (`7_pr_curves.png`)
- PR curves for top models (SAM2-LoRA, SAM2-FT, UNet, YOLO)
- Shows robustness across different thresholds

### 8. AUPRC Summary (`8_auprc_summary.png`)
- Bar chart of AUPRC scores for all models
- Quantitative comparison of PR performance

### 9. Radar Chart (`9_radar_chart.png`)
- Spider/radar chart comparing models across multiple metrics
- Visual representation of trade-offs (mIoU, Precision, Recall, Speed, Efficiency)

### 10. Summary Tables
- `10_summary_table.csv`: Comprehensive CSV with all metrics
- `10_summary_table.md`: Markdown table for easy viewing

## Data Sources
- **SAM2**: JSON files from `sam2_subsets/`, `sam2_lora/`, `sam2_full_dataset/`, `sam2_base/`
- **UNet**: CSV from `unet_models/model_comparison_table.csv`
- **YOLO**: CSV files from `yolo_segmentation/`

## Key Insights
- **SAM2-LoRA**: Optimal balance of accuracy and efficiency
- **Few-shot learning**: Performance saturates at 1000-2000 training images
- **Fine-tuning impact**: LoRA provides significant improvement over zero-shot
- **Model trade-offs**: Clear accuracy vs speed tradeoffs between architectures

## Requirements
```bash
pip install -r requirements_tfm.txt
```

## Notes
- Script automatically handles different data formats (JSON, CSV)
- Generates high-resolution plots (300 DPI) suitable for thesis
- All plots include proper labels, legends, and value annotations
- Output directory is created automatically
