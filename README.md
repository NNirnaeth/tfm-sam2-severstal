# SAM2 for Severstal Steel Defect Detection

This repository contains a comprehensive implementation and evaluation of SAM2 (Segment Anything Model 2) for steel defect detection on the Severstal dataset, along with extensive baseline models (UNet variants, YOLO detection/segmentation) for thorough comparison and ablation studies.

##  Project Overview

This project represents a complete investigation into modern segmentation approaches for industrial defect detection, specifically targeting steel manufacturing quality control. The research encompasses:

- **SAM2 Fine-tuning**: Multiple approaches including full fine-tuning, LoRA adaptation, and few-shot learning
- **Baseline Architectures**: UNet, UNet++, DSC-UNet with various backbones and learning rates
- **Detection + Segmentation Pipeline**: YOLO detection followed by SAM2 refinement
- **Comprehensive Evaluation**: Standardized metrics across all models for fair comparison

##  **EXPERIMENTAL WORKFLOW DIAGRAM**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SEVERSTAL DATASET (25,000+ images)                  │
│                    PNG bitmap compressed annotations (Supervisely format)      │
└─────────────────────────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PREPARATION                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Train Split   │  │   Val Split     │  │   Test Split    │                │
│  │   (80% data)    │  │   (10% data)    │  │   (10% data)    │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SUBSET GENERATION                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│  │                     │ │  500    │ │  1000   │ │  2000   │ │  4000   │     │
│  │                     │ │ images  │ │ images  │ │ images  │ │ images  │     │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
└─────────────────────────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL TRAINING PHASES                             │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 1: BASELINE MODELS                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │   │
│  │  │   UNet Standard │  │    UNet++       │  │   DSC-UNet      │        │   │
│  │  │   ResNet34/50   │  │   ResNet34      │  │   ResNet34      │        │   │
│  │  │   LR: 1e-3/1e-4│  │   LR: 1e-3/1e-4│  │   LR: 1e-3/1e-4│        │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 2: YOLO MODELS                                │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │   │
│  │  │  YOLO Detection │  │ YOLO Segment   │  │ YOLO+SAM2       │        │   │
│  │  │  Full Dataset   │  │ LR: 1e-3/5e-4/ │  │ Pipeline        │        │   │
│  │  │  Binary Defects │  │     1e-4        │  │ Detection→Refine│        │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    PHASE 3: SAM2 MODELS                                │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │   │
│  │  │   SAM2 Base     │  │ SAM2 Fine-tune  │  │   SAM2 + LoRA   │        │   │
│  │  │   (No training) │  │ Small/Large     │  │   Rank: 8/16    │        │   │
│  │  │   Zero-shot     │  │ Full Dataset    │  │   Alpha: 16     │        │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │              SAM2 SUBSET TRAINING (Few-shot)                   │   │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │   │   │
│  │  │  │  100    │ │  200    │ │  500    │ │  1000   │ │  2000   │ │   │   │
│  │  │  │ SAM2-S  │ │ SAM2-S  │ │ SAM2-S  │ │ SAM2-S  │ │ SAM2-S  │ │   │   │
│  │  │  │ SAM2-L  │ │ SAM2-L  │ │ SAM2-L  │ │ SAM2-L  │ │ SAM2-L  │ │   │   │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EVALUATION PHASE                                 │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    STANDARDIZED EVALUATION                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │   │
│  │  │   Same Test Set │  │  Same Metrics   │  │ Same Thresholds │        │   │
│  │  │   (2000 images) │  │  IoU, Dice, F1  │  │  Binary: 0.5   │        │   │
│  │  │   Same Protocol │  │  IoU@50/75/90/95│  │  Optimal: Val  │        │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    COMPARATIVE ANALYSIS                                │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │   │
│  │  │ Performance     │  │ Data Efficiency │  │ Architecture    │        │   │
│  │  │ Rankings        │  │ Scaling Laws    │  │ Ablation Study  │        │   │
│  │  │ mIoU, mDice     │  │ Subset Analysis │  │ LoRA vs Full    │        │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RESULTS STORAGE                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   CSV Results   │  │   Checkpoints   │  │   Visualizations│                │
│  │   All Metrics   │  │   Best Models   │  │   GT vs Pred    │                │
│  │   Hyperparams   │  │   Top 3 by IoU │  │   Before/After  │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

##  Project Structure

```
new_src/
├── training/                 # Training scripts for all models
│   ├── train_sam2_*.py     # SAM2 training (subsets, full dataset, LoRA)
│   ├── train_unet_*.py     # UNet variants training
│   ├── train_yolo_*.py     # YOLO detection and segmentation
│   ├── config.py           # Training configuration
│   └── training_results/   # All training outputs and checkpoints
├── evaluation/              # Evaluation scripts and results
│   ├── eval_sam2_*.py      # SAM2 evaluation across all variants
│   ├── eval_unet_*.py      # UNet comprehensive evaluation
│   ├── eval_yolo_*.py      # YOLO pipeline evaluation
│   ├── config.py           # Evaluation configuration
│   └── evaluation_results/ # All evaluation outputs and metrics
├── experiments/             # Experiment configurations and data preparation
├── utils/                   # Utility functions (prompts, post-processing, visualization)
└── .gitignore              # Excludes heavy result folders
```

##  **STANDARDIZED TRAINING & EVALUATION PROTOCOLS**

### ** COMPARABILITY GUIDELINES FOR TFM**

All experiments follow these standardized protocols to ensure fair comparison:

#### **1. DATA STANDARDIZATION**
- **Test Set**: Fixed 2000 images across all evaluations
- **Annotation Format**: PNG bitmap compressed in JSON (Supervisely)
- **Image Processing**: Consistent resizing and normalization
- **Data Augmentation**: Same augmentation pipeline for all models

#### **2. EVALUATION METRICS (UNIFIED ACROSS ALL MODELS)**
```
Primary Metrics:
├── mIoU (mean Intersection over Union)
├── mDice (mean Dice coefficient)
├── Precision, Recall, F1-score
└── AUPRC (Area Under Precision-Recall Curve)

Threshold Metrics:
├── IoU@50 (IoU ≥ 0.5)
├── IoU@75 (IoU ≥ 0.75)
├── IoU@90 (IoU ≥ 0.90)
└── IoU@95 (IoU ≥ 0.95)

Performance Metrics:
├── Average inference time
├── Total processing time
└── Memory usage
```

#### **3. TRAINING HYPERPARAMETERS (STANDARDIZED)**
```
Common Settings (All Models):
├── Seed: 42 (fixed for reproducibility)
├── CUDA: torch.backends.cudnn.deterministic=True
├── Mixed Precision: AMP enabled
├── Gradient Clipping: 1.0
└── Early Stopping: Patience 2-3 evaluations

Optimization:
├── Optimizer: AdamW
├── Weight Decay: 0.05 (LoRA), 1e-4 (UNet), 1e-4 (YOLO)
├── Scheduler: Cosine annealing with warmup
└── Loss: Dice + Focal BCE (λ=1.0 each)

Checkpointing:
├── Frequency: Every 500-1000 steps
├── Save Top: 3 models by validation mIoU
├── Save Last: True
└── Monitor: val_mIoU (mode='max')
```

#### **4. MODEL-SPECIFIC STANDARDIZATION**

##### **SAM2 Models**
```
Architecture Variants:
├── SAM2-Large: Full parameter fine-tuning
├── SAM2-Small: Speed-optimized variant
└── SAM2-LoRA: Low-rank adaptation (rank=8/16, alpha=16)

Training Approaches:
├── Full Fine-tuning: All parameters updated
├── LoRA: Only Q/K/V/O projections (99% parameter reduction)
└── Subset Training: 100, 200, 500, 1000, 2000, 4000 images

Prompt Engineering:
├── Points: 3-5 positive + 3-5 negative points
├── Boxes: Tight bounding boxes around defects
└── Full Image: Complete image coverage (ablation)
```

##### **UNet Models**
```
Architecture Variants:
├── UNet Standard: ResNet34/50 backbones
├── UNet++: Nested skip connections
└── DSC-UNet: Deep supervision at multiple levels

Training Parameters:
├── Image Size: 1024x256 (aspect ratio preserved)
├── Batch Size: 8-16
├── Learning Rate: 1e-3, 1e-4 variants
└── Loss: Dice + Focal BCE
```

##### **YOLO Models**
```
Detection Models:
├── Architecture: YOLOv8 variants
├── Purpose: Bounding box generation for SAM2
├── Training: Full dataset with converted annotations
└── Output: SAM2-compatible bounding boxes

Segmentation Models:
├── Architecture: YOLOv8-Seg
├── Learning Rates: 1e-3, 5e-4, 1e-4
├── Purpose: Direct segmentation comparison
└── Metrics: Same evaluation suite
```

#### **5. EVALUATION PROTOCOL (IDENTICAL FOR ALL MODELS)**
```
Test Execution:
├── Same 2000 test images
├── Same evaluation script structure
├── Same metric calculation
└── Same threshold calibration

Threshold Selection:
├── Binary Threshold: 0.5 (default)
├── Optimal Threshold: Selected on validation set
├── Applied: Consistently across all models
└── Recorded: In CSV results

Post-processing:
├── Morphology: Opening/closing operations
├── Hole Filling: Automatic defect completion
├── CRF: Optional refinement (configurable)
└── Applied: Same pipeline across all models
```

#### **6. RESULTS STORAGE FORMAT (UNIFIED)**
```
CSV Columns (All Experiments):
exp_id, model, subset_size, variant, prompt_type, img_size, batch_size, 
steps, lr, wd, seed, val_mIoU, val_Dice, test_mIoU, test_Dice, 
IoU@50, IoU@75, IoU@90, IoU@95, Precision, Recall, F1, AUPRC, 
avg_inference_time, ckpt_path, timestamp

File Naming Convention:
exp_{model}_{subset}_{prompt}_{imgsz}_{date-%Y%m%d-%H%M}.csv

Directory Structure:
├── training_results/
│   ├── sam2_subsets/          # Few-shot results
│   ├── sam2_full_dataset/     # Full dataset results
│   ├── sam2_lora/            # LoRA results
│   ├── unet_models/          # UNet variants
│   └── yolo_models/          # YOLO variants
└── evaluation_results/
    ├── csv/                   # Consolidated results
    ├── plots/                 # Performance comparisons
    └── masks/                 # Prediction examples
```

##  Models Implemented and Trained

### 1. **SAM2 (Segment Anything Model 2)**

#### **SAM2 Base (Zero-shot)**
- **Purpose**: Establish baseline performance without fine-tuning
- **Architecture**: SAM2-Large and SAM2-Small pre-trained models
- **Evaluation**: Same test set and metrics for fair comparison
- **Significance**: Shows foundation model capabilities on steel defects

#### **SAM2 Large - Few-Shot Learning (Subsets)**
- **Subset Sizes**: 100, 200, 500, 1000, 2000, 4000 images
- **Architecture**: SAM2-Large backbone
- **Training Strategy**: Full fine-tuning on limited data
- **Results**: 
  - 4000 images: mIoU=0.462, mDice=0.603, IoU@50=39.8%, IoU@75=10.5%
  - 2000 images: mIoU=0.403, mDice=0.538 (best checkpoint)
- **Key Finding**: Performance scales with data size, 4000 images shows significant improvement

#### **SAM2 Large - Full Dataset Fine-tuning**
- **Dataset**: Complete Severstal training set
- **Architecture**: SAM2-Large with full parameter updates
- **Training**: End-to-end fine-tuning on all available data
- **Purpose**: Establish upper bound performance baseline

#### **SAM2 Large - LoRA Adaptation**
- **Method**: Low-Rank Adaptation (LoRA) fine-tuning
- **Parameters**: Rank=8, Alpha=16, Dropout=0.05
- **Target**: Q/K/V/O projections in attention layers
- **Results**: 
  - Test mIoU: 0.505, Test mDice: 0.642
  - IoU@50: 50.5%, IoU@75: 42.4%, IoU@90: 28.8%, IoU@95: 19.6%
  - Precision: 0.730, Recall: 0.648, F1: 0.642
  - AUPRC: 0.779, Avg Inference: 0.214s
- **Advantage**: 99% parameter reduction while maintaining competitive performance

#### **SAM2 Small Variants**
- **Architecture**: SAM2-Small backbone (faster inference)
- **Training**: Parallel experiments with Large variant
- **Purpose**: Speed vs. accuracy trade-off analysis

### 2. **UNet Baseline Models**

#### **UNet Standard with ResNet34 Backbone**
- **Architecture**: Standard U-Net with ResNet34 encoder
- **Learning Rates**: 1e-3 and 1e-4 variants
- **Results**:
  - LR 1e-3: mIoU=0.078, mDice=0.132, IoU@50=7.8%, IoU@75=7.1%
  - LR 1e-4: mIoU=0.084, mDice=0.141, IoU@50=8.4%, IoU@75=7.3%
- **Performance**: Lower than SAM2 but establishes baseline for traditional architectures

#### **UNet++ (Enhanced U-Net)**
- **Architecture**: Nested skip connections for better feature fusion
- **Backbone**: ResNet34 encoder
- **Purpose**: Test if architectural improvements help with steel defect patterns

#### **DSC-UNet (Deep Supervision)**
- **Architecture**: Deep supervision at multiple decoder levels
- **Backbone**: ResNet34 encoder
- **Purpose**: Investigate multi-scale supervision benefits

### 3. **YOLO Detection + Segmentation Pipeline**

#### **YOLO Detection Models**
- **Architecture**: YOLOv8 detection variants
- **Purpose**: Generate bounding boxes for SAM2 refinement
- **Training**: Full Severstal dataset with converted annotations
- **Output**: Bounding box coordinates for defect regions

#### **YOLO Segmentation Models**
- **Architecture**: YOLOv8-Seg variants
- **Learning Rates**: 1e-3, 5e-4, 1e-4
- **Purpose**: Direct segmentation without SAM2 refinement
- **Comparison**: End-to-end vs. two-stage pipeline performance

#### **YOLO + SAM2 Pipeline**
- **Stage 1**: YOLO detection generates bounding boxes
- **Stage 2**: SAM2 refines bounding boxes into precise masks
- **Advantage**: Combines YOLO's speed with SAM2's segmentation quality

##  Comprehensive Evaluation Results

### **Standardized Metrics Across All Models**
All models are evaluated using the same comprehensive metric suite:

- **Primary Metrics**: mIoU, mDice (mean Dice coefficient)
- **Threshold Metrics**: IoU@50, IoU@75, IoU@90, IoU@95
- **Classification Metrics**: Precision, Recall, F1-score
- **Advanced Metrics**: AUPRC (Area Under Precision-Recall Curve)
- **Performance**: Average inference time, total processing time

### **Test Dataset**
- **Size**: 2000 annotated images from Severstal
- **Consistency**: Same test set used across all model evaluations
- **Annotation Format**: PNG bitmap compressed within JSON (Supervisely format)

### **Key Performance Insights**

#### **SAM2 Performance Hierarchy**
1. **LoRA SAM2 Large**: Best overall performance (mIoU=0.505, mDice=0.642)
2. **SAM2 Large Subsets**: 4000 images (mIoU=0.462, mDice=0.603)
3. **SAM2 Large Full Dataset**: Upper bound performance baseline
4. **SAM2 Small Variants**: Speed-optimized alternatives

#### **Baseline Model Performance**
- **UNet Variants**: mIoU range 0.078-0.084, establishing traditional architecture baseline
- **YOLO Pipeline**: Detection + refinement approach for real-time applications

#### **Data Efficiency Analysis**
- **Few-shot Learning**: 4000 images achieve 85% of full dataset performance
- **LoRA Advantage**: Efficient adaptation with minimal parameter overhead
- **Subset Scaling**: Clear correlation between data size and performance

##  Training Methodologies

### **SAM2 Training Approaches**

#### **Full Fine-tuning**
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-5 to 5e-5 range
- **Mixed Precision**: AMP enabled for memory efficiency
- **Gradient Clipping**: 1.0 for stability

#### **LoRA Fine-tuning**
- **Target Layers**: Q/K/V/O projections in attention
- **Rank Options**: 4, 8, 16 for ablation studies
- **Learning Rate**: 1e-3 (higher due to parameter efficiency)
- **Freezing**: Base model weights preserved

#### **Few-shot Learning**
- **Subset Generation**: Stratified sampling for defect balance
- **Data Augmentation**: Extensive augmentation for limited data
- **Early Stopping**: Patience-based with validation monitoring

### **UNet Training**
- **Loss Function**: Dice + Focal BCE combination
- **Image Size**: 1024x256 (aspect ratio preserved)
- **Augmentation**: Elastic transforms, noise, blur, color adjustments
- **Optimization**: AdamW with cosine annealing scheduler

### **YOLO Training**
- **Detection**: Binary defect vs. background
- **Segmentation**: Direct mask prediction
- **Pipeline**: Two-stage detection + refinement approach

##  Prompt Engineering and Post-processing

### **SAM2 Prompt Types**
- **Points**: 3-5 positive + 3-5 negative points per image
- **Boxes**: Tight bounding boxes around defect regions
- **Full Image**: Complete image coverage for ablation studies
- **Consistency**: Fixed prompt type per experiment

### **Post-processing Pipeline**
- **Morphological Operations**: Opening/closing for noise removal
- **Hole Filling**: Automatic defect region completion
- **Optional CRF**: Conditional Random Fields for refinement
- **Configurable**: Flags to activate/deactivate components

##  Results Storage and Reproducibility

### **Experiment Tracking**
- **CSV Format**: Comprehensive results with all hyperparameters
- **Columns**: exp_id, model, subset_size, variant, prompt_type, img_size, batch_size, steps, lr, wd, seed, val_mIoU, val_Dice, test_mIoU, test_Dice, IoU@50, IoU@75, IoU@90, IoU@95, Precision, Recall, F1, ckpt_path, timestamp

### **Checkpoint Management**
- **Frequency**: Every 500-1000 steps
- **Retention**: Top 3 models by validation mIoU
- **Format**: `best_step{step}_iou{val_iou}_dice{val_dice}.torch`

### **Configuration Management**
- **YAML Files**: All experiment parameters documented
- **Reproducibility**: Fixed seed=42, deterministic CUDA
- **Version Control**: All hyperparameters tracked

##  Usage Examples

### **Training SAM2 with LoRA**
```bash
python training/train_lora_sam2_full_dataset.py \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --steps 10000 \
    --batch_size 1 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05
```

### **Training UNet Baseline**
```bash
python training/train_unet_plus_plus.py \
    --lr 1e-4 \
    --batch_size 16 \
    --epochs 200 \
    --patience 10
```

### **Evaluating All Models**
```bash
# SAM2 LoRA evaluation
python evaluation/eval_lora_sam2_full_dataset.py

# UNet comprehensive evaluation
python evaluation/eval_all_unet_models.py

# YOLO pipeline evaluation
python evaluation/eval_yolo+sam2_full_dataset.py
```

##  Technical Requirements

### **Dependencies**
```bash
pip install -r requirements_unet.txt
```

### **Hardware Requirements**
- **GPU**: CUDA-compatible GPU (8GB+ VRAM recommended)
- **Memory**: 16GB+ RAM for large models
- **Storage**: 50GB+ for datasets and checkpoints

### **Reproducibility**
- **Seed**: Fixed at 42
- **CUDA**: Deterministic mode enabled
- **Versions**: All dependencies version-locked

##  License and Citation

This project is part of a Master's thesis on steel defect detection using different models. The implementation provides a comprehensive benchmark for modern segmentation approaches in industrial quality control applications.


