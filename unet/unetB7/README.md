# UNet with EfficientNet-B7 for High-Resolution Image Segmentation

This implementation provides a complete pipeline for image segmentation using UNet architecture with EfficientNet-B7 as the encoder, optimized for high-resolution images (256x1600) as used in steel defect detection.

## Features

- **High-Resolution Support**: Optimized for 256x1600 pixel images
- **EfficientNet-B7 Encoder**: Pre-trained on ImageNet for robust feature extraction
- **Advanced Data Augmentation**: Using Albumentations for better generalization
- **Multiple Loss Functions**: Dice loss, BCE+Dice, and Focal loss
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Easy-to-Use Scripts**: Training, evaluation, and prediction scripts

## Installation

### Option 1: Full Installation
```bash
# Install all dependencies including development tools
pip install -r requirements.txt
```

### Option 2: Minimal Installation
```bash
# Install only core dependencies
pip install -r requirements-minimal.txt
```

### Option 3: GPU Installation
```bash
# For systems with NVIDIA GPUs
pip install -r requirements-gpu.txt
```

### Option 4: Manual Installation
```bash
# Install core dependencies individually
pip install tensorflow>=2.10.0 opencv-python albumentations matplotlib scikit-learn numpy
```

### System Requirements
- **Python**: 3.8 or higher
- **TensorFlow**: 2.10.0 or higher
- **Memory**: At least 16GB RAM (recommended for 256x1600 images)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: At least 5GB free space for models and outputs

## Quick Start

### 1. Prepare Your Data

#### Standard Format
Organize your data in the following structure:
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

#### Severstal Steel Defect Dataset Format
For Severstal dataset with JSON annotations:
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── image1.jpg.json
    ├── image2.jpg.json
    └── ...
```

### 2. Train the Model

#### Standard Format
```bash
python train.py \
    --data_dir /path/to/your/data \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --output_dir ./outputs \
    --experiment_name steel_defect_detection
```

#### Severstal Format (with separate splits)
```bash
python train.py \
    --data_dir /path/to/severstal/splits \
    --use_severstal_format \
    --train_split train_split \
    --val_split val_split \
    --test_split test_split \
    --image_dir img \
    --annotation_dir ann \
    --epochs 100 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --output_dir ./outputs \
    --experiment_name severstal_defect_detection
```

#### Convert Severstal Dataset First
```bash
python convert_severstal.py \
    --data_dir /path/to/severstal/data \
    --image_dir images \
    --annotation_dir annotations \
    --output_mask_dir masks \
    --visualize
```

### 3. Evaluate the Model

```bash
python evaluate.py \
    --model_path ./outputs/steel_defect_detection/models/best_model.h5 \
    --data_dir /path/to/test/data \
    --output_dir ./evaluation_results
```

### 4. Make Predictions

```bash
python predict.py \
    --model_path ./outputs/steel_defect_detection/models/best_model.h5 \
    --input_path /path/to/image.jpg \
    --output_dir ./predictions \
    --save_masks \
    --save_overlay
```

## Architecture Details

### Model Architecture
- **Encoder**: EfficientNet-B7 (pre-trained on ImageNet)
- **Decoder**: UNet-style decoder with skip connections
- **Input Size**: 256x1600x3 (height, width, channels)
- **Output**: 256x1600x1 (binary segmentation mask)

### Skip Connections
The model uses skip connections from different levels of the EfficientNet encoder:
- Block 2a (1/4 resolution)
- Block 3a (1/8 resolution)
- Block 4a (1/16 resolution)
- Block 5a (1/32 resolution)
- Block 6a (1/64 resolution)

### Loss Functions
1. **Dice Loss**: Optimized for segmentation tasks
2. **BCE + Dice**: Combined binary cross-entropy and dice loss
3. **Focal Loss**: Handles class imbalance effectively

## Training Configuration

### Data Augmentation
- Horizontal/Vertical flips
- Random rotation (±10°)
- Random brightness/contrast adjustment
- Gaussian noise and blur
- Random resized crops

### Training Parameters
- **Optimizer**: Adam with cosine annealing
- **Learning Rate**: 1e-4 (initial)
- **Batch Size**: 4 (adjust based on GPU memory)
- **Epochs**: 100 (with early stopping)
- **Patience**: 20 epochs

### Callbacks
- **ModelCheckpoint**: Saves best model based on validation dice coefficient
- **ReduceLROnPlateau**: Reduces learning rate when validation metric plateaus
- **EarlyStopping**: Stops training when no improvement
- **TensorBoard**: Logs training metrics
- **CSVLogger**: Saves training history

## Evaluation Metrics

The model is evaluated using multiple metrics:

1. **Dice Coefficient**: Measures overlap between predicted and ground truth masks
2. **IoU (Intersection over Union)**: Similar to dice but with different normalization
3. **Pixel Accuracy**: Percentage of correctly classified pixels
4. **Precision/Recall/F1**: Standard classification metrics
5. **Confusion Matrix**: Detailed pixel-wise classification results

## Usage Examples

### Basic Training
```python
from unetB7 import UNetEfficientNetB7, DataLoader, DataAugmentation, Trainer

# Create model
model = UNetEfficientNetB7(input_shape=(256, 1600, 3), num_classes=1)
model.build_model()
model.compile_model(loss='bce_dice', learning_rate=1e-4)

# Create data loaders
train_loader = DataLoader(image_paths, mask_paths, batch_size=4)
val_loader = DataLoader(val_image_paths, val_mask_paths, batch_size=4)

# Train
trainer = Trainer(model, train_loader, val_loader)
history = trainer.train(epochs=100)
```

### Custom Evaluation
```python
from unetB7 import Evaluator

# Load trained model
model.load_model('path/to/model.h5')

# Create evaluator
evaluator = Evaluator(model.model)

# Evaluate dataset
metrics = evaluator.evaluate_dataset(test_loader, threshold=0.5)
print(f"Dice Coefficient: {metrics['dice_coefficient']:.4f}")
```

## Performance Tips

### Memory Optimization
- Reduce batch size if you encounter OOM errors
- Use mixed precision training (set in TensorFlow)
- Consider using gradient checkpointing for very large models

### Training Optimization
- Use data augmentation to increase dataset diversity
- Monitor validation metrics to prevent overfitting
- Use class weights for imbalanced datasets
- Experiment with different loss functions

### Inference Optimization
- Use TensorRT for faster inference
- Batch multiple images together
- Consider model quantization for deployment

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Use smaller input resolution
   - Enable mixed precision training

2. **Poor Performance**
   - Check data quality and annotations
   - Increase training epochs
   - Adjust learning rate
   - Try different loss functions

3. **Slow Training**
   - Use GPU acceleration
   - Increase batch size (if memory allows)
   - Use data prefetching

### Debugging
- Enable TensorBoard logging to monitor training
- Use the evaluation script to check model performance
- Visualize predictions to identify issues

## File Structure

```
unetB7/
├── __init__.py              # Package initialization
├── model.py                 # UNet model definition
├── data_utils.py            # Data loading and augmentation
├── trainer.py               # Training utilities
├── evaluator.py             # Evaluation metrics and visualization
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── predict.py               # Prediction script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## References

- [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Steel Defect Detection - Image Segmentation using Keras and Tensorflow](https://dipanshurana.medium.com/steel-defect-detection-image-segmentation-using-keras-and-tensorflow-6118bc586ad2)

## License

This project is open source and available under the MIT License.
