# 📊 Graphics Generation for Severstal Dataset Analysis

This folder contains scripts to generate performance-efficiency plots and learning rate comparison charts for the Severstal steel defect detection models.

## 🎯 Available Scripts

### **1. `generate_performance_efficiency_plot.py`**
Generates a comprehensive performance-efficiency plot showing the trade-off between model performance (mIoU) and computational cost (parameters) for all models.

**Features:**
- **Performance-Efficiency Plot**: mIoU vs Parameters (Millions)
- **Model Categorization**: SAM2 (Diamonds), UNet (Circles), YOLO (Triangles)
- **Trend Lines**: Shows performance trends for each model type
- **Performance Regions**: Highlights high-performance and efficient regions
- **Embedded Table**: Key metrics summary
- **Insights Panel**: Key findings and recommendations

**Generated Plot:** `severstal_performance_efficiency_plot.png`

### **2. Learning Rate Comparison Charts**
Creates bar charts comparing the impact of different learning rates on model performance.

**Features:**
- **mIoU Comparison**: Performance across learning rates
- **mDice Comparison**: Dice coefficient across learning rates
- **Model Variants**: UNet Standard, DSC-UNet, UNet++, YOLOv8-Seg
- **Learning Rates**: 1e-4, 5e-4, 1e-3

**Generated Plot:** `severstal_learning_rate_comparison.png`

## 🚀 Usage

### **Installation**
```bash
# Install required dependencies
pip install -r requirements_plotting.txt
```

### **Execution**
```bash
# Navigate to graphics folder
cd new_src/graphics/

# Run the script
python generate_performance_efficiency_plot.py
```

### **Output Location**
All generated plots are saved in the **same directory** (`new_src/graphics/`):
- `severstal_performance_efficiency_plot.png`
- `severstal_learning_rate_comparison.png`

## 📈 Plot Descriptions

### **Performance-Efficiency Plot**
This plot is similar to the ADE20K performance-efficiency charts and shows:

- **X-axis**: Parameters (Millions) - from 0.8M to 1000M
- **Y-axis**: Severstal mIoU - from 0.1 to 0.75
- **Model Types**:
  - 🔴 **SAM2 Models** (Diamonds): Foundation models with various fine-tuning approaches
  - 🟣 **UNet Models** (Circles): Traditional CNN architectures
  - 🔻 **YOLO Models** (Triangles): Real-time segmentation models

**Key Insights Visualized:**
- **SAM2-LoRA**: Best efficiency-performance balance (0.8M params, 0.505 mIoU)
- **UNet Standard**: Most efficient traditional architecture (25M params, 0.557 mIoU)
- **SAM2-Large**: Highest performance but highest computational cost (1000M params, 0.544-0.687 mIoU)
- **YOLO**: Good balance for real-time applications (20M params, 0.458-0.480 mIoU)

### **Learning Rate Comparison**
Shows the impact of learning rate selection on model performance:

- **UNet Models**: LR=1e-4 consistently outperforms LR=1e-3
- **YOLO Models**: LR=1e-4 is optimal, LR=5e-4 is good, LR=1e-3 is poor
- **Key Finding**: Learning rate configuration is more critical than architectural improvements

## 🎨 Customization

### **Modifying Model Data**
Edit the `models_data` dictionary in the script to:
- Add new models
- Update performance metrics
- Change parameter estimates
- Modify colors and markers

### **Adjusting Plot Style**
Modify the plotting functions to:
- Change figure sizes
- Adjust color schemes
- Modify marker styles
- Update axis limits
- Customize annotations

## 📊 Data Sources

The plots use data from the comprehensive model comparison table in the main README:
- **SAM2 Models**: LoRA, Large FT, Large Base, Small FT, Small Base
- **UNet Models**: Standard (1e-4, 1e-3), DSC-UNet (1e-4, 1e-3), UNet++ (1e-4, 1e-3)
- **YOLO Models**: YOLOv8-Seg (1e-4, 5e-4, 1e-3)

## 🔧 Dependencies

- **matplotlib**: Plotting library
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **seaborn**: Statistical visualization styling

## 📝 Notes

- **Parameter Estimates**: Some parameter counts are estimates based on typical model architectures
- **Performance Metrics**: All metrics are from the actual evaluation results
- **Plot Resolution**: Generated at 300 DPI for publication quality
- **File Formats**: PNG format for compatibility and quality

## 🎯 Use Cases

These plots are ideal for:
- **TFM Presentations**: Visual representation of model performance
- **Research Papers**: Publication-quality figures
- **Performance Analysis**: Understanding model trade-offs
- **Architecture Selection**: Choosing models for specific applications
- **Learning Rate Optimization**: Understanding hyperparameter impact

## 🚨 Troubleshooting

### **Common Issues:**
1. **Import Errors**: Ensure all dependencies are installed
2. **Display Issues**: Script will save plots even if display fails
3. **File Permission**: Ensure write permissions in the graphics directory
4. **Memory Issues**: Large plots may require sufficient RAM

### **Solutions:**
- Install dependencies: `pip install -r requirements_plotting.txt`
- Check file permissions: `ls -la new_src/graphics/`
- Monitor memory usage during execution
- Use virtual environment if needed
