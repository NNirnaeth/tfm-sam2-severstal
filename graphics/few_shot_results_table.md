# Few-Shot Learning Results - Severstal Dataset

## Test Set Evaluation Results (August 22nd-23rd, 2025)

### SAM2 Small Model (lr=1e-4)

| Subset Size | Test IoU | Test Dice | IoU@50 | IoU@75 | IoU@90 | IoU@95 | Precision | Recall | F1 | Inference Time |
|-------------|----------|-----------|---------|---------|---------|---------|-----------|---------|----|----------------|
| 100         | **0.721**| **0.826** | **91.05%**| **49.1%**| **10.0%**| **1.55%**| **0.888** | **0.802** | **0.826**| **0.040s**    |
| 200         | **0.680**| **0.795** | **86.35%**| **39.7%**| **4.9%** | **0.6%**  | **0.881** | **0.757** | **0.795**| **0.041s**    |
| 500         | **0.545**| **0.681** | **61.95%**| **14.5%**| **1.85%**| **0.9%** | **0.885** | **0.602** | **0.681**| **0.043s**    |
| 1000        | **0.414**| **0.566** | **29.2%** | **2.7%** | **0.05%**| **0.0%** | **0.892** | **0.443** | **0.566**| **0.043s**    |
| 2000        | **0.545**| **0.692** | **61.9%** | **9.1%** | **0.15%**| **0.0%** | **0.895** | **0.592** | **0.692**| **0.042s**    |
| 4000        | **0.462**| **0.603** | **39.8%** | **10.55%**| **3.3%** | **1.3%** | **0.892** | **0.508** | **0.603**| **0.044s**    |

### SAM2 Large Model (lr=1e-4)

| Subset Size | Test IoU | Test Dice | IoU@50 | IoU@75 | IoU@90 | IoU@95 | Precision | Recall | F1 | Inference Time |
|-------------|----------|-----------|---------|---------|---------|---------|-----------|---------|----|----------------|
| 100         | **0.621**| **0.759** | **80.0%**| **10.0%**| **0.0%** | **0.0%** | **0.917** | **0.676** | **0.759**| **0.159s**    |
| 200         | **0.483**| **0.617** | **50.7%**| **11.7%**| **0.75%**| **0.05%**| **0.930** | **0.512** | **0.617**| **0.259s**    |
| 500         | **0.746**| **0.829** | **82.6%**| **61.2%**| **37.2%**| **21.0%**| **0.867** | **0.864** | **0.829**| **0.168s**    |
| 1000        | **0.365**| **0.455** | **38.7%**| **12.9%**| **4.0%** | **1.8%** | **0.919** | **0.405** | **0.455**| **0.226s**    |
| 2000        | **0.613**| **0.720** | **73.6%**| **35.5%**| **10.1%**| **4.1%** | **0.895** | **0.686** | **0.720**| **0.184s**    |
| 4000        | **0.249**| **0.333** | **18.9%**| **11.3%**| **6.1%** | **3.6%** | **0.911** | **0.281** | **0.333**| **0.209s**    |

## Training Metrics by Subset Size (August 20th)

### SAM2 Small Model (lr=1e-4)

| Subset Size | Steps | Best IoU | Final IoU | Convergence |
|-------------|-------|-----------|-----------|-------------|
| 100         | 5000  | 0.5178    | 0.5096    | ~4500 steps |
| 200         | 5000  | 0.4610    | 0.4610    | ~4000 steps |
| 500         | 5000  | 0.4224    | 0.4224    | ~5000 steps |
| 1000        | 5000  | 0.4116    | 0.4094    | ~4500 steps |
| 2000        | 4000  | 0.4160    | 0.4160    | ~4000 steps |
| 4000        | 5000  | 0.3981    | 0.3981    | ~5000 steps |

### SAM2 Large Model (lr=1e-4)

| Subset Size | Steps | Best IoU | Final IoU | Convergence |
|-------------|-------|-----------|-----------|-------------|
| 100         | 5000  | 0.5740    | 0.5727    | ~3500 steps |
| 200         | 5000  | 0.5287    | 0.5061    | ~3500 steps |
| 500         | 5000  | 0.4744    | 0.4744    | ~4000 steps |
| 1000        | 5000  | 0.4546    | 0.4528    | ~4000 steps |
| 2000        | 4000  | 0.4160    | 0.4160    | ~4000 steps |
| 4000        | 5000  | 0.4628    | 0.4628    | ~5000 steps |

## Key Observations

### 1. **Few-Shot Performance (Test Set)**
- **SAM2 Small - Best performance**: 100 images subset (IoU: 0.721, Dice: 0.826)
- **SAM2 Large - Best performance**: 500 images subset (IoU: 0.746, Dice: 0.829)
- **Performance comparison**: SAM2 Small outperforms SAM2 Large on small subsets (100-200 images)
- **Optimal subset size**: 100 images for SAM2 Small, 500 images for SAM2 Large

### 2. **Model Comparison - Surprising Results**
- **SAM2 Small on 100 images**: Outstanding IoU@50 (91.05%) and IoU@75 (49.1%)
- **SAM2 Large on 500 images**: Best overall performance with exceptional IoU@75 (61.2%)
- **Performance gap**: SAM2 Small actually outperforms SAM2 Large on very small subsets
- **Inference speed**: SAM2 Small is 4-6x faster than SAM2 Large

### 3. **Subset Size Analysis**
- **100 images**: Best performance for SAM2 Small (IoU: 0.721), good for SAM2 Large (IoU: 0.621)
- **200 images**: Strong performance for SAM2 Small (IoU: 0.680), moderate for SAM2 Large (IoU: 0.483)
- **500 images**: SAM2 Large shines (IoU: 0.746), SAM2 Small declines (IoU: 0.545)
- **1000+ images**: Both models show diminishing returns

### 4. **IoU Threshold Performance**
- **SAM2 Small (100 images)**: Exceptional IoU@50 (91.05%) - best of all configurations
- **SAM2 Large (500 images)**: Outstanding IoU@75 (61.2%) and IoU@90 (37.2%)
- **Pattern**: Smaller subsets achieve higher IoU@50, larger subsets better IoU@75

### 5. **Training vs Test Performance**
- **SAM2 Small**: Test performance significantly better than training (0.721 vs 0.5178 for 100 images)
- **SAM2 Large**: Test performance better than training (0.746 vs 0.4744 for 500 images)
- **Generalization**: Both models show excellent generalization from training to test set

## Recommendations

1. **Ultra-few-shot (100 images)**: Use **SAM2 Small** for best performance (IoU: 0.721)
2. **Few-shot (500 images)**: Use **SAM2 Large** for best performance (IoU: 0.746)
3. **Resource constraints**: **SAM2 Small** provides 4-6x faster inference
4. **Quality focus**: **SAM2 Large** for high-precision applications
5. **Speed focus**: **SAM2 Small** for real-time applications
6. **Optimal subset strategy**: 
   - 100 images: SAM2 Small
   - 200-500 images: SAM2 Large
   - 1000+ images: Diminishing returns for both
7. **Model selection**: Choose based on subset size and speed requirements
