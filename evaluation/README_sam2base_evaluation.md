# Evaluación de SAM2 Base (Sin Fine-tuning) - Múltiples Modos

Este script evalúa los modelos SAM2 Small y Large base (sin fine-tuning) en el dataset de test de Severstal con **tres modos de evaluación diferentes** para comparar rendimiento realista vs teórico.

## 🎯 Modos de Evaluación

### 1. **30_gt_points** (Modo Original)
- **Descripción**: Usa 30 puntos extraídos directamente del Ground Truth
- **Uso**: Comparación con evaluaciones anteriores, rendimiento máximo teórico
- **Realismo**: ❌ No realista (requiere GT perfecto)
- **Rendimiento esperado**: Alto (prompts perfectos)

### 2. **gt_points** (Modo Realista con Anotación)
- **Descripción**: Usa 1-3 puntos del GT (centroide + punto interior + punto negativo)
- **Uso**: Simula anotación manual realista
- **Realismo**: ✅ Realista (anotación manual típica)
- **Rendimiento esperado**: Medio-Alto

### 3. **auto_prompt** (Modo Completamente Automático)
- **Descripción**: Genera prompts automáticamente sin usar GT (grid + boxes)
- **Uso**: Evaluación completamente automática, sin intervención humana
- **Realismo**: ✅ Muy realista (producción real)
- **Rendimiento esperado**: Bajo-Medio

## 🚀 Uso del Script

### Argumentos Principales

```bash
python evaluate_sam2base_large_small.py [OPCIONES]
```

**Opciones disponibles:**
- `--evaluation_mode`: Modo de evaluación (`auto_prompt`, `gt_points`, `30_gt_points`)
- `--num_gt_points`: Número de puntos GT (1-3) cuando `evaluation_mode=gt_points`
- `--model_size`: Tamaño del modelo (`small`, `large`, `both`)
- `--results_dir`: Directorio para guardar resultados
- `--num_viz`: Número de visualizaciones a generar

### Ejemplos de Uso

#### 1. Evaluación Completa (Todos los Modos)
```bash
# Modo 1: 30 puntos GT (máximo teórico)
python evaluate_sam2base_large_small.py --evaluation_mode 30_gt_points --model_size both

# Modo 2: 3 puntos GT (realista)
python evaluate_sam2base_large_small.py --evaluation_mode gt_points --num_gt_points 3 --model_size both

# Modo 3: Prompts automáticos (completamente automático)
python evaluate_sam2base_large_small.py --evaluation_mode auto_prompt --model_size both
```

#### 2. Evaluación Rápida (Solo Large)
```bash
# Con 3 puntos GT
python evaluate_sam2base_large_small.py --evaluation_mode gt_points --num_gt_points 3 --model_size large --num_viz 10
```

#### 3. Comparación de Puntos GT
```bash
# 1 punto (solo centroide)
python evaluate_sam2base_large_small.py --evaluation_mode gt_points --num_gt_points 1 --model_size large

# 2 puntos (centroide + punto interior)
python evaluate_sam2base_large_small.py --evaluation_mode gt_points --num_gt_points 2 --model_size large

# 3 puntos (centroide + punto interior + punto negativo)
python evaluate_sam2base_large_small.py --evaluation_mode gt_points --num_gt_points 3 --model_size large
```

## 📊 Estructura de Resultados

Los resultados se guardan en:
```
evaluation_results/sam2_base/sam2base_evaluation_{modo}_{timestamp}/
├── sam2_small_base_no_fine-tuning_{modo}/
│   ├── sam2base_sam2_small_base_no_fine-tuning_{modo}_{timestamp}.csv
│   └── visualizations/
├── sam2_large_base_no_fine-tuning_{modo}/
│   ├── sam2base_sam2_large_base_no_fine-tuning_{modo}_{timestamp}.csv
│   └── visualizations/
├── final_results_all_models.json
└── evaluation_summary.txt
```

## 📈 Métricas Incluidas

- **Métricas de Segmentación**: IoU, Dice, Precision, Recall, F1
- **Métricas de Umbral**: IoU@50, IoU@75, IoU@90, IoU@95
- **Métricas Benevolentes**: Benevolent@75, Benevolent@any
- **Métricas de Rendimiento**: Tiempo de inferencia, imágenes procesadas
- **Visualizaciones**: Análisis TP/FP/FN para muestras aleatorias

## ⏱️ Tiempos Estimados

| Modelo | Modo | Tiempo Estimado |
|--------|------|-----------------|
| Small | 30_gt_points | ~2.5 horas |
| Small | gt_points | ~2.5 horas |
| Small | auto_prompt | ~3.0 horas |
| Large | 30_gt_points | ~5.0 horas |
| Large | gt_points | ~5.0 horas |
| Large | auto_prompt | ~6.0 horas |

## 🔍 Interpretación de Resultados

### Comparación de Modos
1. **30_gt_points**: Rendimiento máximo teórico con prompts perfectos
2. **gt_points**: Rendimiento realista con anotación manual típica
3. **auto_prompt**: Rendimiento completamente automático sin GT

### Diferencias Esperadas
- **30_gt_points > gt_points > auto_prompt** en términos de IoU/F1
- **auto_prompt** es el más realista para aplicaciones de producción
- **gt_points** balancea realismo y rendimiento

## 🛠️ Configuración Técnica

### Hiperparámetros de Auto-Prompt
- **Grid Spacing**: 128px
- **Box Scales**: [128, 256]px
- **Confidence Threshold**: 0.7
- **NMS IoU Threshold**: 0.65
- **Top-K Filter**: 200

### Generación de Puntos GT
- **1 punto**: Centroide del defecto
- **2 puntos**: Centroide + punto más interior
- **3 puntos**: Centroide + punto interior + punto negativo

## 📝 Notas Importantes

1. **Reproducibilidad**: Seed fijo (42) para resultados consistentes
2. **Memoria**: Auto-prompt puede usar más memoria por la generación de múltiples prompts
3. **Visualizaciones**: Limitadas para evitar problemas de espacio en disco
4. **GPU**: Requiere CUDA para inferencia eficiente

## 🔄 Comparación con Modelos Fine-tuned

Este script proporciona la **baseline** para comparar con:
- SAM2 fine-tuned en subsets (25, 50, 100, 200 imágenes)
- SAM2 fine-tuned en dataset completo
- SAM2 con LoRA
- YOLO + SAM2 pipeline

Los resultados base ayudan a cuantificar la mejora del fine-tuning.

