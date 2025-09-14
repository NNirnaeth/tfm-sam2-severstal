# Auto-prompt Comparison Scripts

## Script 1: mask_autoprompt_SAM2BasevsFT.py

Compara SAM2 Base vs Fine-tuned en modo auto-prompt.

### Uso:

```bash
# Comparar 20 imágenes con modelo Large
python mask_autoprompt_SAM2BasevsFT.py --max_images 20 --model_size large

# Comparar 10 imágenes con modelo Small
python mask_autoprompt_SAM2BasevsFT.py --max_images 10 --model_size small

# Especificar checkpoint personalizado para fine-tuned
python mask_autoprompt_SAM2BasevsFT.py \
    --max_images 15 \
    --model_size large \
    --ft_checkpoint /path/to/your/checkpoint.torch

# Cambiar directorio de salida
python mask_autoprompt_SAM2BasevsFT.py \
    --max_images 25 \
    --output_dir /home/ptp/sam2/results/my_comparison
```

### Parámetros:

- `--max_images`: Número máximo de imágenes a visualizar (default: 20)
- `--model_size`: Tamaño del modelo ('small' o 'large', default: 'large')  
- `--base_checkpoint`: Checkpoint del modelo base (None = pretrained)
- `--ft_checkpoint`: Checkpoint del modelo fine-tuned
- `--output_dir`: Directorio donde guardar las visualizaciones
- `--random_seed`: Semilla aleatoria para selección de imágenes (default: 42)

### Salida:

El script genera:
1. **Visualizaciones comparativas** para cada imagen seleccionada
2. **Estadísticas resumidas** de IoU promedio
3. **Análisis TP/FP/FN** para ambos modelos

Cada visualización incluye:
- Imagen original
- Ground truth
- Predicción SAM2 Base
- Predicción SAM2 Fine-tuned  
- Análisis de errores para ambos modelos

### Ejemplo de uso rápido:

```bash
cd /home/ptp/sam2/new_src/experiments
python mask_autoprompt_SAM2BasevsFT.py --max_images 10
```

Esto procesará 10 imágenes aleatorias y guardará los resultados en:
`/home/ptp/sam2/new_src/experiments/visualizations/autoprompt_comparison/`

