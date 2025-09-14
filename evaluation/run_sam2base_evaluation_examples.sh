#!/bin/bash
# Script de ejemplo para ejecutar evaluaciones de SAM2 base con diferentes modos

echo "=== Ejemplos de Evaluación de SAM2 Base (Sin Fine-tuning) ==="
echo ""

# Cambiar al directorio correcto
cd /home/ptp/sam2/new_src/evaluation

echo "1. Evaluación con 30 puntos del GT (modo original):"
echo "python evaluate_sam2base_large_small.py --evaluation_mode 30_gt_points --model_size both"
echo ""

echo "2. Evaluación con 1-3 puntos del GT (más realista):"
echo "python evaluate_sam2base_large_small.py --evaluation_mode gt_points --num_gt_points 3 --model_size both"
echo ""

echo "3. Evaluación con prompts automáticos (sin GT - más realista):"
echo "python evaluate_sam2base_large_small.py --evaluation_mode auto_prompt --model_size both"
echo ""

echo "4. Evaluación solo del modelo Large con 3 puntos GT:"
echo "python evaluate_sam2base_large_small.py --evaluation_mode gt_points --num_gt_points 3 --model_size large"
echo ""

echo "5. Evaluación solo del modelo Small con prompts automáticos:"
echo "python evaluate_sam2base_large_small.py --evaluation_mode auto_prompt --model_size small"
echo ""

echo "6. Evaluación con menos visualizaciones (más rápido):"
echo "python evaluate_sam2base_large_small.py --evaluation_mode gt_points --num_gt_points 3 --model_size both --num_viz 10"
echo ""

echo "=== Comparación de Modos ==="
echo "Para comparar los tres modos en el mismo modelo:"
echo ""
echo "# Modo 1: 30 puntos GT (máximo rendimiento teórico)"
echo "python evaluate_sam2base_large_small.py --evaluation_mode 30_gt_points --model_size large"
echo ""
echo "# Modo 2: 3 puntos GT (realista con anotación manual)"
echo "python evaluate_sam2base_large_small.py --evaluation_mode gt_points --num_gt_points 3 --model_size large"
echo ""
echo "# Modo 3: Prompts automáticos (completamente automático)"
echo "python evaluate_sam2base_large_small.py --evaluation_mode auto_prompt --model_size large"
echo ""

echo "Los resultados se guardarán en:"
echo "- evaluation_results/sam2_base/sam2base_evaluation_{modo}_{timestamp}/"
echo "- Cada modo tendrá su propio directorio con sufijo descriptivo"
echo ""

echo "=== Notas ==="
echo "- Modo '30_gt_points': Usa 30 puntos del GT (rendimiento máximo teórico)"
echo "- Modo 'gt_points': Usa 1-3 puntos del GT (centroide + punto interior + punto negativo)"
echo "- Modo 'auto_prompt': Usa prompts automáticos sin GT (grid + boxes)"
echo "- Los resultados incluyen métricas completas y visualizaciones"
echo "- Tiempo estimado: Small ~2.5h, Large ~5h por modo"

