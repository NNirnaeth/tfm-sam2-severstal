#!/usr/bin/env python3
"""
Fase 0 - Preparación para SAM2 + LoRA + Pseudo-labels
TFM: Segmentación binaria de defectos metálicos en Severstal

Funcionalidades:
1. Split de datos (10% validación interna estratificada)
2. Conversión a máscaras binarias (4 clases -> 1 clase)
3. Generación de prompts mixtos (puntos + cajas)
4. Preparación de dataset con augmentations
5. Configuración de entrenamiento
"""

import os
import sys
import json
import shutil
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add paths
sys.path.append('..')
sys.path.append('../../libs/sam2base')

class SeverstalLoRAPreparation:
    def __init__(self, base_dir="../../"):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "datasets/severstal"
        self.output_dir = self.base_dir / "datasets/severstal_lora"
        
        # Configuración de la Fase 0
        self.val_split_ratio = 0.1  # 10% para validación interna
        self.target_size = (1024, 1024)  # Tamaño objetivo (ajustar según GPU)
        self.seed = 42
        
        # Configuración de prompts
        self.points_per_mask = (1, 5)  # Rango de puntos por máscara
        self.negative_points = (1, 3)  # Puntos negativos fuera de la máscara
        self.box_prompt_ratio = 0.3  # 30% de prompts de caja
        
        # Configuración de augmentations
        self.augmentation_config = {
            "random_crop": (0.6, 1.0),
            "flip_h": 0.5,
            "flip_v": 0.3,
            "rotation": (-10, 10),
            "brightness": (-0.2, 0.2),
            "contrast": (-0.2, 0.2),
            "blur_prob": 0.1,
            "mixup_prob": 0.2,
            "cutmix_prob": 0.1
        }
        
        # Crear directorios de salida
        self._create_output_directories()
        
        # Fijar seed para reproducibilidad
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def _create_output_directories(self):
        """Crear estructura de directorios para LoRA"""
        directories = [
            self.output_dir / "train" / "img",
            self.output_dir / "train" / "ann",
            self.output_dir / "val" / "img", 
            self.output_dir / "val" / "ann",
            self.output_dir / "configs",
            self.output_dir / "logs",
            self.output_dir / "visualizations"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f" Directorios creados en: {self.output_dir}")
    
    def prepare_stratified_split(self):
        """Preparar split estratificado por clase de defecto"""
        print("\n Preparando split estratificado...")
        
        # Cargar todas las anotaciones
        ann_dir = self.dataset_dir / "train_split" / "ann"
        img_dir = self.dataset_dir / "train_split" / "img"
        
        annotations = []
        for ann_file in ann_dir.glob("*.json"):
            img_file = ann_file.stem + ".jpg"
            img_path = img_dir / img_file
            
            if img_path.exists():
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
                    annotations.append({
                        'image': str(img_path),
                        'annotation': str(ann_file),
                        'defect_classes': self._count_defect_classes(ann_data)
                    })
        
        print(f" Total de imágenes: {len(annotations)}")
        
        # Split estratificado por número de clases de defecto
        defect_counts = [ann['defect_classes'] for ann in annotations]
        train_data, val_data = train_test_split(
            annotations, 
            test_size=self.val_split_ratio,
            stratify=defect_counts,
            random_state=self.seed
        )
        
        print(f" Train: {len(train_data)} imágenes")
        print(f" Val: {len(val_data)} imágenes")
        
        return train_data, val_data
    
    def _count_defect_classes(self, ann_data):
        """Contar número de clases de defecto en una anotación"""
        if 'objects' not in ann_data:
            return 0
        
        defect_types = set()
        for obj in ann_data['objects']:
            if 'classTitle' in obj:
                defect_types.add(obj['classTitle'])
        
        return len(defect_types)
    
    def convert_to_binary_masks(self, data_split, split_name):
        """Convertir anotaciones de 4 clases a máscaras binarias"""
        print(f"\n Convirtiendo {split_name} a máscaras binarias...")
        
        output_img_dir = self.output_dir / split_name / "img"
        output_ann_dir = self.output_dir / split_name / "ann"
        
        processed_count = 0
        
        for item in tqdm(data_split, desc=f"Procesando {split_name}"):
            try:
                # Cargar imagen
                image = cv2.imread(item['image'])
                if image is None:
                    continue
                
                # Cargar anotación
                with open(item['annotation'], 'r') as f:
                    ann_data = json.load(f)
                
                # Generar máscara binaria
                binary_mask = self._create_binary_mask(ann_data, image.shape[:2])
                
                if binary_mask is None or not binary_mask.any():
                    continue
                
                # Generar prompts mixtos
                prompts = self._generate_mixed_prompts(binary_mask)
                
                if prompts is None:
                    continue
                
                # Guardar imagen procesada
                img_filename = Path(item['image']).name
                img_output_path = output_img_dir / img_filename
                cv2.imwrite(str(img_output_path), image)
                
                # Guardar anotación binaria con prompts
                ann_filename = Path(item['annotation']).stem + "_binary.json"
                ann_output_path = output_ann_dir / ann_filename
                
                binary_annotation = {
                    "image": img_filename,
                    "binary_mask": binary_mask.tolist(),
                    "prompts": prompts,
                    "original_annotation": str(item['annotation']),
                    "image_size": image.shape[:2]
                }
                
                with open(ann_output_path, 'w') as f:
                    json.dump(binary_annotation, f, indent=2)
                
                processed_count += 1
                
            except Exception as e:
                print(f" Error procesando {item['image']}: {e}")
                continue
        
        print(f" {split_name}: {processed_count} imágenes procesadas")
        return processed_count
    
    def _create_binary_mask(self, ann_data, image_shape):
        """Crear máscara binaria combinando todas las clases de defecto"""
        if 'objects' not in ann_data or not ann_data['objects']:
            return None
        
        height, width = image_shape
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        
        for obj in ann_data['objects']:
            if 'bitmap' not in obj or 'data' not in obj['bitmap']:
                continue
            
            try:
                # Decodificar bitmap PNG comprimido (formato Severstal)
                import base64
                import io
                
                # El formato Severstal usa base64 directo, no zlib
                png_data = base64.b64decode(obj['bitmap']['data'])
                
                # Cargar PNG desde bytes
                png_image = Image.open(io.BytesIO(png_data))
                obj_mask = np.array(png_image)
                
                # Convertir a binario
                obj_mask = obj_mask > 0
                
                # Obtener origen del bitmap (posición en la imagen)
                origin = obj['bitmap'].get('origin', [0, 0])
                origin_x, origin_y = origin[0], origin[1]
                
                # Calcular dimensiones del bitmap
                mask_height, mask_width = obj_mask.shape
                
                # Colocar el bitmap en la posición correcta
                end_y = min(origin_y + mask_height, height)
                end_x = min(origin_x + mask_width, width)
                
                # Asegurar que no exceda los límites
                start_y = max(0, origin_y)
                start_x = max(0, origin_x)
                
                # Ajustar índices del bitmap si es necesario
                bitmap_start_y = max(0, -origin_y)
                bitmap_start_x = max(0, -origin_x)
                
                # Colocar el bitmap en la máscara principal
                binary_mask[start_y:end_y, start_x:end_x] = np.logical_or(
                    binary_mask[start_y:end_y, start_x:end_x],
                    obj_mask[bitmap_start_y:bitmap_start_y + (end_y - start_y), 
                            bitmap_start_x:bitmap_start_x + (end_x - start_x)]
                )
                
            except Exception as e:
                print(f" Error decodificando bitmap: {e}")
                continue
        
        return binary_mask.astype(np.uint8)
    
    def _generate_mixed_prompts(self, binary_mask):
        """Generar prompts mixtos (puntos + cajas) para entrenamiento"""
        # Encontrar coordenadas de la máscara
        defect_coords = np.where(binary_mask > 0)
        
        if len(defect_coords[0]) == 0:
            return None
        
        prompts = {
            "points": [],
            "point_labels": [],
            "boxes": []
        }
        
        # Generar puntos positivos dentro de la máscara
        num_positive = np.random.randint(*self.points_per_mask)
        positive_indices = np.random.choice(
            len(defect_coords[0]), 
            min(num_positive, len(defect_coords[0])), 
            replace=False
        )
        
        for idx in positive_indices:
            y, x = defect_coords[0][idx], defect_coords[1][idx]
            prompts["points"].append([x, y])
            prompts["point_labels"].append(1)  # Punto positivo
        
        # Generar puntos negativos fuera de la máscara
        num_negative = np.random.randint(*self.negative_points)
        height, width = binary_mask.shape
        
        negative_points = 0
        attempts = 0
        max_attempts = 100
        
        while negative_points < num_negative and attempts < max_attempts:
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            if binary_mask[y, x] == 0:  # Fuera de la máscara
                prompts["points"].append([x, y])
                prompts["point_labels"].append(0)  # Punto negativo
                negative_points += 1
            
            attempts += 1
        
        # Generar caja mínima si hay suficientes puntos positivos
        if len(defect_coords[0]) > 10:  # Solo si hay defecto significativo
            y_min, y_max = defect_coords[0].min(), defect_coords[0].max()
            x_min, x_max = defect_coords[1].min(), defect_coords[1].max()
            
            # Añadir padding pequeño
            padding = 5
            y_min = max(0, y_min - padding)
            y_max = min(height - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(width - 1, x_max + padding)
            
            prompts["boxes"].append([x_min, y_min, x_max, y_max])
        
        return prompts
    
    def create_dataset_config(self):
        """Crear archivo de configuración del dataset"""
        config = {
            "dataset_name": "severstal_lora_binary",
            "train_images": str(self.output_dir / "train" / "img"),
            "train_annotations": str(self.output_dir / "val" / "ann"),
            "val_images": str(self.output_dir / "val" / "img"),
            "val_annotations": str(self.output_dir / "val" / "ann"),
            "test_images": str(self.dataset_dir / "test_split" / "img"),
            "test_annotations": str(self.dataset_dir / "test_split" / "ann"),
            "target_size": self.target_size,
            "prompt_config": {
                "points_per_mask": self.points_per_mask,
                "negative_points": self.negative_points,
                "box_prompt_ratio": self.box_prompt_ratio
            },
            "augmentation_config": self.augmentation_config,
            "seed": self.seed
        }
        
        config_path = self.output_dir / "configs" / "dataset_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f" Configuración guardada en: {config_path}")
        return config
    
    def create_training_config(self):
        """Crear configuración de entrenamiento para LoRA"""
        training_config = {
            "model": {
                "base_model": "sam2_hiera_large",
                "config_path": "../../configs/sam2/sam2_hiera_l.yaml",
                "checkpoint_path": "../../models/sam2_base_models/sam2_hiera_large.pt"
            },
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "out_proj",
                    "mlp.w1", "mlp.w2", "mlp.w3"
                ]
            },
            "training": {
                "learning_rate": 2e-4,
                "weight_decay": 0.05,
                "batch_size": 8,
                "accumulation_steps": 2,
                "total_steps": 32000,
                "warmup_steps": 1000,
                "eval_steps": 500,
                "save_steps": 500,
                "early_stopping_patience": 3
            },
            "loss": {
                "dice_weight": 1.0,
                "focal_weight": 1.0,
                "focal_gamma": 2.0,
                "focal_alpha": 0.25
            },
            "augmentation": {
                "train": True,
                "val": False,
                "mixup_prob": 0.2,
                "cutmix_prob": 0.1
            }
        }
        
        config_path = self.output_dir / "configs" / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        print(f" Configuración de entrenamiento guardada en: {config_path}")
        return training_config
    
    def visualize_sample(self, num_samples=5):
        """Visualizar muestras del dataset preparado"""
        print(f"\n Visualizando {num_samples} muestras...")
        
        train_img_dir = self.output_dir / "train" / "img"
        train_ann_dir = self.output_dir / "train" / "ann"
        
        image_files = list(train_img_dir.glob("*.jpg"))[:num_samples]
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_file in enumerate(image_files):
            # Cargar imagen
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Cargar anotación binaria
            ann_file = train_ann_dir / (img_file.stem + "_binary.json")
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            
            binary_mask = np.array(ann_data["binary_mask"])
            prompts = ann_data["prompts"]
            
            # Imagen original
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Imagen {i+1}")
            axes[i, 0].axis('off')
            
            # Máscara binaria
            axes[i, 1].imshow(binary_mask, cmap='gray')
            axes[i, 1].set_title("Máscara Binaria")
            axes[i, 1].axis('off')
            
            # Imagen con prompts
            axes[i, 2].imshow(image)
            
            # Dibujar puntos
            for j, (x, y) in enumerate(prompts["points"]):
                color = 'red' if prompts["point_labels"][j] == 1 else 'blue'
                axes[i, 2].scatter(x, y, c=color, s=50, alpha=0.8)
            
            # Dibujar cajas
            for box in prompts["boxes"]:
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle(
                    (x_min, y_min), x_max-x_min, y_max-y_min,
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                axes[i, 2].add_patch(rect)
            
            axes[i, 2].set_title("Prompts (Rojo: +, Azul: -, Verde: Caja)")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización
        vis_path = self.output_dir / "visualizations" / "sample_visualization.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Visualización guardada en: {vis_path}")
    
    def run_preparation(self):
        """Ejecutar toda la preparación de la Fase 0"""
        print(" Iniciando Fase 0 - Preparación para SAM2 + LoRA")
        print("=" * 60)
        
        # 1. Split estratificado
        train_data, val_data = self.prepare_stratified_split()
        
        # 2. Conversión a máscaras binarias
        train_count = self.convert_to_binary_masks(train_data, "train")
        val_count = self.convert_to_binary_masks(val_data, "val")
        
        # 3. Crear configuraciones
        dataset_config = self.create_dataset_config()
        training_config = self.create_training_config()
        
        # 4. Visualizar muestras
        self.visualize_sample()
        
        # 5. Resumen final
        print("\n" + "=" * 60)
        print(" FASE 0 COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        print(f" Dataset preparado:")
        print(f"   - Train: {train_count} imágenes")
        print(f"   - Val: {val_count} imágenes")
        print(f"   - Test: 2000 imágenes (separadas)")
        print(f" Directorio de salida: {self.output_dir}")
        print(f" Configuraciones guardadas en: {self.output_dir}/configs/")
        print(f" Visualizaciones en: {self.output_dir}/visualizations/")
        print("\n Próximo paso: Implementar Fase 1 (LoRA training)")
        
        return {
            "train_count": train_count,
            "val_count": val_count,
            "output_dir": str(self.output_dir),
            "dataset_config": dataset_config,
            "training_config": training_config
        }

def main():
    """Función principal para ejecutar la preparación"""
    preparator = SeverstalLoRAPreparation()
    results = preparator.run_preparation()
    
    # Guardar resumen
    summary_path = preparator.output_dir / "preparation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Resumen guardado en: {summary_path}")

if __name__ == "__main__":
    main()