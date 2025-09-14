#!/usr/bin/env python3
"""
Utilidades para SAM2 + LoRA en Severstal
Funciones auxiliares para preparación de datos y conversión de formatos
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import base64
import zlib
import io
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

def decode_bitmap_to_mask(bitmap_data: str, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Decodificar bitmap PNG comprimido a máscara numpy
    
    Args:
        bitmap_data: String base64 del bitmap comprimido
        target_shape: Tupla (height, width) del tamaño objetivo
    
    Returns:
        Máscara numpy de shape target_shape
    """
    try:
        # Decodificar base64
        compressed_data = base64.b64decode(bitmap_data)
        
        # Descomprimir zlib
        decompressed_data = zlib.decompress(compressed_data)
        
        # Cargar PNG desde bytes
        png_image = Image.open(io.BytesIO(decompressed_data))
        mask = np.array(png_image)
        
        # Convertir a binario
        binary_mask = mask > 0
        
        # Redimensionar si es necesario
        if binary_mask.shape != target_shape:
            mask_pil = Image.fromarray(binary_mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            binary_mask = np.array(mask_pil) > 0
        
        return binary_mask.astype(np.uint8)
        
    except Exception as e:
        print(f"Error decodificando bitmap: {e}")
        return np.zeros(target_shape, dtype=np.uint8)

def create_binary_mask_from_objects(ann_data: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Crear máscara binaria combinando todos los objetos de defecto
    
    Args:
        ann_data: Datos de anotación JSON
        image_shape: Tupla (height, width) de la imagen
    
    Returns:
        Máscara binaria combinada
    """
    if 'objects' not in ann_data or not ann_data['objects']:
        return np.zeros(image_shape, dtype=np.uint8)
    
    height, width = image_shape
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    
    for obj in ann_data['objects']:
        if 'bitmap' not in obj or 'data' not in obj['bitmap']:
            continue
        
        try:
            obj_mask = decode_bitmap_to_mask(obj['bitmap']['data'], (height, width))
            binary_mask = np.logical_or(binary_mask, obj_mask)
        except Exception as e:
            print(f"Error procesando objeto: {e}")
            continue
    
    return binary_mask.astype(np.uint8)

def generate_point_prompts(binary_mask: np.ndarray, 
                          num_positive: int = 3, 
                          num_negative: int = 2,
                          min_defect_size: int = 10) -> Dict:
    """
    Generar prompts de puntos para entrenamiento
    
    Args:
        binary_mask: Máscara binaria del defecto
        num_positive: Número de puntos positivos
        num_negative: Número de puntos negativos
        min_defect_size: Tamaño mínimo del defecto para generar prompts
    
    Returns:
        Diccionario con puntos y etiquetas
    """
    defect_coords = np.where(binary_mask > 0)
    
    if len(defect_coords[0]) < min_defect_size:
        return None
    
    prompts = {
        "points": [],
        "point_labels": []
    }
    
    # Puntos positivos dentro del defecto
    if len(defect_coords[0]) > 0:
        positive_indices = np.random.choice(
            len(defect_coords[0]), 
            min(num_positive, len(defect_coords[0])), 
            replace=False
        )
        
        for idx in positive_indices:
            y, x = defect_coords[0][idx], defect_coords[1][idx]
            prompts["points"].append([x, y])
            prompts["point_labels"].append(1)
    
    # Puntos negativos fuera del defecto
    height, width = binary_mask.shape
    negative_points = 0
    attempts = 0
    max_attempts = 200
    
    while negative_points < num_negative and attempts < max_attempts:
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        if binary_mask[y, x] == 0:  # Fuera del defecto
            prompts["points"].append([x, y])
            prompts["point_labels"].append(0)
            negative_points += 1
        
        attempts += 1
    
    return prompts if prompts["points"] else None

def generate_box_prompts(binary_mask: np.ndarray, 
                        padding: int = 5,
                        min_defect_size: int = 20) -> List[List[int]]:
    """
    Generar prompts de caja para entrenamiento
    
    Args:
        binary_mask: Máscara binaria del defecto
        padding: Padding adicional alrededor de la caja
        min_defect_size: Tamaño mínimo del defecto para generar caja
    
    Returns:
        Lista de cajas [x_min, y_min, x_max, y_max]
    """
    defect_coords = np.where(binary_mask > 0)
    
    if len(defect_coords[0]) < min_defect_size:
        return []
    
    height, width = binary_mask.shape
    
    # Calcular límites del defecto
    y_min, y_max = defect_coords[0].min(), defect_coords[0].max()
    x_min, x_max = defect_coords[1].min(), defect_coords[1].max()
    
    # Añadir padding
    y_min = max(0, y_min - padding)
    y_max = min(height - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(width - 1, x_max + padding)
    
    return [[x_min, y_min, x_max, y_max]]

def generate_mixed_prompts(binary_mask: np.ndarray,
                          point_ratio: float = 0.7,
                          min_defect_size: int = 10) -> Dict:
    """
    Generar prompts mixtos (puntos + cajas) para entrenamiento
    
    Args:
        binary_mask: Máscara binaria del defecto
        point_ratio: Proporción de prompts de puntos vs cajas
        min_defect_size: Tamaño mínimo del defecto
    
    Returns:
        Diccionario con prompts mixtos
    """
    if np.random.random() < point_ratio:
        # Generar prompts de puntos
        prompts = generate_point_prompts(binary_mask, min_defect_size=min_defect_size)
        if prompts:
            prompts["boxes"] = []
            return prompts
    else:
        # Generar prompts de caja
        boxes = generate_box_prompts(binary_mask, min_defect_size=min_defect_size)
        if boxes:
            return {
                "points": [],
                "point_labels": [],
                "boxes": boxes
            }
    
    # Fallback a puntos si no se pueden generar cajas
    return generate_point_prompts(binary_mask, min_defect_size=min_defect_size)

def resize_image_with_padding(image: np.ndarray, 
                            target_size: Tuple[int, int],
                            fill_value: int = 0) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Redimensionar imagen manteniendo aspect ratio con padding
    
    Args:
        image: Imagen numpy (H, W, C)
        target_size: Tupla (height, width) objetivo
        fill_value: Valor para el padding
    
    Returns:
        Tupla (imagen redimensionada, escala aplicada)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calcular escala manteniendo aspect ratio
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Redimensionar imagen
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Crear imagen con padding
    padded = np.full((target_h, target_w, image.shape[2]), fill_value, dtype=image.dtype)
    
    # Calcular posición para centrar
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Colocar imagen redimensionada
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded, scale

def apply_augmentations(image: np.ndarray, 
                       mask: np.ndarray,
                       config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplicar augmentations a imagen y máscara
    
    Args:
        image: Imagen numpy
        mask: Máscara numpy
        config: Configuración de augmentations
    
    Returns:
        Tupla (imagen aumentada, máscara aumentada)
    """
    aug_image = image.copy()
    aug_mask = mask.copy()
    
    # Flip horizontal
    if np.random.random() < config.get("flip_h", 0.5):
        aug_image = cv2.flip(aug_image, 1)
        aug_mask = cv2.flip(aug_mask, 1)
    
    # Flip vertical
    if np.random.random() < config.get("flip_v", 0.3):
        aug_image = cv2.flip(aug_image, 0)
        aug_mask = cv2.flip(aug_mask, 0)
    
    # Rotación
    if "rotation" in config:
        angle_range = config["rotation"]
        angle = np.random.uniform(angle_range[0], angle_range[1])
        
        # Calcular matriz de rotación
        h, w = aug_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Aplicar rotación
        aug_image = cv2.warpAffine(aug_image, rotation_matrix, (w, h))
        aug_mask = cv2.warpAffine(aug_mask, rotation_matrix, (w, h))
    
    # Ajuste de brillo y contraste
    if "brightness" in config:
        brightness_factor = np.random.uniform(
            1 + config["brightness"][0], 
            1 + config["brightness"][1]
        )
        aug_image = np.clip(aug_image * brightness_factor, 0, 255).astype(np.uint8)
    
    if "contrast" in config:
        contrast_factor = np.random.uniform(
            1 + config["contrast"][0], 
            1 + config["contrast"][1]
        )
        mean = np.mean(aug_image)
        aug_image = np.clip((aug_image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
    
    # Blur gaussiano
    if np.random.random() < config.get("blur_prob", 0.1):
        kernel_size = np.random.choice([3, 5])
        aug_image = cv2.GaussianBlur(aug_image, (kernel_size, kernel_size), 0)
    
    return aug_image, aug_mask

def compute_metrics(pred_mask: np.ndarray, 
                   gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Calcular métricas de evaluación
    
    Args:
        pred_mask: Máscara predicha
        gt_mask: Máscara ground truth
    
    Returns:
        Diccionario con métricas
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union
    
    # Precisión y Recall
    tp = intersection
    fp = pred_mask.sum() - intersection
    fn = gt_mask.sum() - intersection
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Dice coefficient
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
    
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice
    }

def save_annotation_with_prompts(image_path: str,
                                binary_mask: np.ndarray,
                                prompts: Dict,
                                output_path: str,
                                metadata: Optional[Dict] = None) -> bool:
    """
    Guardar anotación con prompts en formato JSON
    
    Args:
        image_path: Ruta de la imagen original
        binary_mask: Máscara binaria
        prompts: Diccionario con prompts
        output_path: Ruta de salida para la anotación
        metadata: Metadatos adicionales
    
    Returns:
        True si se guardó exitosamente
    """
    try:
        annotation = {
            "image": Path(image_path).name,
            "binary_mask": binary_mask.tolist(),
            "prompts": prompts,
            "image_size": binary_mask.shape,
            "timestamp": str(Path(image_path).stat().st_mtime)
        }
        
        if metadata:
            annotation.update(metadata)
        
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error guardando anotación: {e}")
        return False

def validate_dataset_structure(dataset_dir: str) -> bool:
    """
    Validar estructura del dataset preparado
    
    Args:
        dataset_dir: Directorio del dataset
    
    Returns:
        True si la estructura es válida
    """
    required_dirs = ["train/img", "train/ann", "val/img", "val/ann"]
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Directorio requerido no encontrado: {full_path}")
            return False
    
    # Verificar que hay archivos en train
    train_img_dir = os.path.join(dataset_dir, "train/img")
    train_ann_dir = os.path.join(dataset_dir, "train/ann")
    
    train_images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))]
    train_annotations = [f for f in os.listdir(train_ann_dir) if f.endswith('.json')]
    
    if len(train_images) == 0:
        print("❌ No se encontraron imágenes en train/img")
        return False
    
    if len(train_annotations) == 0:
        print("❌ No se encontraron anotaciones en train/ann")
        return False
    
    print(f"✅ Dataset válido: {len(train_images)} imágenes, {len(train_annotations)} anotaciones")
    return True







































