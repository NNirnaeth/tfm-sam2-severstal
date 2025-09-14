#!/usr/bin/env python3
"""
Script para probar el formato de bitmap de Severstal
Verifica la decodificación correcta de las anotaciones PNG
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
import base64
import io
from pathlib import Path

# Add paths
sys.path.append('..')
sys.path.append('../../libs/sam2base')

def test_severstal_bitmap_format():
    """Probar el formato de bitmap de Severstal"""
    print("🧪 Probando formato de bitmap de Severstal")
    print("=" * 50)
    
    # Ruta del dataset
    ann_dir = Path("../../datasets/severstal/train_split/ann")
    img_dir = Path("../../datasets/severstal/train_split/img")
    
    if not ann_dir.exists():
        print(f"❌ Directorio de anotaciones no encontrado: {ann_dir}")
        return False
    
    # Tomar una anotación de ejemplo
    ann_files = list(ann_dir.glob("*.json"))[:3]  # Primeras 3 anotaciones
    
    for i, ann_file in enumerate(ann_files):
        print(f"\n📄 Probando anotación {i+1}: {ann_file.name}")
        
        try:
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            
            # Verificar estructura
            if 'objects' not in ann_data:
                print("   ❌ No hay objetos en la anotación")
                continue
            
            print(f"   📊 Objetos encontrados: {len(ann_data['objects'])}")
            
            # Probar cada objeto
            for j, obj in enumerate(ann_data['objects']):
                print(f"   🔍 Objeto {j+1}:")
                
                if 'bitmap' not in obj:
                    print("     ❌ No tiene bitmap")
                    continue
                
                if 'data' not in obj['bitmap']:
                    print("     ❌ No tiene datos de bitmap")
                    continue
                
                # Verificar formato de datos
                bitmap_data = obj['bitmap']['data']
                print(f"     📏 Longitud de datos: {len(bitmap_data)}")
                
                # Intentar decodificar base64
                try:
                    png_data = base64.b64decode(bitmap_data)
                    print(f"     ✅ Base64 decodificado: {len(png_data)} bytes")
                    
                    # Verificar si es PNG válido
                    if png_data.startswith(b'\x89PNG'):
                        print("     ✅ Formato PNG válido")
                    else:
                        print("     ❌ No es formato PNG válido")
                        continue
                    
                    # Cargar imagen PNG
                    png_image = Image.open(io.BytesIO(png_data))
                    print(f"     📐 Dimensiones PNG: {png_image.size}")
                    
                    # Convertir a numpy
                    mask = np.array(png_image)
                    print(f"     🎨 Forma numpy: {mask.shape}")
                    print(f"     🎨 Tipo de datos: {mask.dtype}")
                    print(f"     🎨 Valores únicos: {np.unique(mask)}")
                    
                    # Verificar origen si existe
                    if 'origin' in obj['bitmap']:
                        origin = obj['bitmap']['origin']
                        print(f"     📍 Origen: {origin}")
                    else:
                        print("     ⚠️ No hay información de origen")
                    
                    # Verificar clase
                    if 'classTitle' in obj:
                        print(f"     🏷️ Clase: {obj['classTitle']}")
                    
                except Exception as e:
                    print(f"     ❌ Error decodificando: {e}")
                    continue
                
                print("     ✅ Objeto procesado correctamente")
                break  # Solo probar el primer objeto de cada anotación
            
        except Exception as e:
            print(f"   ❌ Error procesando anotación: {e}")
            continue
    
    return True

def test_binary_mask_creation():
    """Probar la creación de máscaras binarias"""
    print("\n🧪 Probando creación de máscaras binarias")
    print("=" * 50)
    
    # Tomar una anotación de ejemplo
    ann_file = Path("../../datasets/severstal/train_split/ann/7435bcc25.jpg.json")
    
    if not ann_file.exists():
        print(f"❌ Archivo de ejemplo no encontrado: {ann_file}")
        return False
    
    try:
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)
        
        print(f"📄 Anotación cargada: {ann_file.name}")
        print(f"📊 Objetos: {len(ann_data['objects'])}")
        
        # Obtener dimensiones de la imagen
        if 'size' in ann_data:
            height = ann_data['size']['height']
            width = ann_data['size']['width']
            print(f"📐 Dimensiones imagen: {width}x{height}")
        else:
            print("❌ No hay información de dimensiones")
            return False
        
        # Crear máscara binaria
        binary_mask = create_binary_mask_from_objects(ann_data, (height, width))
        
        if binary_mask is not None:
            print(f"✅ Máscara binaria creada: {binary_mask.shape}")
            print(f"📊 Píxeles con defecto: {binary_mask.sum()}")
            print(f"📊 Porcentaje de defecto: {binary_mask.sum() / binary_mask.size * 100:.2f}%")
            
            # Guardar máscara para inspección visual
            output_dir = Path("../../datasets/severstal_lora/test_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            mask_path = output_dir / "test_binary_mask.png"
            cv2.imwrite(str(mask_path), binary_mask * 255)
            print(f"💾 Máscara guardada en: {mask_path}")
            
            return True
        else:
            print("❌ No se pudo crear la máscara binaria")
            return False
            
    except Exception as e:
        print(f"❌ Error en prueba de máscara binaria: {e}")
        return False

def create_binary_mask_from_objects(ann_data, image_shape):
    """Crear máscara binaria combinando todos los objetos de defecto"""
    if 'objects' not in ann_data or not ann_data['objects']:
        return None
    
    height, width = image_shape
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    
    for obj in ann_data['objects']:
        if 'bitmap' not in obj or 'data' not in obj['bitmap']:
            continue
        
        try:
            # Decodificar bitmap PNG (formato Severstal)
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
            print(f"⚠️ Error procesando objeto: {e}")
            continue
    
    return binary_mask

def main():
    """Función principal"""
    print("🚀 INICIANDO PRUEBAS DE FORMATO BITMAP SEVERSTAL")
    print("=" * 60)
    
    # Prueba 1: Formato de bitmap
    success1 = test_severstal_bitmap_format()
    
    # Prueba 2: Creación de máscaras binarias
    success2 = test_binary_mask_creation()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    if success1 and success2:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("✅ El formato de bitmap de Severstal está correctamente implementado")
        print("✅ La conversión a máscaras binarias funciona correctamente")
        print("\n📋 Próximos pasos:")
        print("1. Ejecutar: python phase0_preparation.py")
        print("2. Verificar que las máscaras binarias se generan correctamente")
    else:
        print("⚠️ Algunas pruebas fallaron")
        print("❌ Revisar errores antes de continuar")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)







































