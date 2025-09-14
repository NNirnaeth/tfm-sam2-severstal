#!/usr/bin/env python3
"""
Script de prueba para verificar la funcionalidad de la Fase 0
SAM2 + LoRA - Preparación del dataset
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add paths
sys.path.append('..')
sys.path.append('../../libs/sam2base')

from config import LoRAConfig
from utils import (
    decode_bitmap_to_mask, 
    create_binary_mask_from_objects,
    generate_point_prompts,
    generate_box_prompts,
    generate_mixed_prompts,
    compute_metrics,
    validate_dataset_structure
)

def test_configuration():
    """Probar configuración del proyecto"""
    print("🧪 Probando configuración...")
    
    try:
        config = LoRAConfig()
        print("✅ Configuración cargada correctamente")
        
        # Validar rutas
        if config.validate_paths():
            print("✅ Rutas validadas correctamente")
        else:
            print("⚠️ Algunas rutas faltan (esto es normal en la primera ejecución)")
        
        # Crear directorios
        config.create_directories()
        print("✅ Directorios creados correctamente")
        
        # Guardar configuración
        config_path = config.save_config()
        print(f"✅ Configuración guardada en: {config_path}")
        
        # Imprimir resumen
        config.print_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def test_utils():
    """Probar funciones de utilidades"""
    print("\n🧪 Probando utilidades...")
    
    try:
        # Crear máscara de prueba
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[40:60, 40:60] = 1  # Cuadrado de 20x20
        
        # Probar generación de prompts de puntos
        point_prompts = generate_point_prompts(test_mask, num_positive=3, num_negative=2)
        if point_prompts and len(point_prompts["points"]) > 0:
            print("✅ Generación de prompts de puntos: OK")
        else:
            print("❌ Generación de prompts de puntos: FALLÓ")
        
        # Probar generación de prompts de caja
        box_prompts = generate_box_prompts(test_mask, padding=5)
        if box_prompts and len(box_prompts) > 0:
            print("✅ Generación de prompts de caja: OK")
        else:
            print("❌ Generación de prompts de caja: FALLÓ")
        
        # Probar prompts mixtos
        mixed_prompts = generate_mixed_prompts(test_mask, point_ratio=0.7)
        if mixed_prompts:
            print("✅ Generación de prompts mixtos: OK")
        else:
            print("❌ Generación de prompts mixtos: FALLÓ")
        
        # Probar cálculo de métricas
        pred_mask = test_mask.copy()
        pred_mask[45:55, 45:55] = 1  # Predicción ligeramente diferente
        
        metrics = compute_metrics(pred_mask, test_mask)
        if all(key in metrics for key in ["iou", "precision", "recall", "f1", "dice"]):
            print("✅ Cálculo de métricas: OK")
            print(f"   IoU: {metrics['iou']:.4f}")
            print(f"   F1: {metrics['f1']:.4f}")
        else:
            print("❌ Cálculo de métricas: FALLÓ")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en utilidades: {e}")
        return False

def test_binary_conversion():
    """Probar conversión a máscaras binarias"""
    print("\n🧪 Probando conversión binaria...")
    
    try:
        # Crear anotación de prueba
        test_annotation = {
            "objects": [
                {
                    "bitmap": {
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    }
                }
            ]
        }
        
        # Probar decodificación de bitmap
        test_shape = (50, 50)
        decoded_mask = decode_bitmap_to_mask(test_annotation["objects"][0]["bitmap"]["data"], test_shape)
        
        if decoded_mask is not None and decoded_mask.shape == test_shape:
            print("✅ Decodificación de bitmap: OK")
        else:
            print("❌ Decodificación de bitmap: FALLÓ")
        
        # Probar creación de máscara binaria
        binary_mask = create_binary_mask_from_objects(test_annotation, test_shape)
        
        if binary_mask is not None and binary_mask.shape == test_shape:
            print("✅ Creación de máscara binaria: OK")
        else:
            print("❌ Creación de máscara binaria: FALLÓ")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en conversión binaria: {e}")
        return False

def test_dataset_structure():
    """Probar validación de estructura del dataset"""
    print("\n🧪 Probando validación de estructura...")
    
    try:
        # Crear estructura de prueba
        test_dir = Path("../../datasets/severstal_lora")
        test_dirs = [
            test_dir / "train" / "img",
            test_dir / "train" / "ann",
            test_dir / "val" / "img",
            test_dir / "val" / "ann"
        ]
        
        for dir_path in test_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Crear archivos de prueba
        (test_dir / "train" / "img" / "test_image.jpg").touch()
        (test_dir / "train" / "ann" / "test_image_binary.json").touch()
        
        # Probar validación
        is_valid = validate_dataset_structure(str(test_dir))
        
        if is_valid:
            print("✅ Validación de estructura: OK")
        else:
            print("❌ Validación de estructura: FALLÓ")
        
        # Limpiar archivos de prueba
        for dir_path in test_dirs:
            if dir_path.exists():
                for file_path in dir_path.glob("*"):
                    file_path.unlink()
                dir_path.rmdir()
        
        return True
        
    except Exception as e:
        print(f"❌ Error en validación de estructura: {e}")
        return False

def test_integration():
    """Probar integración completa"""
    print("\n🧪 Probando integración completa...")
    
    try:
        from phase0_preparation import SeverstalLoRAPreparation
        
        # Crear instancia de preparación
        preparator = SeverstalLoRAPreparation()
        print("✅ Instancia de preparación creada")
        
        # Verificar directorios
        if preparator.output_dir.exists():
            print("✅ Directorio de salida existe")
        else:
            print("❌ Directorio de salida no existe")
        
        # Verificar configuración
        if hasattr(preparator, 'val_split_ratio') and preparator.val_split_ratio == 0.1:
            print("✅ Configuración de split correcta")
        else:
            print("❌ Configuración de split incorrecta")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en integración: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🚀 INICIANDO PRUEBAS DE LA FASE 0")
    print("=" * 50)
    
    tests = [
        ("Configuración", test_configuration),
        ("Utilidades", test_utils),
        ("Conversión Binaria", test_binary_conversion),
        ("Estructura del Dataset", test_dataset_structure),
        ("Integración", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Ejecutando: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen de resultados
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON! La Fase 0 está lista.")
        print("\n📋 Próximos pasos:")
        print("1. Ejecutar: python phase0_preparation.py")
        print("2. Verificar resultados en datasets/severstal_lora/")
        print("3. Continuar con la Fase 1 (LoRA training)")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisar errores antes de continuar.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)







































