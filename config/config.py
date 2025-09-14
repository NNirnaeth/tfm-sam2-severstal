#!/usr/bin/env python3
"""
Configuración centralizada para SAM2 + LoRA en Severstal
TFM: Segmentación binaria de defectos metálicos
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

class LoRAConfig:
    """Configuración principal para el proyecto SAM2 + LoRA"""
    
    def __init__(self, base_dir: str = "../../"):
        self.base_dir = Path(base_dir)
        
        # Directorios del proyecto
        self.project_dirs = {
            "base": self.base_dir,
            "datasets": self.base_dir / "datasets",
            "severstal": self.base_dir / "datasets/severstal",
            "severstal_lora": self.base_dir / "datasets/severstal_lora",
            "models": self.base_dir / "models",
            "configs": self.base_dir / "configs",
            "results": self.base_dir / "results",
            "logs": self.base_dir / "logs"
        }
        
        # Configuración del dataset
        self.dataset_config = {
            "name": "severstal_lora_binary",
            "original_classes": 4,  # Clases originales de defectos
            "target_classes": 1,    # Clase binaria (defecto vs no defecto)
            "val_split_ratio": 0.1,  # 10% para validación interna
            "test_images": 2000,    # Imágenes de test separadas
            "image_formats": [".jpg", ".png"],
            "annotation_format": "json"
        }
        
        # Configuración de prompts
        self.prompt_config = {
            "points_per_mask": (1, 5),      # Rango de puntos positivos
            "negative_points": (1, 3),      # Puntos negativos
            "box_prompt_ratio": 0.3,        # 30% de prompts de caja
            "min_defect_size": 10,          # Tamaño mínimo para generar prompts
            "padding": 5,                   # Padding para cajas
            "max_attempts": 200             # Intentos máximos para puntos negativos
        }
        
        # Configuración de augmentations
        self.augmentation_config = {
            "train": {
                "random_crop": (0.6, 1.0),
                "flip_h": 0.5,
                "flip_v": 0.3,
                "rotation": (-10, 10),
                "brightness": (-0.2, 0.2),
                "contrast": (-0.2, 0.2),
                "blur_prob": 0.1,
                "mixup_prob": 0.2,
                "cutmix_prob": 0.1,
                "random_erasing": 0.1
            },
            "val": {
                "random_crop": None,
                "flip_h": 0.0,
                "flip_v": 0.0,
                "rotation": (0, 0),
                "brightness": (0, 0),
                "contrast": (0, 0),
                "blur_prob": 0.0,
                "mixup_prob": 0.0,
                "cutmix_prob": 0.0,
                "random_erasing": 0.0
            }
        }
        
        # Configuración del modelo
        self.model_config = {
            "base_model": "sam2_hiera_large",
            "config_path": str(self.base_dir / "configs/sam2/sam2_hiera_l.yaml"),
            "checkpoint_path": str(self.base_dir / "models/sam2_base_models/sam2_hiera_large.pt"),
            "target_size": (1024, 1024),  # Ajustar según GPU
            "freeze_image_encoder": True,
            "freeze_prompt_encoder": True,
            "freeze_mask_decoder": False
        }
        
        # Configuración de LoRA
        self.lora_config = {
            "r": 8,                    # Rank de LoRA
            "alpha": 16,               # Factor de escalado
            "dropout": 0.05,           # Dropout en LoRA
            "bias": "none",            # Configuración de bias
            "target_modules": [        # Módulos objetivo para LoRA
                "q_proj", "k_proj", "v_proj", "out_proj",
                "mlp.w1", "mlp.w2", "mlp.w3"
            ],
            "modules_to_save": [       # Módulos a guardar completamente
                "mask_decoder"
            ]
        }
        
        # Configuración de entrenamiento
        self.training_config = {
            "optimizer": {
                "type": "AdamW",
                "learning_rate": 2e-4,
                "weight_decay": 0.05,
                "betas": (0.9, 0.999),
                "eps": 1e-8
            },
            "scheduler": {
                "type": "cosine",
                "warmup_steps": 1000,
                "warmup_ratio": 0.1,
                "min_lr_ratio": 0.01
            },
            "training": {
                "batch_size": 8,
                "accumulation_steps": 2,
                "total_steps": 32000,
                "eval_steps": 500,
                "save_steps": 500,
                "log_steps": 100,
                "early_stopping_patience": 3,
                "early_stopping_min_delta": 0.001
            },
            "loss": {
                "dice_weight": 1.0,
                "focal_weight": 1.0,
                "focal_gamma": 2.0,
                "focal_alpha": 0.25,
                "iou_weight": 0.5
            },
            "regularization": {
                "grad_clip_norm": 1.0,
                "label_smoothing": 0.05,
                "drop_path": 0.1
            }
        }
        
        # Configuración de evaluación
        self.evaluation_config = {
            "metrics": ["iou", "precision", "recall", "f1", "dice"],
            "iou_thresholds": [0.5, 0.75, 0.9],
            "prompt_strategies": ["points", "boxes", "mixed"],
            "num_prompts_per_image": 5,
            "confidence_thresholds": [0.5, 0.7, 0.9]
        }
        
        # Configuración de pseudo-labels (Fase 2)
        self.pseudo_label_config = {
            "confidence_threshold": 0.85,
            "tta_consistency_threshold": 0.8,
            "max_instances_per_image": 8,
            "pseudo_weight": 0.3,
            "consistency_weight": 0.1,
            "curriculum_steps": [5000, 10000],
            "threshold_reduction": [0.85, 0.8, 0.75]
        }
        
        # Configuración de logging
        self.logging_config = {
            "log_dir": str(self.base_dir / "logs/lora_training"),
            "tensorboard": True,
            "wandb": False,  # Cambiar a True si usas Weights & Biases
            "save_visualizations": True,
            "save_checkpoints": True,
            "log_gradients": False
        }
        
        # Configuración de hardware
        self.hardware_config = {
            "device": "cuda",
            "mixed_precision": True,
            "num_workers": 4,
            "pin_memory": True,
            "deterministic": True,
            "seed": 42
        }
    
    def get_dataset_paths(self) -> Dict[str, str]:
        """Obtener rutas del dataset"""
        return {
            "train_images": str(self.project_dirs["severstal_lora"] / "train" / "img"),
            "train_annotations": str(self.project_dirs["severstal_lora"] / "train" / "ann"),
            "val_images": str(self.project_dirs["severstal_lora"] / "val" / "img"),
            "val_annotations": str(self.project_dirs["severstal_lora"] / "val" / "ann"),
            "test_images": str(self.project_dirs["severstal"] / "test_split" / "img"),
            "test_annotations": str(self.project_dirs["severstal"] / "test_split" / "ann")
        }
    
    def get_model_paths(self) -> Dict[str, str]:
        """Obtener rutas del modelo"""
        return {
            "base_model": self.model_config["checkpoint_path"],
            "config_file": self.model_config["config_path"],
            "output_dir": str(self.project_dirs["models"] / "severstal_lora"),
            "checkpoint_dir": str(self.project_dirs["models"] / "severstal_lora" / "checkpoints")
        }
    
    def get_training_paths(self) -> Dict[str, str]:
        """Obtener rutas para entrenamiento"""
        return {
            "log_dir": self.logging_config["log_dir"],
            "checkpoint_dir": str(self.project_dirs["models"] / "severstal_lora" / "checkpoints"),
            "results_dir": str(self.project_dirs["results"] / "lora_results"),
            "visualizations_dir": str(self.project_dirs["results"] / "lora_visualizations")
        }
    
    def validate_paths(self) -> bool:
        """Validar que todas las rutas necesarias existen"""
        required_paths = [
            self.project_dirs["severstal"],
            self.project_dirs["configs"],
            self.model_config["checkpoint_path"],
            self.model_config["config_path"]
        ]
        
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(str(path))
        
        if missing_paths:
            print("❌ Rutas faltantes:")
            for path in missing_paths:
                print(f"   - {path}")
            return False
        
        print("✅ Todas las rutas necesarias existen")
        return True
    
    def create_directories(self) -> None:
        """Crear directorios necesarios para el proyecto"""
        directories = [
            self.project_dirs["severstal_lora"] / "train" / "img",
            self.project_dirs["severstal_lora"] / "train" / "ann",
            self.project_dirs["severstal_lora"] / "val" / "img",
            self.project_dirs["severstal_lora"] / "val" / "ann",
            self.project_dirs["severstal_lora"] / "configs",
            self.project_dirs["severstal_lora"] / "logs",
            self.project_dirs["severstal_lora"] / "visualizations",
            Path(self.get_training_paths()["checkpoint_dir"]),
            Path(self.get_training_paths()["log_dir"]),
            Path(self.get_training_paths()["results_dir"]),
            Path(self.get_training_paths()["visualizations_dir"])
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directorios del proyecto creados")
    
    def save_config(self, output_path: str = None) -> str:
        """Guardar configuración completa en archivo JSON"""
        if output_path is None:
            output_path = str(self.project_dirs["severstal_lora"] / "configs" / "complete_config.json")
        
        config_dict = {
            "project_dirs": {k: str(v) for k, v in self.project_dirs.items()},
            "dataset_config": self.dataset_config,
            "prompt_config": self.prompt_config,
            "augmentation_config": self.augmentation_config,
            "model_config": self.model_config,
            "lora_config": self.lora_config,
            "training_config": self.training_config,
            "evaluation_config": self.evaluation_config,
            "pseudo_label_config": self.pseudo_label_config,
            "logging_config": self.logging_config,
            "hardware_config": self.hardware_config
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Configuración guardada en: {output_path}")
        return output_path
    
    def load_config(self, config_path: str) -> None:
        """Cargar configuración desde archivo JSON"""
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Actualizar configuración
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        print(f"✅ Configuración cargada desde: {config_path}")
    
    def print_summary(self) -> None:
        """Imprimir resumen de la configuración"""
        print("\n" + "=" * 60)
        print("📋 RESUMEN DE CONFIGURACIÓN SAM2 + LoRA")
        print("=" * 60)
        
        print(f"🎯 Objetivo: Segmentación binaria de defectos metálicos")
        print(f"📊 Dataset: {self.dataset_config['name']}")
        print(f"🔧 Modelo base: {self.model_config['base_model']}")
        print(f"🎨 LoRA rank: {self.lora_config['r']}")
        print(f"📚 Pasos totales: {self.training_config['training']['total_steps']}")
        print(f"💾 Batch size efectivo: {self.training_config['training']['batch_size'] * self.training_config['training']['accumulation_steps']}")
        print(f"🎓 Learning rate: {self.training_config['optimizer']['learning_rate']}")
        
        print(f"\n📁 Directorios principales:")
        print(f"   - Dataset: {self.project_dirs['severstal_lora']}")
        print(f"   - Modelos: {self.get_model_paths()['output_dir']}")
        print(f"   - Logs: {self.logging_config['log_dir']}")
        
        print(f"\n⚙️ Configuración de prompts:")
        print(f"   - Puntos por máscara: {self.prompt_config['points_per_mask']}")
        print(f"   - Ratio cajas: {self.prompt_config['box_prompt_ratio']}")
        print(f"   - Tamaño mínimo defecto: {self.prompt_config['min_defect_size']}")
        
        print("=" * 60)

def get_default_config() -> LoRAConfig:
    """Obtener configuración por defecto"""
    return LoRAConfig()

def create_config_from_args(args) -> LoRAConfig:
    """Crear configuración personalizada desde argumentos de línea de comandos"""
    config = LoRAConfig()
    
    # Personalizar configuración según argumentos
    if hasattr(args, 'learning_rate'):
        config.training_config['optimizer']['learning_rate'] = args.learning_rate
    
    if hasattr(args, 'batch_size'):
        config.training_config['training']['batch_size'] = args.batch_size
    
    if hasattr(args, 'total_steps'):
        config.training_config['training']['total_steps'] = args.total_steps
    
    if hasattr(args, 'lora_rank'):
        config.lora_config['r'] = args.lora_rank
    
    if hasattr(args, 'target_size'):
        config.model_config['target_size'] = args.target_size
    
    return config

if __name__ == "__main__":
    # Ejemplo de uso
    config = LoRAConfig()
    config.print_summary()
    
    # Validar rutas
    if config.validate_paths():
        # Crear directorios
        config.create_directories()
        
        # Guardar configuración
        config.save_config()
    else:
        print("❌ No se pueden crear directorios - rutas faltantes")







































