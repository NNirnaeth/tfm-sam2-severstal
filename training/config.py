"""
Global configuration for training scripts
Centralizes all paths and settings for consistent results organization
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_ROOT = PROJECT_ROOT / "new_src" / "training"
TRAINING_RESULTS = TRAINING_ROOT / "training_results"

# Results directories by category
RESULTS_DIRS = {
    "sam2_base": TRAINING_RESULTS / "sam2_base",
    "sam2_full_dataset": TRAINING_RESULTS / "sam2_full_dataset", 
    "sam2_subsets": TRAINING_RESULTS / "sam2_subsets",
    "sam2_lora": TRAINING_RESULTS / "sam2_lora",
    "yolo_segmentation": TRAINING_RESULTS / "yolo_segmentation",
    "unet_models": TRAINING_RESULTS / "unet_models",
    "yolo_detection": TRAINING_RESULTS / "yolo_detection",
}

# Specialized directories
SPECIAL_DIRS = {
    "checkpoints": TRAINING_RESULTS / "checkpoints",
    "logs": TRAINING_RESULTS / "logs", 
    "metrics": TRAINING_RESULTS / "metrics",
    "configs": TRAINING_RESULTS / "configs",
}

# Default output directories for each script type
SCRIPT_DEFAULTS = {
    "yolo_seg": RESULTS_DIRS["yolo_segmentation"],
    "yolo_detect": RESULTS_DIRS["yolo_detection"],
    "unet": RESULTS_DIRS["unet_models"],
    "unetpp": RESULTS_DIRS["unet_models"],
    "dsc_unet": RESULTS_DIRS["unet_models"],
    "lora_sam2": RESULTS_DIRS["sam2_lora"],
    "sam2_subsets": RESULTS_DIRS["sam2_subsets"],
    "sam2_full": RESULTS_DIRS["sam2_full_dataset"],
    "sam2_base": RESULTS_DIRS["sam2_base"],
}

# Legacy path mappings (for backward compatibility)
LEGACY_PATHS = {
    "runs_tfm": RESULTS_DIRS["yolo_segmentation"],
    "models/unet_severstal": RESULTS_DIRS["unet_models"],
    "models/unet++_severstal": RESULTS_DIRS["unet_models"],
    "models/severstal_subsets": RESULTS_DIRS["sam2_subsets"],
    "models/severstal_updated": RESULTS_DIRS["sam2_full_dataset"],
    "models/severstal_lora": RESULTS_DIRS["sam2_lora"],
    "new_src/training/results": TRAINING_RESULTS,
    "new_src/training/models": TRAINING_RESULTS / "unet_models",
}

def get_results_dir(category: str) -> Path:
    """Get results directory for a specific category"""
    if category in RESULTS_DIRS:
        return RESULTS_DIRS[category]
    elif category in SPECIAL_DIRS:
        return SPECIAL_DIRS[category]
    else:
        return TRAINING_RESULTS / category

def get_script_default(script_type: str) -> Path:
    """Get default output directory for a specific script type"""
    return SCRIPT_DEFAULTS.get(script_type, TRAINING_RESULTS)

def ensure_dir_exists(path: Path) -> Path:
    """Ensure directory exists and return path"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_checkpoints_dir() -> Path:
    """Get checkpoints directory"""
    return ensure_dir_exists(SPECIAL_DIRS["checkpoints"])

def get_logs_dir() -> Path:
    """Get logs directory"""
    return ensure_dir_exists(SPECIAL_DIRS["logs"])

def get_metrics_dir() -> Path:
    """Get metrics directory"""
    return ensure_dir_exists(SPECIAL_DIRS["metrics"])

def get_legacy_path_mapping(old_path: str) -> Path:
    """Map legacy paths to new unified structure"""
    return LEGACY_PATHS.get(old_path, TRAINING_RESULTS)

# Print configuration summary
if __name__ == "__main__":
    print("Training Configuration:")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Training Root: {TRAINING_ROOT}")
    print(f"Results Root: {TRAINING_RESULTS}")
    print("\nCategory Directories:")
    for category, path in RESULTS_DIRS.items():
        print(f"  {category}: {path}")
    print("\nSpecial Directories:")
    for special, path in SPECIAL_DIRS.items():
        print(f"  {special}: {path}")
    print("\nLegacy Path Mappings:")
    for legacy, new_path in LEGACY_PATHS.items():
        print(f"  {legacy} → {new_path}")

