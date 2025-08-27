"""
Global configuration for evaluation scripts
Centralizes all paths and settings for consistent results organization
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EVALUATION_ROOT = PROJECT_ROOT / "new_src" / "evaluation"
EVALUATION_RESULTS = EVALUATION_ROOT / "evaluation_results"

# Results directories by category
RESULTS_DIRS = {
    "sam2_base": EVALUATION_RESULTS / "sam2_base",
    "sam2_full_dataset": EVALUATION_RESULTS / "sam2_full_dataset", 
    "sam2_subsets": EVALUATION_RESULTS / "sam2_subsets",
    "sam2_lora": EVALUATION_RESULTS / "sam2_lora",
    "yolo_segmentation": EVALUATION_RESULTS / "yolo_segmentation",
    "unet_models": EVALUATION_RESULTS / "unet_models",
    "yolo_detection": EVALUATION_RESULTS / "yolo_detection",
    "yolo_sam2_pipeline": EVALUATION_RESULTS / "yolo_sam2_pipeline",
}

# Specialized directories
SPECIAL_DIRS = {
    "csv": EVALUATION_RESULTS / "csv",
    "json": EVALUATION_RESULTS / "json", 
    "plots": EVALUATION_RESULTS / "plots",
    "masks": EVALUATION_RESULTS / "masks",
    "logs": EVALUATION_RESULTS / "logs",
}

# Default output directories for each script type
SCRIPT_DEFAULTS = {
    "yolo_sam2": RESULTS_DIRS["yolo_sam2_pipeline"],
    "yolov8_seg": RESULTS_DIRS["yolo_segmentation"],
    "unet": RESULTS_DIRS["unet_models"],
    "lora_sam2": RESULTS_DIRS["sam2_lora"],
    "sam2_subsets": RESULTS_DIRS["sam2_subsets"],
    "sam2_full": RESULTS_DIRS["sam2_full_dataset"],
    "sam2_base": RESULTS_DIRS["sam2_base"],
    "yolo_detect": RESULTS_DIRS["yolo_detection"],
}

def get_results_dir(category: str) -> Path:
    """Get results directory for a specific category"""
    if category in RESULTS_DIRS:
        return RESULTS_DIRS[category]
    elif category in SPECIAL_DIRS:
        return SPECIAL_DIRS[category]
    else:
        return EVALUATION_RESULTS / category

def get_script_default(script_type: str) -> Path:
    """Get default output directory for a specific script type"""
    return SCRIPT_DEFAULTS.get(script_type, EVALUATION_RESULTS)

def ensure_dir_exists(path: Path) -> Path:
    """Ensure directory exists and return path"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_csv_dir() -> Path:
    """Get CSV results directory"""
    return ensure_dir_exists(SPECIAL_DIRS["csv"])

def get_plots_dir() -> Path:
    """Get plots directory"""
    return ensure_dir_exists(SPECIAL_DIRS["plots"])

def get_json_dir() -> Path:
    """Get JSON results directory"""
    return ensure_dir_exists(SPECIAL_DIRS["json"])

# Legacy path mappings (for backward compatibility)
LEGACY_PATHS = {
    "data/results": RESULTS_DIRS["yolo_segmentation"],
    "runs/unet_evaluation": RESULTS_DIRS["unet_models"],
    "runs/lora_sam2_evaluation": RESULTS_DIRS["sam2_lora"],
    "runs/unet_comprehensive_eval": RESULTS_DIRS["unet_models"],
    "new_src/evaluation/results": EVALUATION_RESULTS,
}

def get_legacy_path_mapping(old_path: str) -> Path:
    """Map legacy paths to new unified structure"""
    return LEGACY_PATHS.get(old_path, EVALUATION_RESULTS)

# Print configuration summary
if __name__ == "__main__":
    print("Evaluation Configuration:")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Evaluation Root: {EVALUATION_ROOT}")
    print(f"Results Root: {EVALUATION_RESULTS}")
    print("\nCategory Directories:")
    for category, path in RESULTS_DIRS.items():
        print(f"  {category}: {path}")
    print("\nSpecial Directories:")
    for special, path in SPECIAL_DIRS.items():
        print(f"  {special}: {path}")

