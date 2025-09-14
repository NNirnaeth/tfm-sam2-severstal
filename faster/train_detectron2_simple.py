from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import os

# Disable TensorBoard logging to avoid NumPy 2.0 issues
os.environ['DETECTRON2_DISABLE_TENSORBOARD'] = '1'

# paths Severstal dataset
# Images are 256x1600 pixels (height x width)
TR_IMG = "datasets/severstal/splits/full/train2017/"
TR_ANN = "datasets/severstal/splits/full/annotations/instances_train2017.json"
VL_IMG = "datasets/severstal/splits/full/val2017"
VL_ANN = "datasets/severstal/splits/full/annotations/instances_val2017.json"

# Register COCO datasets
register_coco_instances("severstal_train", {}, TR_ANN, TR_IMG)
register_coco_instances("severstal_val",   {}, VL_ANN, VL_IMG)

# Add metadata for visualization - CORREGIDO: Solo 1 clase
MetadataCatalog.get("severstal_train").set(thing_classes=["defect"])
MetadataCatalog.get("severstal_val").set(thing_classes=["defect"])

# Configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

# Dataset configuration
cfg.DATASETS.TRAIN = ("severstal_train",)
cfg.DATASETS.TEST  = ("severstal_val",)

# DataLoader configuration
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2  # Reduced batch size for 256x1600 images

# Solver configuration - adjusted for large images
cfg.SOLVER.BASE_LR = 0.001  # Smaller LR for large images
cfg.SOLVER.MAX_ITER = 1000  # Reduced iterations for testing
cfg.SOLVER.STEPS = (600, 800)  # LR decay steps
cfg.SOLVER.GAMMA = 0.1

# Model configuration - CORREGIDO: Solo 1 clase
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 defect type (no background class in Detectron2)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Input configuration for 256x1600 images
cfg.INPUT.MIN_SIZE_TRAIN = (256, 512, 768)  # Multi-scale training
cfg.INPUT.MAX_SIZE_TRAIN = 1600
cfg.INPUT.MIN_SIZE_TEST = 256
cfg.INPUT.MAX_SIZE_TEST = 1600

# Output directory
cfg.OUTPUT_DIR = "./output_detectron2"

# Disable TensorBoard to avoid NumPy 2.0 issues
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.TEST.EVAL_PERIOD = 0  # Disable evaluation during training

def visualize_dataset(dataset_name, num_samples=5):
    """
    Visualize dataset samples with bounding boxes to verify annotations
    """
    print(f"Visualizing {num_samples} samples from {dataset_name}...")
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    # Randomly sample images
    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
    
    for i, d in enumerate(samples):
        # Load image
        img = cv2.imread(d["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualizer
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        
        # Draw annotations
        vis = visualizer.draw_dataset_dict(d)
        
        # Save visualization
        output_path = f"./output_detectron2/visualization_{dataset_name}_{i}.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR))
        
        print(f"Saved: {output_path}")
        print(f"Image shape: {img.shape}")
        print(f"Number of annotations: {len(d['annotations'])}")
        
        # Print annotation details
        for j, ann in enumerate(d['annotations']):
            bbox = ann['bbox']  # [x, y, w, h]
            category_id = ann['category_id']
            print(f"  Annotation {j}: bbox={bbox}, category={category_id}")
        print("-" * 50)

# Visualize training and validation datasets
print("=" * 60)
print("VISUALIZING DATASET ANNOTATIONS")
print("=" * 60)
visualize_dataset("severstal_train", num_samples=3)
visualize_dataset("severstal_val", num_samples=2)

# Create trainer and train
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

# Simple trainer without early stopping to avoid data loader issues
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(False)

# Start training
try:
    trainer.train()
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nTraining error: {e}")
    print("This might be due to NumPy 2.0 compatibility issues")

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)

