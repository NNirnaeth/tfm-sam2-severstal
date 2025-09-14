import os
import json
from PIL import Image
from ..utils import decode_bitmap_to_mask
import numpy as np
import cv2

# Define paths
IMG_DIR = "/home/ptp/sam2/datasets/yolo_subsets/images/train"
ANN_DIR = "/home/ptp/sam2/datasets/severstal/train_split/ann"
OUT_LABELS_TRAIN = "/home/ptp/sam2/datasets/yolo_subsets/labels/train"
OUT_LABELS_VAL = "/home/ptp/sam2/datasets/yolo_subsets/labels/val"

os.makedirs(OUT_LABELS_TRAIN, exist_ok=True)
os.makedirs(OUT_LABELS_VAL, exist_ok=True)

# Clase única = 0
CLASS_ID = 0

def mask_to_yolo_bbox(mask):
    """Get YOLO-format bbox from binary mask"""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return x_min, y_min, x_max, y_max

def convert_annotations(split="train"):
    if split == "train":
        label_out = OUT_LABELS_TRAIN
        ann_path = "/home/ptp/sam2/datasets/severstal/train_split/ann"
        img_path = "/home/ptp/sam2/datasets/yolo_subsets/images/train"
    else:
        label_out = OUT_LABELS_VAL
        ann_path = "/home/ptp/sam2/datasets/severstal/test_split/ann"
        img_path = "/home/ptp/sam2/datasets/yolo_subsets/images/val"

    for fname in os.listdir(ann_path):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(ann_path, fname), "r") as f:
            data = json.load(f)

        img_name = fname.replace(".jpg.json", ".jpg")
        image_file = os.path.join(img_path, img_name)
        if not os.path.exists(image_file):
            print(f"Imagen no encontrada: {img_name}")
            continue

        width = data["size"]["width"]
        height = data["size"]["height"]
        boxes = []

        for obj in data.get("objects", []):
            if obj.get("geometryType") != "bitmap":
                continue
            mask = decode_bitmap_to_mask(obj["bitmap"]["data"])
            if mask is None:
                continue
            x0, y0 = obj["bitmap"]["origin"]
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[y0:y0+mask.shape[0], x0:x0+mask.shape[1]] = mask
            bbox = mask_to_yolo_bbox(full_mask)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                w = (x_max - x_min) / width
                h = (y_max - y_min) / height
                boxes.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        label_file = os.path.join(label_out, fname.replace(".jpg.json", ".txt"))
        with open(label_file, "w") as f_out:
            f_out.write("\n".join(boxes))

if __name__ == "__main__":
    convert_annotations("train")
    convert_annotations("val")
    print("Conversión completada.")

