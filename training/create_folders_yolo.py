import os
import shutil
import random
from tqdm import tqdm

BASE_DIR = "/home/ptp/sam2/datasets/yolo_subsets"
subsets = ["train_25", "train_50", "train_100", "train_200", "train_500"]

for old_name in subsets:
    old_path = os.path.join(BASE_DIR, old_name)
    new_name = old_name.replace("train_", "sub_")
    new_path = os.path.join(BASE_DIR, new_name)

    # Rename folder
    os.rename(old_path, new_path)

    # Create new folders
    img_path = os.path.join(new_path, "img")
    ann_path = os.path.join(new_path, "ann")
    train_img = os.path.join(new_path, "images/train")
    val_img = os.path.join(new_path, "images/val")
    train_lbl = os.path.join(new_path, "labels/train")
    val_lbl = os.path.join(new_path, "labels/val")

    os.makedirs(train_img, exist_ok=True)
    os.makedirs(val_img, exist_ok=True)
    os.makedirs(train_lbl, exist_ok=True)
    os.makedirs(val_lbl, exist_ok=True)

    images = sorted([f for f in os.listdir(img_path) if f.endswith(".jpg")])
    random.shuffle(images)
    split = int(len(images) * 0.1)

    val_images = images[:split]
    train_images = images[split:]

    print(f"\n{subsets} → {len(train_images)} train, {len(val_images)} val")

    for img in tqdm(train_images, desc="  → Train"):
        base = img.replace(".jpg", "")
        shutil.move(os.path.join(img_path, img), os.path.join(train_img, img))
        shutil.move(os.path.join(ann_path, base + ".jpg.json"), os.path.join(train_lbl, base + ".jpg.json"))

    for img in tqdm(val_images, desc="  → Val"):
        base = img.replace(".jpg", "")
        shutil.move(os.path.join(img_path, img), os.path.join(val_img, img))
        shutil.move(os.path.join(ann_path, base + ".jpg.json"), os.path.join(val_lbl, base + ".jpg.json"))

    # Delete emptu folders
    shutil.rmtree(img_path)
    shutil.rmtree(ann_path)

print("\n Division completed")

