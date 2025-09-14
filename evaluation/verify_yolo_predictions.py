#!/usr/bin/env python3
"""
Verify YOLO predictions by sampling random images and overlaying bounding boxes.
Checks for geometric biases and visualizes detection results.
"""

import os
import random
import argparse
from pathlib import Path
import numpy as np
import cv2

def read_yolo_txt(txt_path):
    """Read YOLO txt lines: cls cx cy w h [conf] -> list of dicts"""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 5: 
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            boxes.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf})
    return boxes

def yolo_to_xyxy(b, W, H):
    """Convert normalized cx,cy,w,h to pixel x1,y1,x2,y2 (clamped)"""
    cx, cy, w, h = b["cx"]*W, b["cy"]*H, b["w"]*W, b["h"]*H
    x1 = max(0, int(round(cx - w/2))); y1 = max(0, int(round(cy - h/2)))
    x2 = min(W-1, int(round(cx + w/2))); y2 = min(H-1, int(round(cy + h/2)))
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
    return x1, y1, x2, y2

def draw_overlay(img, boxes_xyxy, confs, out_path):
    """Draw boxes and confidences on an RGB image and save"""
    vis = img.copy()
    for (x1,y1,x2,y2),c in zip(boxes_xyxy, confs):
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(vis, f"{c:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def pick_image_for_stem(images_dir, stem):
    """Find image file by stem with any common extension"""
    exts = ["*.jpg","*.png","*.jpeg","*.bmp","*.tif","*.tiff"]
    for e in exts:
        cands = list(Path(images_dir).glob(f"{stem}{e[1:]}"))  # e like *.jpg
        if cands: return str(cands[0])
    cands = []
    for e in exts: cands += list(Path(images_dir).glob(e))
    for p in cands:
        if p.stem == stem: return str(p)
    return None

def main():
    ap = argparse.ArgumentParser(description="Verify YOLO predictions with visual overlays")
    ap.add_argument("--images_dir", required=True, help="Path to images/test_split directory")
    ap.add_argument("--labels_dir", required=True, help="Path to predict_test_corrected/labels directory")
    ap.add_argument("--out_dir", default="viz_bboxes_sample", help="Output directory for visualization images")
    ap.add_argument("--n", type=int, default=16, help="Number of random images to sample")
    ap.add_argument("--topk", type=int, default=20, help="Max boxes per image to draw")
    ap.add_argument("--min_conf", type=float, default=0.0, help="Min confidence to draw")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # Collect stems from images_dir
    stems = sorted({p.stem for p in Path(args.images_dir).glob("*.*")})
    sample = random.sample(stems, min(args.n, len(stems)))

    print(f"Sampling {len(sample)} random images from {len(stems)} total images")
    print(f"Images directory: {args.images_dir}")
    print(f"Labels directory: {args.labels_dir}")
    print(f"Output directory: {args.out_dir}")

    all_w, all_h = [], []
    tall_cnt, wide_cnt = 0, 0
    no_detection_cnt = 0
    total_boxes = 0

    for i, stem in enumerate(sample):
        print(f"Processing {i+1}/{len(sample)}: {stem}")
        
        img_path = pick_image_for_stem(args.images_dir, stem)
        if img_path is None:
            print(f"[WARN] image not found for stem {stem}")
            continue
            
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        txt_path = Path(args.labels_dir)/f"{stem}.txt"
        boxes = read_yolo_txt(str(txt_path))
        
        if not boxes:
            # Save "no detections" overlay
            out = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            cv2.putText(out, "NO DETECTIONS", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            cv2.imwrite(str(Path(args.out_dir)/f"{stem}_none.jpg"), out)
            no_detection_cnt += 1
            continue

        # Filter by confidence and keep top-k
        boxes = [b for b in boxes if b["conf"] >= args.min_conf]
        boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)[:args.topk]
        total_boxes += len(boxes)

        # Stats of normalized sizes
        for b in boxes:
            all_w.append(b["w"]); all_h.append(b["h"])
            if b["h"] >= 0.95: tall_cnt += 1
            if b["w"] >= 0.95: wide_cnt += 1

        # Draw
        xyxys = [yolo_to_xyxy(b, W, H) for b in boxes]
        confs = [b["conf"] for b in boxes]
        draw_overlay(img, xyxys, confs, str(Path(args.out_dir)/f"{stem}.jpg"))

    # Print comprehensive report
    print(f"\n=== VERIFICATION SUMMARY ===")
    print(f"Total images processed: {len(sample)}")
    print(f"Images with detections: {len(sample) - no_detection_cnt}")
    print(f"Images without detections: {no_detection_cnt}")
    print(f"Total bounding boxes drawn: {total_boxes}")
    print(f"Average boxes per image: {total_boxes / max(1, len(sample) - no_detection_cnt):.2f}")

    # Print quick bias report
    if all_w and all_h:
        def q(arr, p): return float(np.percentile(arr, p))
        print(f"\n=== GEOMETRY STATISTICS (normalized) ===")
        print(f"Total boxes analyzed: {len(all_w)}")
        print(f"Width stats:  min={min(all_w):.3f}  p50={q(all_w,50):.3f}  p95={q(all_w,95):.3f}  max={max(all_w):.3f}")
        print(f"Height stats: min={min(all_h):.3f}  p50={q(all_h,50):.3f}  p95={q(all_h,95):.3f}  max={max(all_h):.3f}")
        print(f"Tall boxes (h>=0.95): {tall_cnt} ({tall_cnt/len(all_h)*100:.1f}%)")
        print(f"Wide boxes (w>=0.95): {wide_cnt} ({wide_cnt/len(all_w)*100:.1f}%)")
        
        # Check for potential issues
        if tall_cnt > len(all_h) * 0.1:  # More than 10% are very tall
            print(f"⚠️  WARNING: High proportion of tall boxes detected. Check for letterboxing issues.")
        if wide_cnt > len(all_w) * 0.1:  # More than 10% are very wide
            print(f"⚠️  WARNING: High proportion of wide boxes detected. Check for padding issues.")
            
        # Aspect ratio analysis
        aspect_ratios = [w/h for w, h in zip(all_w, all_h)]
        print(f"Aspect ratio: min={min(aspect_ratios):.3f}  p50={q(aspect_ratios,50):.3f}  p95={q(aspect_ratios,95):.3f}  max={max(aspect_ratios):.3f}")
        
    else:
        print("No boxes collected for stats (all empty?).")

    print(f"\nVisualization images saved to: {args.out_dir}")
    print("Check the images to verify detection quality and identify any issues.")

if __name__ == "__main__":
    main()
