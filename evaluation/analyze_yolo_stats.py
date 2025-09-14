#!/usr/bin/env python3
"""
Analyze YOLO prediction statistics in detail.
Provides comprehensive analysis of detection patterns and potential issues.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def analyze_predictions(labels_dir, out_dir):
    """Comprehensive analysis of YOLO predictions"""
    
    # Collect all data
    all_boxes = []
    image_stats = []
    confidence_dist = []
    
    label_files = list(Path(labels_dir).glob("*.txt"))
    print(f"Analyzing {len(label_files)} prediction files...")
    
    for txt_path in label_files:
        boxes = read_yolo_txt(str(txt_path))
        
        # Image-level stats
        num_detections = len(boxes)
        image_stats.append({
            'filename': txt_path.stem,
            'num_detections': num_detections,
            'has_detections': num_detections > 0
        })
        
        # Box-level stats
        for box in boxes:
            all_boxes.append({
                'filename': txt_path.stem,
                'cls': box['cls'],
                'cx': box['cx'],
                'cy': box['cy'],
                'w': box['w'],
                'h': box['h'],
                'conf': box['conf'],
                'aspect_ratio': box['w'] / box['h'],
                'area': box['w'] * box['h']
            })
            confidence_dist.append(box['conf'])
    
    # Convert to DataFrames
    df_boxes = pd.DataFrame(all_boxes)
    df_images = pd.DataFrame(image_stats)
    
    print(f"\n=== COMPREHENSIVE ANALYSIS ===")
    print(f"Total images: {len(df_images)}")
    print(f"Images with detections: {df_images['has_detections'].sum()}")
    print(f"Images without detections: {(~df_images['has_detections']).sum()}")
    print(f"Total detections: {len(df_boxes)}")
    
    if len(df_boxes) > 0:
        print(f"\n=== DETECTION STATISTICS ===")
        print(f"Average detections per image: {df_images['num_detections'].mean():.2f}")
        print(f"Max detections in single image: {df_images['num_detections'].max()}")
        print(f"Min detections in single image: {df_images['num_detections'].min()}")
        
        print(f"\n=== CONFIDENCE STATISTICS ===")
        print(f"Confidence - min: {df_boxes['conf'].min():.3f}")
        print(f"Confidence - mean: {df_boxes['conf'].mean():.3f}")
        print(f"Confidence - median: {df_boxes['conf'].median():.3f}")
        print(f"Confidence - p95: {df_boxes['conf'].quantile(0.95):.3f}")
        print(f"Confidence - max: {df_boxes['conf'].max():.3f}")
        
        print(f"\n=== GEOMETRY STATISTICS ===")
        print(f"Width - min: {df_boxes['w'].min():.3f}, p50: {df_boxes['w'].median():.3f}, p95: {df_boxes['w'].quantile(0.95):.3f}, max: {df_boxes['w'].max():.3f}")
        print(f"Height - min: {df_boxes['h'].min():.3f}, p50: {df_boxes['h'].median():.3f}, p95: {df_boxes['h'].quantile(0.95):.3f}, max: {df_boxes['h'].max():.3f}")
        print(f"Aspect ratio - min: {df_boxes['aspect_ratio'].min():.3f}, p50: {df_boxes['aspect_ratio'].median():.3f}, p95: {df_boxes['aspect_ratio'].quantile(0.95):.3f}, max: {df_boxes['aspect_ratio'].max():.3f}")
        print(f"Area - min: {df_boxes['area'].min():.3f}, p50: {df_boxes['area'].median():.3f}, p95: {df_boxes['area'].quantile(0.95):.3f}, max: {df_boxes['area'].max():.3f}")
        
        # Check for potential issues
        print(f"\n=== POTENTIAL ISSUES ===")
        tall_boxes = (df_boxes['h'] >= 0.95).sum()
        wide_boxes = (df_boxes['w'] >= 0.95).sum()
        very_small_boxes = (df_boxes['area'] <= 0.001).sum()
        very_large_boxes = (df_boxes['area'] >= 0.5).sum()
        
        print(f"Very tall boxes (h>=0.95): {tall_boxes} ({tall_boxes/len(df_boxes)*100:.1f}%)")
        print(f"Very wide boxes (w>=0.95): {wide_boxes} ({wide_boxes/len(df_boxes)*100:.1f}%)")
        print(f"Very small boxes (area<=0.001): {very_small_boxes} ({very_small_boxes/len(df_boxes)*100:.1f}%)")
        print(f"Very large boxes (area>=0.5): {very_large_boxes} ({very_large_boxes/len(df_boxes)*100:.1f}%)")
        
        # Position analysis
        print(f"\n=== POSITION ANALYSIS ===")
        edge_cx = ((df_boxes['cx'] <= 0.05) | (df_boxes['cx'] >= 0.95)).sum()
        edge_cy = ((df_boxes['cy'] <= 0.05) | (df_boxes['cy'] >= 0.95)).sum()
        print(f"Boxes near horizontal edges (cx<=0.05 or cx>=0.95): {edge_cx} ({edge_cx/len(df_boxes)*100:.1f}%)")
        print(f"Boxes near vertical edges (cy<=0.05 or cy>=0.95): {edge_cy} ({edge_cy/len(df_boxes)*100:.1f}%)")
        
        # Create visualizations
        create_visualizations(df_boxes, df_images, out_dir)
        
        # Save detailed CSV
        csv_path = os.path.join(out_dir, "detailed_analysis.csv")
        df_boxes.to_csv(csv_path, index=False)
        print(f"\nDetailed analysis saved to: {csv_path}")
    
    return df_boxes, df_images

def create_visualizations(df_boxes, df_images, out_dir):
    """Create visualization plots"""
    
    if len(df_boxes) == 0:
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YOLO Detection Analysis', fontsize=16)
    
    # 1. Confidence distribution
    axes[0,0].hist(df_boxes['conf'], bins=50, alpha=0.7, color='blue')
    axes[0,0].set_title('Confidence Distribution')
    axes[0,0].set_xlabel('Confidence')
    axes[0,0].set_ylabel('Count')
    axes[0,0].axvline(df_boxes['conf'].mean(), color='red', linestyle='--', label=f'Mean: {df_boxes["conf"].mean():.3f}')
    axes[0,0].legend()
    
    # 2. Width vs Height scatter
    axes[0,1].scatter(df_boxes['w'], df_boxes['h'], alpha=0.6, s=20)
    axes[0,1].set_title('Width vs Height (Normalized)')
    axes[0,1].set_xlabel('Width')
    axes[0,1].set_ylabel('Height')
    axes[0,1].set_xlim(0, 1)
    axes[0,1].set_ylim(0, 1)
    
    # 3. Aspect ratio distribution
    axes[0,2].hist(df_boxes['aspect_ratio'], bins=50, alpha=0.7, color='green')
    axes[0,2].set_title('Aspect Ratio Distribution')
    axes[0,2].set_xlabel('Aspect Ratio (w/h)')
    axes[0,2].set_ylabel('Count')
    axes[0,2].axvline(df_boxes['aspect_ratio'].median(), color='red', linestyle='--', label=f'Median: {df_boxes["aspect_ratio"].median():.3f}')
    axes[0,2].legend()
    
    # 4. Detections per image
    axes[1,0].hist(df_images['num_detections'], bins=range(0, df_images['num_detections'].max()+2), alpha=0.7, color='orange')
    axes[1,0].set_title('Detections per Image')
    axes[1,0].set_xlabel('Number of Detections')
    axes[1,0].set_ylabel('Number of Images')
    
    # 5. Box area distribution
    axes[1,1].hist(df_boxes['area'], bins=50, alpha=0.7, color='purple')
    axes[1,1].set_title('Box Area Distribution')
    axes[1,1].set_xlabel('Area (normalized)')
    axes[1,1].set_ylabel('Count')
    axes[1,1].axvline(df_boxes['area'].median(), color='red', linestyle='--', label=f'Median: {df_boxes["area"].median():.3f}')
    axes[1,1].legend()
    
    # 6. Position heatmap
    axes[1,2].hist2d(df_boxes['cx'], df_boxes['cy'], bins=20, alpha=0.7)
    axes[1,2].set_title('Detection Position Heatmap')
    axes[1,2].set_xlabel('Center X')
    axes[1,2].set_ylabel('Center Y')
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "detection_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization plots saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze YOLO prediction statistics")
    parser.add_argument("--labels_dir", required=True, help="Path to prediction labels directory")
    parser.add_argument("--out_dir", default="analysis_results", help="Output directory for analysis results")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    df_boxes, df_images = analyze_predictions(args.labels_dir, args.out_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
