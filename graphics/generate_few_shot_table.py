#!/usr/bin/env python3
"""
Generate Few-Shot Learning Results Table
Visualizes the performance of SAM2 models across different subset sizes
Based on actual test set evaluation results from August 23rd, 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

def create_few_shot_table():
    """Create a comprehensive table visualization of few-shot results"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Data for SAM2 Small (only 4000 images has test results)
    small_data = {
        'Subset': [4000],
        'Test_IoU': [0.462],
        'Test_Dice': [0.603],
        'IoU@50': [39.8],
        'IoU@75': [10.5]
    }
    
    # Data for SAM2 Large (test set results from Aug 23rd)
    large_data = {
        'Subset': [100, 200, 500, 1000, 2000, 4000],
        'Test_IoU': [0.621, 0.483, 0.746, 0.365, 0.613, 0.249],
        'Test_Dice': [0.759, 0.617, 0.829, 0.455, 0.720, 0.333],
        'IoU@50': [80.0, 50.7, 82.6, 38.7, 73.6, 18.9],
        'IoU@75': [10.0, 11.7, 61.2, 12.9, 35.5, 11.3]
    }
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Few-Shot Learning Results - SAM2 on Severstal Dataset (Test Set)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Plot 1: IoU Performance Comparison
    x = np.arange(len(large_data['Subset']))
    width = 0.35
    
    # Only plot SAM2 Large for IoU comparison (SAM2 Small has limited data)
    bars1 = ax1.bar(x, large_data['Test_IoU'], width, 
                     label='SAM2 Large', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Subset Size (Number of Images)', fontsize=12)
    ax1.set_ylabel('Test IoU Score', fontsize=12)
    ax1.set_title('Test Set IoU Performance by Subset Size', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(large_data['Subset'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Dice Performance Comparison
    bars2 = ax2.bar(x, large_data['Test_Dice'], width, 
                     label='SAM2 Large', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Subset Size (Number of Images)', fontsize=12)
    ax2.set_ylabel('Test Dice Score', fontsize=12)
    ax2.set_title('Test Set Dice Performance by Subset Size', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(large_data['Subset'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: IoU@50 Performance
    bars3 = ax3.bar(x, large_data['IoU@50'], width, 
                     label='SAM2 Large', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('Subset Size (Number of Images)', fontsize=12)
    ax3.set_ylabel('IoU@50 (%)', fontsize=12)
    ax3.set_title('IoU@50 Performance by Subset Size', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(large_data['Subset'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: IoU@75 Performance
    bars4 = ax4.bar(x, large_data['IoU@75'], width, 
                     label='SAM2 Large', alpha=0.8, color='lightcoral')
    
    ax4.set_xlabel('Subset Size (Number of Images)', fontsize=12)
    ax4.set_ylabel('IoU@75 (%)', fontsize=12)
    ax4.set_title('IoU@75 Performance by Subset Size', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(large_data['Subset'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('few_shot_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Few-shot performance analysis plot saved as 'few_shot_performance_analysis.png'")
    
    return fig

def create_efficiency_heatmap():
    """Create a heatmap showing efficiency metrics"""
    
    # Efficiency data (IoU per 1000 images)
    subset_sizes = [100, 200, 500, 1000, 2000, 4000]
    
    # Calculate efficiency: IoU / (subset_size/100)
    large_efficiency = [0.621/1.0, 0.483/2.0, 0.746/5.0, 
                       0.365/10.0, 0.613/20.0, 0.249/40.0]
    
    # Create heatmap data
    efficiency_data = np.array([large_efficiency])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create heatmap
    im = ax.imshow(efficiency_data, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(subset_sizes)))
    ax.set_xticklabels(subset_sizes)
    ax.set_yticks([0])
    ax.set_yticklabels(['SAM2 Large'])
    
    ax.set_xlabel('Subset Size (Number of Images)', fontsize=12)
    ax.set_title('Training Efficiency: IoU per 100 Images', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('IoU per 100 Images', fontsize=12)
    
    # Add text annotations
    for j in range(len(subset_sizes)):
        text = ax.text(j, 0, f'{efficiency_data[0, j]:.3f}',
                      ha="center", va="center", color="black", fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('few_shot_efficiency_heatmap.png', dpi=300, bbox_inches='tight')
    print("Efficiency heatmap saved as 'few_shot_efficiency_heatmap.png'")
    
    return fig

def create_comprehensive_comparison():
    """Create a comprehensive comparison of all metrics"""
    
    # Data for comprehensive comparison
    subset_sizes = [100, 200, 500, 1000, 2000, 4000]
    
    # Metrics for SAM2 Large
    iou_scores = [0.621, 0.483, 0.746, 0.365, 0.613, 0.249]
    dice_scores = [0.759, 0.617, 0.829, 0.455, 0.720, 0.333]
    iou_50 = [80.0, 50.7, 82.6, 38.7, 73.6, 18.9]
    iou_75 = [10.0, 11.7, 61.2, 12.9, 35.5, 11.3]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create multiple y-axes
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot IoU and Dice (left y-axis)
    line1 = ax.plot(subset_sizes, iou_scores, 'o-', color='blue', linewidth=2, 
                    markersize=8, label='Test IoU')
    line2 = ax.plot(subset_sizes, dice_scores, 's-', color='green', linewidth=2, 
                    markersize=8, label='Test Dice')
    
    # Plot IoU@50 (right y-axis)
    line3 = ax2.plot(subset_sizes, iou_50, '^-', color='red', linewidth=2, 
                     markersize=8, label='IoU@50 (%)')
    
    # Plot IoU@75 (third y-axis)
    line4 = ax3.plot(subset_sizes, iou_75, 'd-', color='orange', linewidth=2, 
                     markersize=8, label='IoU@75 (%)')
    
    # Set labels and titles
    ax.set_xlabel('Subset Size (Number of Images)', fontsize=12)
    ax.set_ylabel('IoU / Dice Score', fontsize=12, color='blue')
    ax2.set_ylabel('IoU@50 (%)', fontsize=12, color='red')
    ax3.set_ylabel('IoU@75 (%)', fontsize=12, color='orange')
    ax.set_title('Comprehensive Few-Shot Performance Analysis - SAM2 Large', 
                 fontsize=14, fontweight='bold')
    
    # Set colors for y-axis labels
    ax.yaxis.label.set_color('blue')
    ax2.yaxis.label.set_color('red')
    ax3.yaxis.label.set_color('orange')
    
    # Set colors for y-axis ticks
    ax.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='red')
    ax3.tick_params(axis='y', colors='orange')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    
    # Highlight best performance
    best_idx = np.argmax(iou_scores)
    ax.annotate(f'Best: {iou_scores[best_idx]:.3f}', 
                xy=(subset_sizes[best_idx], iou_scores[best_idx]),
                xytext=(subset_sizes[best_idx] + 200, iou_scores[best_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('few_shot_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive analysis plot saved as 'few_shot_comprehensive_analysis.png'")
    
    return fig

if __name__ == "__main__":
    print("Generating few-shot learning results visualizations...")
    
    # Generate performance comparison
    fig1 = create_few_shot_table()
    
    # Generate efficiency heatmap
    fig2 = create_efficiency_heatmap()
    
    # Generate comprehensive comparison
    fig3 = create_comprehensive_comparison()
    
    print("All visualizations completed!")
    plt.show()
