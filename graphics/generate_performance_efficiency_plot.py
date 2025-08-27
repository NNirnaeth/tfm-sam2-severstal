#!/usr/bin/env python3
"""
Generate Performance-Efficiency Plot for Severstal Dataset Models
Similar to the ADE20K performance-efficiency plot showing mIoU vs Parameters
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns
import os

# Set style for better visualization - use compatible styles
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
        print("Using default matplotlib style")

# Set seaborn palette if available
try:
    sns.set_palette("husl")
except:
    print("Seaborn palette not available, using default colors")

def create_performance_efficiency_plot():
    """Create performance-efficiency plot for Severstal models"""
    
    # Model data from the README table
    models_data = {
        'SAM2-LoRA': {
            'mIoU': 0.505,
            'params_millions': 0.8,  # LoRA has ~99% parameter reduction
            'model_type': 'SAM2',
            'color': '#FF6B6B',
            'marker': 'D',
            'size': 200
        },
        'SAM2-Large': {
            'mIoU': 0.544,
            'params_millions': 1000,  # SAM2-Large has ~1B parameters
            'model_type': 'SAM2',
            'color': '#4ECDC4',
            'marker': 'D',
            'size': 200
        },
        'SAM2-Large-Base': {
            'mIoU': 0.687,
            'params_millions': 1000,  # Same as SAM2-Large
            'model_type': 'SAM2',
            'color': '#45B7D1',
            'marker': 'D',
            'size': 200
        },
        'SAM2-Small': {
            'mIoU': 0.323,
            'params_millions': 100,  # SAM2-Small has ~100M parameters
            'model_type': 'SAM2',
            'color': '#96CEB4',
            'marker': 'D',
            'size': 200
        },
        'SAM2-Small-Base': {
            'mIoU': 0.123,
            'params_millions': 100,  # Same as SAM2-Small
            'model_type': 'SAM2',
            'color': '#FFEAA7',
            'marker': 'D',
            'size': 200
        },
        'UNet-Std-1e4': {
            'mIoU': 0.557,
            'params_millions': 25,  # UNet with ResNet34 backbone
            'model_type': 'UNet',
            'color': '#DDA0DD',
            'marker': 'o',
            'size': 150
        },
        'UNet-Std-1e3': {
            'mIoU': 0.546,
            'params_millions': 25,  # Same architecture
            'model_type': 'UNet',
            'color': '#98D8C8',
            'marker': 'o',
            'size': 150
        },
        'DSC-UNet-1e4': {
            'mIoU': 0.559,
            'params_millions': 30,  # Deep supervision adds parameters
            'model_type': 'UNet',
            'color': '#F7DC6F',
            'marker': 's',
            'size': 150
        },
        'DSC-UNet-1e3': {
            'mIoU': 0.534,
            'params_millions': 30,  # Same architecture
            'model_type': 'UNet',
            'color': '#BB8FCE',
            'marker': 's',
            'size': 150
        },
        'UNet++-1e4': {
            'mIoU': 0.544,
            'params_millions': 28,  # Nested connections add parameters
            'model_type': 'UNet',
            'color': '#85C1E9',
            'marker': '^',
            'size': 150
        },
        'UNet++-1e3': {
            'mIoU': 0.524,
            'params_millions': 28,  # Same architecture
            'model_type': 'UNet',
            'color': '#F8C471',
            'marker': '^',
            'size': 150
        },
        'YOLOv8-Seg-1e4': {
            'mIoU': 0.480,
            'params_millions': 20,  # YOLOv8-Seg parameters
            'model_type': 'YOLO',
            'color': '#E74C3C',
            'marker': 'v',
            'size': 150
        },
        'YOLOv8-Seg-5e4': {
            'mIoU': 0.473,
            'params_millions': 20,  # Same architecture
            'model_type': 'YOLO',
            'color': '#C0392B',
            'marker': 'v',
            'size': 150
        },
        'YOLOv8-Seg-1e3': {
            'mIoU': 0.458,
            'params_millions': 20,  # Same architecture
            'model_type': 'YOLO',
            'color': '#A93226',
            'marker': 'v',
            'size': 150
        }
    }
    
    # Create figure with more space for layout
    fig = plt.figure(figsize=(18, 14))
    
    # Create main plot area with adjusted position - more space for plot
    ax = fig.add_axes([0.15, 0.25, 0.6, 0.6])  # [left, bottom, width, height] - reduced width to make room for key insights
    
    # Plot models by type with increased spacing
    for model_name, data in models_data.items():
        ax.scatter(data['params_millions'], data['mIoU'], 
                  c=data['color'], marker=data['marker'], s=data['size'],
                  alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add model labels with better positioning
        ax.annotate(model_name, 
                   (data['params_millions'], data['mIoU']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add trend lines for each model type
    model_types = ['SAM2', 'UNet', 'YOLO']
    colors = ['#FF6B6B', '#DDA0DD', '#E74C3C']
    
    for i, model_type in enumerate(model_types):
        type_data = [(d['params_millions'], d['mIoU']) 
                    for d in models_data.values() 
                    if d['model_type'] == model_type]
        
        if len(type_data) > 1:
            # Sort by parameters
            type_data.sort(key=lambda x: x[0])
            params, mious = zip(*type_data)
            
            # Add trend line
            ax.plot(params, mious, color=colors[i], linewidth=2, alpha=0.7,
                   label=f'{model_type} Trend')
    
    # Customize the plot
    ax.set_xlabel('Parámetros (Millones)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Severstal mIoU', fontsize=14, fontweight='bold')
    ax.set_title('Trade-off Rendimiento-Eficiencia: Modelos de Segmentación en Dataset Severstal\n'
                'Performance-Efficiency Trade-off: Segmentation Models on Severstal Dataset', 
                fontsize=16, fontweight='bold', pad=30)
    
    # Set axis limits with much more spacing to prevent overlapping
    ax.set_xlim(-100, 1200)
    ax.set_ylim(0.0, 0.8)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF6B6B', 
                  markersize=10, label='SAM2 Models'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DDA0DD', 
                  markersize=10, label='UNet Models'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#E74C3C', 
                  markersize=10, label='YOLO Models')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add performance regions
    # High performance region
    high_perf_rect = Rectangle((0, 0.5), 1100, 0.3, 
                              facecolor='green', alpha=0.1, edgecolor='green', linewidth=2)
    ax.add_patch(high_perf_rect)
    ax.text(50, 0.65, 'Alto Rendimiento\nHigh Performance', 
            fontsize=12, fontweight='bold', color='green')
    
    # Efficient region
    efficient_rect = Rectangle((0, 0.0), 50, 0.5, 
                              facecolor='blue', alpha=0.1, edgecolor='blue', linewidth=2)
    ax.add_patch(efficient_rect)
    ax.text(25, 0.25, 'Eficiente\nEfficient', 
            fontsize=12, fontweight='bold', color='blue', ha='center')
    
    # Add key insights at the BOTTOM RIGHT of the plot area
    insights_text = """
    Key Insights:
    • SAM2-LoRA: Best efficiency-performance balance
    • UNet Standard: Most efficient traditional architecture
    • SAM2-Large: Highest performance but highest computational cost
    • YOLO: Good balance for real-time applications
    """
    
    # Position key insights at bottom right of the plot
    ax.text(0.98, 0.02, insights_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='blue'))
    
    # Add embedded table at the BOTTOM of the figure
    table_data = [
        ['Model', 'mIoU', 'Params (M)', 'Type'],
        ['SAM2-LoRA', '0.505', '0.8', 'LoRA'],
        ['UNet-Std-1e4', '0.557', '25', 'Traditional'],
        ['YOLOv8-Seg-1e4', '0.480', '20', 'Real-time'],
        ['SAM2-Large', '0.544', '1000', 'Foundation']
    ]
    
    table = fig.add_axes([0.15, 0.05, 0.6, 0.15])  # Position at bottom, aligned with plot
    table.axis('tight')
    table.axis('off')
    
    # Create table
    table_obj = table.table(cellText=table_data[1:], colLabels=table_data[0],
                           cellLoc='center', loc='center',
                           bbox=[0, 0, 1, 1])
    
    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(10)
    table_obj.scale(1, 2)
    
    # Style table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:  # Header row
                table_obj[(i, j)].set_facecolor('#4ECDC4')
                table_obj[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table_obj[(i, j)].set_facecolor('#F8F9FA')
    
    return fig

def create_learning_rate_comparison():
    """Create learning rate comparison subplot"""
    
    # Learning rate comparison data
    lr_data = {
        'UNet Standard': {
            '1e-4': {'mIoU': 0.557, 'mDice': 0.692},
            '1e-3': {'mIoU': 0.546, 'mDice': 0.679}
        },
        'DSC-UNet': {
            '1e-4': {'mIoU': 0.559, 'mDice': 0.692},
            '1e-3': {'mIoU': 0.534, 'mDice': 0.667}
        },
        'UNet++': {
            '1e-4': {'mIoU': 0.544, 'mDice': 0.679},
            '1e-3': {'mIoU': 0.524, 'mDice': 0.648}
        },
        'YOLOv8-Seg': {
            '1e-4': {'mIoU': 0.480, 'mDice': 0.624},
            '5e-4': {'mIoU': 0.473, 'mDice': 0.615},
            '1e-3': {'mIoU': 0.458, 'mDice': 0.599}
        }
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # mIoU comparison
    x = np.arange(len(lr_data))
    width = 0.35
    
    for i, (model, lrs) in enumerate(lr_data.items()):
        if len(lrs) == 2:  # UNet variants
            lr1, lr2 = list(lrs.keys())
            mious = [lrs[lr1]['mIoU'], lrs[lr2]['mIoU']]
            ax1.bar([i*3, i*3+1], mious, width, 
                   label=f'{model} ({lr1}, {lr2})', alpha=0.8)
        else:  # YOLO
            mious = [lrs[lr]['mIoU'] for lr in lrs.keys()]
            ax1.bar([i*3, i*3+1, i*3+2], mious, width, 
                   label=f'{model} (1e-4, 5e-4, 1e-3)', alpha=0.8)
    
    ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax1.set_ylabel('mIoU', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de Learning Rates: mIoU', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # mDice comparison
    for i, (model, lrs) in enumerate(lr_data.items()):
        if len(lrs) == 2:  # UNet variants
            lr1, lr2 = list(lrs.keys())
            mdices = [lrs[lr1]['mDice'], lrs[lr2]['mDice']]
            ax2.bar([i*3, i*3+1], mdices, width, 
                   label=f'{model} ({lr1}, {lr2})', alpha=0.8)
        else:  # YOLO
            mdices = [lrs[lr]['mDice'] for lr in lrs.keys()]
            ax2.bar([i*3, i*3+1, i*3+2], mdices, width, 
                   label=f'{model} (1e-4, 5e-4, 1e-3)', alpha=0.8)
    
    ax2.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax2.set_ylabel('mDice', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de Learning Rates: mDice', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Get current directory (graphics folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate main performance-efficiency plot
    print("Generando gráfica principal de rendimiento-eficiencia...")
    fig1 = create_performance_efficiency_plot()
    plot1_path = os.path.join(current_dir, 'severstal_performance_efficiency_plot.png')
    fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica principal guardada como: {plot1_path}")
    
    # Generate learning rate comparison
    print("Generando gráfica de comparación de learning rates...")
    fig2 = create_learning_rate_comparison()
    plot2_path = os.path.join(current_dir, 'severstal_learning_rate_comparison.png')
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica de comparación guardada como: {plot2_path}")
    
    print("\n¡Gráficas generadas exitosamente!")
    print(f"1. {plot1_path} - Trade-off rendimiento-eficiencia")
    print(f"2. {plot2_path} - Comparación de learning rates")
    
    # Show plots
    plt.show()
