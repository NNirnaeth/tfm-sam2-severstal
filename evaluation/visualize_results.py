#!/usr/bin/env python3
"""
Visualization script for SAM2 evaluation results
Following TFM guidelines with Okabe-Ito color palette
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

# Set matplotlib parameters for TFM quality
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300
})

# Okabe-Ito color palette (colorblind-friendly)
OKABE_ITO_COLORS = {
    'blue': '#0072B2',      # Blue
    'green': '#009E73',     # Green  
    'orange': '#E69F00',    # Orange
    'red': '#D55E00',       # Red
    'purple': '#CC79A7',    # Pink/Purple
    'brown': '#F0E442',     # Yellow
    'pink': '#56B4E9',      # Light Blue
    'gray': '#999999'       # Gray
}

# SAM2 specific colors
SAM2_COLORS = {
    'small': OKABE_ITO_COLORS['blue'],
    'large': OKABE_ITO_COLORS['green'],
    'iou': OKABE_ITO_COLORS['blue'],
    'f1': OKABE_ITO_COLORS['green'],
    'recall': OKABE_ITO_COLORS['orange']
}

class SAM2Visualizer:
    """Visualization class for SAM2 evaluation results"""
    
    def __init__(self, output_dir: str = "/home/ptp/sam2/Plots", experiment: str = "sam2_base"):
        self.base_output_dir = Path(output_dir)
        self.experiment = experiment
        self.output_dir = self.base_output_dir / experiment
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_sam2_prompts_comparison(self, save_format: str = "pdf") -> None:
        """
        Create grouped bar chart for SAM2 evaluation modes
        """
        # Updated data from your results
        labels = ["30 pts Small", "30 pts Large", "3 pts Small", "3 pts Large", 
                 "Auto Small", "Auto Large"]
        iou = [0.284, 0.687, 0.290, 0.313, 0.436, 0.683]
        f1 = [0.429, 0.804, 0.420, 0.465, 0.587, 0.806]
        recall = [0.300, 0.795, 0.307, 0.332, 0.496, 0.783]
        
        # Create figure with proper aspect ratio
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up bar positions
        x = np.arange(len(labels))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x - width, iou, width, label='IoU', 
                      color=SAM2_COLORS['iou'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x, f1, width, label='F1-Score', 
                      color=SAM2_COLORS['f1'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars3 = ax.bar(x + width, recall, width, label='Recall', 
                      color=SAM2_COLORS['recall'], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Customize plot
        ax.set_ylabel('Métricas de Evaluación', fontweight='bold')
        ax.set_xlabel('Modos de Evaluación SAM2', fontweight='bold')
        ax.set_title('Comparación de Métricas SAM2: Small vs Large\n(n=2000, Prompts: 30 pts, 3 pts, Auto)', 
                    fontweight='bold', pad=20)
        
        # Set x-axis labels with rotation
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Set y-axis limits for better comparison
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # Add value labels on bars
        self._add_value_labels(ax, bars1, iou)
        self._add_value_labels(ax, bars2, f1)
        self._add_value_labels(ax, bars3, recall)
        
        # Customize legend
        ax.legend(loc='upper left', frameon=False, ncol=3, 
                 bbox_to_anchor=(0, 1.02))
        
        # Add subtle grid
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Tight layout and save
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_prompts_comparison.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_prompts_comparison.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_prompts_comparison.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ SAM2 prompts comparison saved to {self.output_dir}")
    
    def _add_value_labels(self, ax, bars, values):
        """Add value labels on top of bars"""
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    def create_model_size_comparison(self, save_format: str = "pdf") -> None:
        """
        Create comparison between Small and Large models
        """
        # Data organized by model size
        metrics = ['IoU', 'F1-Score', 'Recall']
        small_values = [0.1228, 0.2171, 0.1268]  # 30 pts
        large_values = [0.6868, 0.8041, 0.7950]  # 30 pts
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars1 = ax.bar(x - width/2, small_values, width, label='SAM2-Small', 
                      color=SAM2_COLORS['small'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, large_values, width, label='SAM2-Large', 
                      color=SAM2_COLORS['large'], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        ax.set_ylabel('Valores de Métricas', fontweight='bold')
        ax.set_xlabel('Métricas de Evaluación', fontweight='bold')
        ax.set_title('Comparación SAM2-Small vs SAM2-Large\n(30 puntos, n=2000)', 
                    fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.0)
        
        # Add value labels
        for bars, values in [(bars1, small_values), (bars2, large_values)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_model_size_comparison.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_model_size_comparison.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_model_size_comparison.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ Model size comparison saved to {self.output_dir}")
    
    def create_prompt_type_analysis(self, save_format: str = "pdf") -> None:
        """
        Create analysis of different prompt types
        """
        # Data for prompt type analysis
        prompt_types = ['30 puntos', '3 puntos', 'Auto']
        small_iou = [0.1228, 0.3493, 0.8011]
        large_iou = [0.6868, 0.3126, 0.6834]
        
        x = np.arange(len(prompt_types))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        bars1 = ax.bar(x - width/2, small_iou, width, label='SAM2-Small', 
                      color=SAM2_COLORS['small'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, large_iou, width, label='SAM2-Large', 
                      color=SAM2_COLORS['large'], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        ax.set_ylabel('IoU Score', fontweight='bold')
        ax.set_xlabel('Tipo de Prompt', fontweight='bold')
        ax.set_title('Análisis de Tipos de Prompt SAM2\n(n=2000, métrica: IoU)', 
                    fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_types)
        ax.set_ylim(0, 1.0)
        
        # Add value labels
        for bars, values in [(bars1, small_iou), (bars2, large_iou)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_prompt_type_analysis.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_prompt_type_analysis.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_prompt_type_analysis.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ Prompt type analysis saved to {self.output_dir}")
    
    def create_finetuned_metrics_comparison(self, save_format: str = "png") -> None:
        """
        Create grouped bar chart for SAM2 Fine-tuned models (IoU, Dice, F1)
        """
        # Updated data from the table for Fine-tuned models
        labels = ["30 pts Small FT", "30 pts Large FT", "3 pts Small FT", "3 pts Large FT", 
                 "Auto Small FT", "Auto Large FT"]
        
        # Updated values including Small FT 3 pts data
        iou = [0.323, 0.412, 0.300, 0.435, 0.036, 0.194]
        dice = [0.471, 0.534, 0.460, 0.570, 0.067, 0.289]
        f1 = [0.471, 0.534, 0.450, 0.570, 0.067, 0.289]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set up bar positions
        x = np.arange(len(labels))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x - width, iou, width, label='IoU', 
                      color=SAM2_COLORS['iou'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x, dice, width, label='Dice', 
                      color=SAM2_COLORS['f1'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', 
                      color=SAM2_COLORS['recall'], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Customize plot
        ax.set_ylabel('Valores de Métricas', fontweight='bold')
        ax.set_xlabel('Modos de Evaluación SAM2 Fine-tuned', fontweight='bold')
        ax.set_title('Comparación de Métricas SAM2 Fine-tuned: Small vs Large\n(n=2000, Prompts: 30 pts, 3 pts, Auto)', 
                    fontweight='bold', pad=20)
        
        # Set x-axis labels with rotation
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Set y-axis limits
        ax.set_ylim(0, 0.7)
        ax.set_yticks(np.arange(0, 0.8, 0.1))
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, iou)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        for i, (bar, value) in enumerate(zip(bars2, dice)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        for i, (bar, value) in enumerate(zip(bars3, f1)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Customize legend
        ax.legend(loc='upper right', frameon=False, ncol=3)
        
        # Add subtle grid
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Tight layout and save
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_finetuned_metrics_comparison.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_finetuned_metrics_comparison.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_finetuned_metrics_comparison.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ Fine-tuned metrics comparison saved to {self.output_dir}")
    
    def create_zeroshot_vs_finetuned_comparison(self, save_format: str = "png") -> None:
        """
        Create line chart comparing Zero-shot vs Fine-tuned performance
        """
        # Data for comparison (using Dice metric as requested)
        prompt_modes = ['30 pts', '3 pts', 'Auto']
        
        # Zero-shot data (from original sam2_base results)
        zeroshot_dice = [0.2171, 0.5046, 0.8812]  # Small model
        zeroshot_dice_large = [0.8041, 0.4648, 0.8061]  # Large model
        
        # Fine-tuned data (from the table)
        finetuned_dice_small = [0.471, 0.0, 0.067]  # Small FT (3 pts = N/A)
        finetuned_dice_large = [0.534, 0.570, 0.289]  # Large FT
        
        x = np.arange(len(prompt_modes))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Small model comparison
        ax1.plot(x, zeroshot_dice, 'o-', label='Zero-shot', 
                color=OKABE_ITO_COLORS['blue'], linewidth=2, markersize=8)
        ax1.plot(x, finetuned_dice_small, 's-', label='Fine-tuned', 
                color=OKABE_ITO_COLORS['green'], linewidth=2, markersize=8)
        
        ax1.set_ylabel('Dice Score', fontweight='bold')
        ax1.set_xlabel('Modos de Prompt', fontweight='bold')
        ax1.set_title('SAM2-Small: Zero-shot vs Fine-tuned\n(n=2000)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(prompt_modes)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.legend(frameon=False)
        
        # Add value labels
        for i, (zs, ft) in enumerate(zip(zeroshot_dice, finetuned_dice_small)):
            ax1.text(i, zs + 0.05, f'{zs:.3f}', ha='center', va='bottom', fontsize=9)
            if ft > 0:  # Skip N/A values
                ax1.text(i, ft + 0.05, f'{ft:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Large model comparison
        ax2.plot(x, zeroshot_dice_large, 'o-', label='Zero-shot', 
                color=OKABE_ITO_COLORS['blue'], linewidth=2, markersize=8)
        ax2.plot(x, finetuned_dice_large, 's-', label='Fine-tuned', 
                color=OKABE_ITO_COLORS['green'], linewidth=2, markersize=8)
        
        ax2.set_ylabel('Dice Score', fontweight='bold')
        ax2.set_xlabel('Modos de Prompt', fontweight='bold')
        ax2.set_title('SAM2-Large: Zero-shot vs Fine-tuned\n(n=2000)', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(prompt_modes)
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.legend(frameon=False)
        
        # Add value labels
        for i, (zs, ft) in enumerate(zip(zeroshot_dice_large, finetuned_dice_large)):
            ax2.text(i, zs + 0.05, f'{zs:.3f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i, ft + 0.05, f'{ft:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_zeroshot_vs_finetuned.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_zeroshot_vs_finetuned.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_zeroshot_vs_finetuned.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ Zero-shot vs Fine-tuned comparison saved to {self.output_dir}")
    
    def create_subset_performance_analysis(self, save_format: str = "png") -> None:
        """
        Create line plot comparing Small vs Large models across different dataset sizes
        """
        # Data from the table for subset analysis
        dataset_sizes = ['500', '1000', '2000']
        
        # Small model data
        small_iou = [0.381, 0.427, 0.436]
        small_dice = [0.528, 0.598, 0.610]
        
        # Large model data
        large_iou = [0.451, 0.447, 0.452]
        large_dice = [0.615, 0.626, 0.640]
        
        x = np.arange(len(dataset_sizes))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Dice comparison (main metric)
        ax1.plot(x, small_dice, 'o-', label='SAM2-Small', 
                color=SAM2_COLORS['small'], linewidth=2.5, markersize=10, 
                markerfacecolor=SAM2_COLORS['small'], markeredgecolor='white', markeredgewidth=1)
        ax1.plot(x, large_dice, '^-', label='SAM2-Large', 
                color=SAM2_COLORS['large'], linewidth=2.5, markersize=10,
                markerfacecolor=SAM2_COLORS['large'], markeredgecolor='white', markeredgewidth=1)
        
        ax1.set_ylabel('Dice Score', fontweight='bold')
        ax1.set_xlabel('Tamaño del Dataset', fontweight='bold')
        ax1.set_title('Rendimiento por Tamaño de Dataset\n(Métrica: Dice)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(dataset_sizes)
        ax1.set_ylim(0.4, 0.7)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.legend(frameon=False, loc='lower right')
        
        # Add value labels for Dice
        for i, (small_val, large_val) in enumerate(zip(small_dice, large_dice)):
            ax1.text(i, small_val + 0.01, f'{small_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i, large_val + 0.01, f'{large_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # IoU comparison (optional metric)
        ax2.plot(x, small_iou, 'o-', label='SAM2-Small', 
                color=SAM2_COLORS['small'], linewidth=2.5, markersize=10,
                markerfacecolor=SAM2_COLORS['small'], markeredgecolor='white', markeredgewidth=1)
        ax2.plot(x, large_iou, '^-', label='SAM2-Large', 
                color=SAM2_COLORS['large'], linewidth=2.5, markersize=10,
                markerfacecolor=SAM2_COLORS['large'], markeredgecolor='white', markeredgewidth=1)
        
        ax2.set_ylabel('IoU Score', fontweight='bold')
        ax2.set_xlabel('Tamaño del Dataset', fontweight='bold')
        ax2.set_title('Rendimiento por Tamaño de Dataset\n(Métrica: IoU)', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(dataset_sizes)
        ax2.set_ylim(0.3, 0.5)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.legend(frameon=False, loc='lower right')
        
        # Add value labels for IoU
        for i, (small_val, large_val) in enumerate(zip(small_iou, large_iou)):
            ax2.text(i, small_val + 0.01, f'{small_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i, large_val + 0.01, f'{large_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_subset_performance_analysis.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_subset_performance_analysis.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_subset_performance_analysis.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ Subset performance analysis saved to {self.output_dir}")
    
    def create_sam2_lora_prompt_comparison(self, save_format: str = "png") -> None:
        """
        Create grouped bar chart comparing mIoU and mDice between prompt modes for SAM2+LoRA
        """
        # Data from the provided table
        data = {
            '30 pts': {'IoU': 0.240, 'Dice': 0.360},
            '3 pts': {'IoU': 0.290, 'Dice': 0.397},
            'Auto': {'IoU': 0.716, 'Dice': 0.804}
        }
        
        # Extract data for plotting
        modes = list(data.keys())
        iou_values = [data[mode]['IoU'] for mode in modes]
        dice_values = [data[mode]['Dice'] for mode in modes]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set bar width and positions
        bar_width = 0.35
        x = np.arange(len(modes))
        
        # Create bars using the same colors as other plots
        bars1 = ax.bar(x - bar_width/2, iou_values, bar_width, label='mIoU', 
                      color=SAM2_COLORS['iou'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + bar_width/2, dice_values, bar_width, label='mDice', 
                      color=SAM2_COLORS['f1'], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('Modos de Evaluación', fontweight='bold')
        ax.set_ylabel('Valor (0-1)', fontweight='bold')
        ax.set_title('Comparación de Rendimiento SAM2+LoRA por Modo de Prompt\nmIoU vs mDice', 
                    fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        self._add_value_labels(ax, bars1, iou_values)
        self._add_value_labels(ax, bars2, dice_values)
        
        # Add legend
        ax.legend(loc='upper left', frameon=False, fontsize=11)
        
        # Tight layout and save
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_lora_prompt_modes_comparison.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_lora_prompt_modes_comparison.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_lora_prompt_modes_comparison.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ SAM2+LoRA prompt modes comparison saved to {self.output_dir}")
    
    def create_faster_sam2_learning_curves(self, save_format: str = "png") -> None:
        """
        Create learning curves comparing Faster SAM2 Base vs Fine-tuned across subset sizes
        """
        # Data from the provided tables
        subset_sizes = [500, 1000, 2000, 4200]
        
        # Faster + SAM2 Base data
        base_iou = [0.460, 0.482, 0.488, 0.496]
        base_dice = [0.597, 0.621, 0.628, 0.636]
        
        # Faster + SAM2 FT Large data
        ft_iou = [0.483, 0.490, 0.522, 0.525]
        ft_dice = [0.617, 0.625, 0.657, 0.661]
        
        x = np.arange(len(subset_sizes))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # mDice comparison (main metric)
        ax1.plot(x, base_dice, 'o-', label='Faster + SAM2 Base', 
                color=OKABE_ITO_COLORS['blue'], linewidth=2.5, markersize=10, 
                markerfacecolor=OKABE_ITO_COLORS['blue'], markeredgecolor='white', markeredgewidth=1)
        ax1.plot(x, ft_dice, '^-', label='Faster + SAM2 FT Large', 
                color=OKABE_ITO_COLORS['green'], linewidth=2.5, markersize=10,
                markerfacecolor=OKABE_ITO_COLORS['green'], markeredgecolor='white', markeredgewidth=1)
        
        ax1.set_ylabel('mDice Score', fontweight='bold')
        ax1.set_xlabel('Tamaño del Subset', fontweight='bold')
        ax1.set_title('Curvas de Aprendizaje Faster SAM2\n(Métrica: mDice)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(subset_sizes)
        ax1.set_ylim(0.5, 0.7)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.legend(frameon=False, loc='lower right')
        
        # Add value labels for mDice
        for i, (base_val, ft_val) in enumerate(zip(base_dice, ft_dice)):
            ax1.text(i, base_val + 0.005, f'{base_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i, ft_val + 0.005, f'{ft_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # mIoU comparison (optional metric)
        ax2.plot(x, base_iou, 'o-', label='Faster + SAM2 Base', 
                color=OKABE_ITO_COLORS['blue'], linewidth=2.5, markersize=10,
                markerfacecolor=OKABE_ITO_COLORS['blue'], markeredgecolor='white', markeredgewidth=1)
        ax2.plot(x, ft_iou, '^-', label='Faster + SAM2 FT Large', 
                color=OKABE_ITO_COLORS['green'], linewidth=2.5, markersize=10,
                markerfacecolor=OKABE_ITO_COLORS['green'], markeredgecolor='white', markeredgewidth=1)
        
        ax2.set_ylabel('mIoU Score', fontweight='bold')
        ax2.set_xlabel('Tamaño del Subset', fontweight='bold')
        ax2.set_title('Curvas de Aprendizaje Faster SAM2\n(Métrica: mIoU)', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(subset_sizes)
        ax2.set_ylim(0.4, 0.6)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.legend(frameon=False, loc='lower right')
        
        # Add value labels for mIoU
        for i, (base_val, ft_val) in enumerate(zip(base_iou, ft_iou)):
            ax2.text(i, base_val + 0.005, f'{base_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i, ft_val + 0.005, f'{ft_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_faster_sam2_learning_curves.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_faster_sam2_learning_curves.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_faster_sam2_learning_curves.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ Faster SAM2 learning curves saved to {self.output_dir}")
    
    def create_yolo_sam2_learning_curves(self, save_format: str = "png") -> None:
        """
        Create learning curves comparing YOLO + SAM2 Base vs Fine-tuned across subset sizes
        """
        # Data from the provided table
        subset_sizes = [500, 1000, 2000, 'Full']
        subset_labels = ['500', '1000', '2000', 'Full']
        
        # YOLO + SAM2 Large Base data (only Full available)
        base_dice = [None, None, None, 0.604]  # Only Full subset available
        
        # YOLO + SAM2 Fine-tuned data
        ft_dice = [0.591, 0.614, 0.613, 0.647]  # 500, 1000, 2000, Full
        
        x = np.arange(len(subset_sizes))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Fine-tuned line (complete data)
        ax.plot(x, ft_dice, '^-', label='YOLO + SAM2 Fine-tuned', 
                color=OKABE_ITO_COLORS['green'], linewidth=2.5, markersize=10,
                markerfacecolor=OKABE_ITO_COLORS['green'], markeredgecolor='white', markeredgewidth=1)
        
        # Plot Base line (only Full subset)
        base_x = [3]  # Only Full subset (index 3)
        base_y = [0.604]  # Only Full subset value
        ax.plot(base_x, base_y, 'o', label='YOLO + SAM2 Base', 
                color=OKABE_ITO_COLORS['blue'], markersize=12,
                markerfacecolor=OKABE_ITO_COLORS['blue'], markeredgecolor='white', markeredgewidth=1)
        
        # Add horizontal line for Base to show it's only available at Full
        ax.axhline(y=0.604, color=OKABE_ITO_COLORS['blue'], linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_ylabel('mDice Score', fontweight='bold')
        ax.set_xlabel('Tamaño del Subset', fontweight='bold')
        ax.set_title('Curvas de Aprendizaje YOLO + SAM2\n(mDice vs Tamaño de Subset)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subset_labels)
        ax.set_ylim(0.55, 0.67)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(frameon=False, loc='lower right')
        
        # Add value labels
        for i, (ft_val, base_val) in enumerate(zip(ft_dice, base_dice)):
            if ft_val is not None:
                ax.text(i, ft_val + 0.003, f'{ft_val:.3f}', ha='center', va='bottom', fontsize=9)
            if base_val is not None:
                ax.text(i, base_val + 0.003, f'{base_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add annotation for Base (only Full available)
        ax.annotate('Solo disponible en Full', xy=(3, 0.604), xytext=(2.5, 0.62),
                   arrowprops=dict(arrowstyle='->', color=OKABE_ITO_COLORS['blue'], alpha=0.7),
                   fontsize=8, color=OKABE_ITO_COLORS['blue'], ha='center')
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_yolo_sam2_learning_curves.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_yolo_sam2_learning_curves.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_yolo_sam2_learning_curves.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ YOLO + SAM2 learning curves saved to {self.output_dir}")
    
    def create_unet_variants_comparison(self, save_format: str = "png") -> None:
        """
        Create grouped bar chart comparing U-Net variants (U-Net, U-Net++, DSC-U-Net)
        """
        # Data from the provided table
        models = ['U-Net', 'U-Net++', 'DSC-U-Net']
        iou_values = [0.557, 0.544, 0.559]
        dice_values = [0.692, 0.679, 0.693]
        f1_values = [0.692, 0.679, 0.693]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set bar width and positions
        bar_width = 0.25
        x = np.arange(len(models))
        
        # Create bars using consistent colors
        bars1 = ax.bar(x - bar_width, iou_values, bar_width, label='mIoU', 
                      color=SAM2_COLORS['iou'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x, dice_values, bar_width, label='mDice', 
                      color=SAM2_COLORS['f1'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars3 = ax.bar(x + bar_width, f1_values, bar_width, label='F1-Score', 
                      color=SAM2_COLORS['recall'], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('Variantes de U-Net', fontweight='bold')
        ax.set_ylabel('Valor de Métricas (0-1)', fontweight='bold')
        ax.set_title('Comparación de Variantes de U-Net\nmIoU, mDice y F1-Score', 
                    fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 0.8)
        ax.set_yticks(np.arange(0, 0.9, 0.1))
        ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        self._add_value_labels(ax, bars1, iou_values)
        self._add_value_labels(ax, bars2, dice_values)
        self._add_value_labels(ax, bars3, f1_values)
        
        # Add legend
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=11, ncol=3)
        
        # Tight layout and save
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_unet_variants_comparison.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_unet_variants_comparison.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_unet_variants_comparison.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ U-Net variants comparison saved to {self.output_dir}")
    
    def create_yolo_unet_comparison(self, save_format: str = "png") -> None:
        """
        Create grouped bar chart comparing YOLOv8 Segment vs U-Net variants
        """
        # Data from the provided tables
        models = ['YOLOv8 Segment', 'U-Net R34', 'U-Net R50', 'U-Net++', 'DSC-U-Net']
        
        # Using Full subset data for comparison (most complete)
        iou_values = [0.480, 0.540, 0.544, 0.544, 0.559]  # YOLOv8 Full, U-Net R34 2000, U-Net R50 2000, U-Net++, DSC-U-Net
        dice_values = [0.624, 0.676, 0.680, 0.679, 0.693]  # Same order
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set bar width and positions
        bar_width = 0.35
        x = np.arange(len(models))
        
        # Create bars using consistent colors
        bars1 = ax.bar(x - bar_width/2, iou_values, bar_width, label='mIoU', 
                      color=SAM2_COLORS['iou'], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + bar_width/2, dice_values, bar_width, label='mDice', 
                      color=SAM2_COLORS['f1'], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('Modelos de Segmentación', fontweight='bold')
        ax.set_ylabel('Valor de Métricas (0-1)', fontweight='bold')
        ax.set_title('Comparación YOLOv8 Segment vs Variantes de U-Net\nmIoU y mDice (Subset Full/2000)', 
                    fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 0.8)
        ax.set_yticks(np.arange(0, 0.9, 0.1))
        ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        self._add_value_labels(ax, bars1, iou_values)
        self._add_value_labels(ax, bars2, dice_values)
        
        # Add legend
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=11, ncol=2)
        
        # Tight layout and save
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in save_format.split(','):
            if fmt.lower() == 'pdf':
                plt.savefig(self.output_dir / f'{self.experiment}_yolo_unet_comparison.pdf', 
                           format='pdf', bbox_inches='tight', dpi=300)
            elif fmt.lower() == 'svg':
                plt.savefig(self.output_dir / f'{self.experiment}_yolo_unet_comparison.svg', 
                           format='svg', bbox_inches='tight')
            elif fmt.lower() == 'png':
                plt.savefig(self.output_dir / f'{self.experiment}_yolo_unet_comparison.png', 
                           format='png', bbox_inches='tight', dpi=600)
        
        plt.show()
        print(f"✓ YOLOv8 vs U-Net comparison saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize SAM2 evaluation results')
    parser.add_argument('--output_dir', type=str, default='/home/ptp/sam2/Plots',
                       help='Base output directory for visualizations')
    parser.add_argument('--experiment', type=str, default='sam2_base',
                       help='Experiment name (creates subfolder)')
    parser.add_argument('--format', type=str, default='png',
                       help='Output formats (comma-separated): pdf, svg, png')
    parser.add_argument('--plots', type=str, nargs='+', 
                       default=['prompts', 'model_size', 'prompt_type'],
                       choices=['prompts', 'model_size', 'prompt_type', 'finetuned_metrics', 'zeroshot_vs_finetuned', 'subset_performance', 'lora_prompt_comparison', 'faster_sam2_learning_curves', 'yolo_sam2_learning_curves', 'unet_variants_comparison', 'yolo_unet_comparison'],
                       help='Which plots to generate')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = SAM2Visualizer(args.output_dir, args.experiment)
    
    # Generate requested plots
    if 'prompts' in args.plots:
        print("Generating SAM2 prompts comparison...")
        viz.create_sam2_prompts_comparison(args.format)
    
    if 'model_size' in args.plots:
        print("Generating model size comparison...")
        viz.create_model_size_comparison(args.format)
    
    if 'prompt_type' in args.plots:
        print("Generating prompt type analysis...")
        viz.create_prompt_type_analysis(args.format)
    
    if 'finetuned_metrics' in args.plots:
        print("Generating fine-tuned metrics comparison...")
        viz.create_finetuned_metrics_comparison(args.format)
    
    if 'zeroshot_vs_finetuned' in args.plots:
        print("Generating zero-shot vs fine-tuned comparison...")
        viz.create_zeroshot_vs_finetuned_comparison(args.format)
    
    if 'subset_performance' in args.plots:
        print("Generating subset performance analysis...")
        viz.create_subset_performance_analysis(args.format)
    
    if 'lora_prompt_comparison' in args.plots:
        print("Generating SAM2+LoRA prompt modes comparison...")
        viz.create_sam2_lora_prompt_comparison(args.format)
    
    if 'faster_sam2_learning_curves' in args.plots:
        print("Generating Faster SAM2 learning curves...")
        viz.create_faster_sam2_learning_curves(args.format)
    
    if 'yolo_sam2_learning_curves' in args.plots:
        print("Generating YOLO + SAM2 learning curves...")
        viz.create_yolo_sam2_learning_curves(args.format)
    
    if 'unet_variants_comparison' in args.plots:
        print("Generating U-Net variants comparison...")
        viz.create_unet_variants_comparison(args.format)
    
    if 'yolo_unet_comparison' in args.plots:
        print("Generating YOLOv8 vs U-Net comparison...")
        viz.create_yolo_unet_comparison(args.format)
    
    print(f"\n✓ All visualizations completed and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
