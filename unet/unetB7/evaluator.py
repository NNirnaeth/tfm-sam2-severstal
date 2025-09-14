"""
Evaluation utilities for UNet with EfficientNet-B7
Provides comprehensive evaluation metrics and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Tuple, Optional
import cv2
from .data_utils import DataPreprocessor


class Evaluator:
    """
    Comprehensive evaluator for segmentation models
    Provides various metrics and visualizations
    """
    
    def __init__(self, model, class_names: List[str] = None):
        """
        Initialize evaluator
        
        Args:
            model: Trained UNet model
            class_names: List of class names for visualization
        """
        self.model = model
        self.class_names = class_names or ['Background', 'Defect']
        
    def dice_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-5) -> float:
        """
        Calculate Dice coefficient
        
        Args:
            y_true: Ground truth mask
            y_pred: Predicted mask
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    def iou_score(self, y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-5) -> float:
        """
        Calculate Intersection over Union (IoU)
        
        Args:
            y_true: Ground truth mask
            y_pred: Predicted mask
            smooth: Smoothing factor
            
        Returns:
            IoU score
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def pixel_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate pixel accuracy
        
        Args:
            y_true: Ground truth mask
            y_pred: Predicted mask
            
        Returns:
            Pixel accuracy
        """
        return np.mean(y_true == y_pred)
    
    def precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score
        
        Args:
            y_true: Ground truth mask
            y_pred: Predicted mask
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Flatten arrays
        y_true_f = y_true_binary.flatten()
        y_pred_f = y_pred_binary.flatten()
        
        # Calculate TP, FP, FN, TN
        tp = np.sum((y_true_f == 1) & (y_pred_f == 1))
        fp = np.sum((y_true_f == 0) & (y_pred_f == 1))
        fn = np.sum((y_true_f == 1) & (y_pred_f == 0))
        tn = np.sum((y_true_f == 0) & (y_pred_f == 0))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    def evaluate_batch(self, images: np.ndarray, true_masks: np.ndarray, 
                      predictions: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate a batch of predictions
        
        Args:
            images: Batch of input images
            true_masks: Batch of ground truth masks
            predictions: Batch of predicted masks
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with average metrics
        """
        batch_size = len(images)
        metrics = {
            'dice_coefficient': [],
            'iou_score': [],
            'pixel_accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for i in range(batch_size):
            true_mask = true_masks[i].squeeze()
            pred_mask = predictions[i].squeeze()
            
            # Calculate metrics
            dice = self.dice_coefficient(true_mask, pred_mask)
            iou = self.iou_score(true_mask, pred_mask)
            pixel_acc = self.pixel_accuracy(true_mask, (pred_mask > threshold).astype(int))
            prf = self.precision_recall_f1(true_mask, pred_mask, threshold)
            
            # Store metrics
            metrics['dice_coefficient'].append(dice)
            metrics['iou_score'].append(iou)
            metrics['pixel_accuracy'].append(pixel_acc)
            metrics['precision'].append(prf['precision'])
            metrics['recall'].append(prf['recall'])
            metrics['f1_score'].append(prf['f1_score'])
        
        # Calculate averages
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return avg_metrics
    
    def evaluate_dataset(self, data_loader, threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model on entire dataset
        
        Args:
            data_loader: Data loader for evaluation
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_metrics = {
            'dice_coefficient': [],
            'iou_score': [],
            'pixel_accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        print("Evaluating dataset...")
        for batch_idx in range(len(data_loader)):
            images, true_masks = data_loader[batch_idx]
            predictions = self.model.predict(images)
            
            batch_metrics = self.evaluate_batch(images, true_masks, predictions, threshold)
            
            for metric, value in batch_metrics.items():
                all_metrics[metric].append(value)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Calculate final averages
        final_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        return final_metrics
    
    def plot_metrics_distribution(self, data_loader, threshold: float = 0.5, 
                                 save_path: Optional[str] = None):
        """
        Plot distribution of metrics across the dataset
        
        Args:
            data_loader: Data loader for evaluation
            threshold: Threshold for binary classification
            save_path: Path to save the plot
        """
        all_metrics = {
            'dice_coefficient': [],
            'iou_score': [],
            'pixel_accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        print("Calculating metrics distribution...")
        for batch_idx in range(len(data_loader)):
            images, true_masks = data_loader[batch_idx]
            predictions = self.model.predict(images)
            
            batch_metrics = self.evaluate_batch(images, true_masks, predictions, threshold)
            
            for metric, value in batch_metrics.items():
                all_metrics[metric].append(value)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, values) in enumerate(all_metrics.items()):
            axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.3f}')
            axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', 
                          label=f'±1σ: {std_val:.3f}')
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, data_loader, threshold: float = 0.5, 
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix for pixel-wise classification
        
        Args:
            data_loader: Data loader for evaluation
            threshold: Threshold for binary classification
            save_path: Path to save the plot
        """
        all_true = []
        all_pred = []
        
        print("Calculating confusion matrix...")
        for batch_idx in range(len(data_loader)):
            images, true_masks = data_loader[batch_idx]
            predictions = self.model.predict(images)
            
            for i in range(len(images)):
                true_mask = true_masks[i].squeeze()
                pred_mask = predictions[i].squeeze()
                
                # Flatten and binarize
                true_flat = (true_mask > threshold).astype(int).flatten()
                pred_flat = (pred_mask > threshold).astype(int).flatten()
                
                all_true.extend(true_flat)
                all_pred.extend(pred_flat)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_true, all_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def visualize_predictions(self, data_loader, num_samples: int = 6, 
                            threshold: float = 0.5, save_path: Optional[str] = None):
        """
        Visualize model predictions with ground truth
        
        Args:
            data_loader: Data loader for visualization
            num_samples: Number of samples to visualize
            threshold: Threshold for binary classification
            save_path: Path to save the plot
        """
        # Get a batch
        images, true_masks = data_loader[0]
        predictions = self.model.predict(images[:num_samples])
        
        # Create visualization
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            true_mask = true_masks[i].squeeze()
            pred_mask = predictions[i].squeeze()
            binary_pred = (pred_mask > threshold).astype(np.float32)
            
            # Calculate metrics for this sample
            dice = self.dice_coefficient(true_mask, pred_mask)
            iou = self.iou_score(true_mask, pred_mask)
            
            # Original image
            axes[0, i].imshow(images[i])
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            # Ground truth
            axes[1, i].imshow(true_mask, cmap='gray')
            axes[1, i].set_title(f'Ground Truth {i+1}')
            axes[1, i].axis('off')
            
            # Prediction
            axes[2, i].imshow(binary_pred, cmap='gray')
            axes[2, i].set_title(f'Prediction {i+1}\nDice: {dice:.3f}, IoU: {iou:.3f}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions visualization saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, data_loader, threshold: float = 0.5, 
                       save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Generate comprehensive evaluation report
        
        Args:
            data_loader: Data loader for evaluation
            threshold: Threshold for binary classification
            save_path: Path to save the report
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Generating comprehensive evaluation report...")
        
        # Evaluate dataset
        metrics = self.evaluate_dataset(data_loader, threshold)
        
        # Print report
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(f"Threshold: {threshold}")
        print(f"Number of samples: {len(data_loader) * data_loader.batch_size}")
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write("EVALUATION REPORT\n")
                f.write("="*50 + "\n")
                f.write(f"Threshold: {threshold}\n")
                f.write(f"Number of samples: {len(data_loader) * data_loader.batch_size}\n")
                f.write("\nMetrics:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric.replace('_', ' ').title()}: {value:.4f}\n")
            print(f"Report saved to: {save_path}")
        
        return metrics
