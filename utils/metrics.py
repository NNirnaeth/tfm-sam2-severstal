#!/usr/bin/env python3
"""
Unified metrics calculation for segmentation evaluation
Computes all standard metrics: mIoU, IoU@thresholds, Dice, Precision, Recall, F1, Accuracy
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

class SegmentationMetrics:
    """Unified metrics calculator for segmentation tasks"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Standard IoU thresholds
        self.iou_thresholds = [0.5, 0.75, 0.9, 0.95]
        
    def compute_pixel_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """Compute pixel-level metrics"""
        # Ensure binary masks
        pred_binary = (pred_mask > 0).astype(bool)
        gt_binary = (gt_mask > 0).astype(bool)
        
        # Intersection and Union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # IoU
        iou = intersection / (union + 1e-8)
        
        # Pixel-level metrics
        tp = intersection
        fp = pred_binary.sum() - intersection
        fn = gt_binary.sum() - intersection
        tn = (~pred_binary & ~gt_binary).sum()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Accuracy (pixel accuracy)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Dice Coefficient (F1-score for segments)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        return {
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'dice': float(dice),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    def compute_iou_at_thresholds(self, ious: List[float]) -> Dict[str, float]:
        """Compute IoU at different thresholds"""
        ious_array = np.array(ious)
        results = {}
        
        for threshold in self.iou_thresholds:
            threshold_percentage = threshold * 100
            key = f'iou_at_{int(threshold_percentage)}'
            results[key] = float((ious_array >= threshold).mean() * 100)
        
        return results
    
    def compute_benevolent_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """Compute benevolent metrics (partial overlap acceptance)"""
        pred_binary = (pred_mask > 0).astype(bool)
        gt_binary = (gt_mask > 0).astype(bool)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        gt_area = gt_binary.sum()
        
        # Benevolent: prediction covers ≥75% of GT
        benevolent_75 = (intersection / (gt_area + 1e-8)) >= 0.75
        
        # Benevolent: any overlap
        benevolent_any = intersection > 0
        
        # Coverage ratio
        coverage_ratio = intersection / (gt_area + 1e-8)
        
        return {
            'benevolent_75': float(benevolent_75),
            'benevolent_any': float(benevolent_any),
            'coverage_ratio': float(coverage_ratio)
        }
    
    def evaluate_batch(self, predictions: List[np.ndarray], ground_truths: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate a batch of predictions vs ground truths"""
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths must match")
        
        batch_metrics = []
        ious = []
        
        for pred, gt in zip(predictions, ground_truths):
            # Compute metrics for this sample
            sample_metrics = self.compute_pixel_metrics(pred, gt)
            benevolent_metrics = self.compute_benevolent_metrics(pred, gt)
            
            # Combine all metrics
            combined_metrics = {**sample_metrics, **benevolent_metrics}
            batch_metrics.append(combined_metrics)
            
            # Collect IoU for threshold analysis
            ious.append(sample_metrics['iou'])
        
        # Aggregate metrics across batch
        aggregated = self.aggregate_metrics(batch_metrics)
        
        # Add IoU@thresholds
        iou_threshold_metrics = self.compute_iou_at_thresholds(ious)
        aggregated.update(iou_threshold_metrics)
        
        return aggregated
    
    def aggregate_metrics(self, batch_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across a batch of samples"""
        if not batch_metrics:
            return {}
        
        # Get all metric names
        metric_names = batch_metrics[0].keys()
        
        aggregated = {}
        for metric in metric_names:
            values = [sample[metric] for sample in batch_metrics]
            
            # Mean
            aggregated[f'mean_{metric}'] = float(np.mean(values))
            
            # Std
            aggregated[f'std_{metric}'] = float(np.std(values))
            
            # Min/Max
            aggregated[f'min_{metric}'] = float(np.min(values))
            aggregated[f'max_{metric}'] = float(np.max(values))
        
        return aggregated
    
    def save_results_to_csv(self, results: Dict[str, float], experiment_name: str, 
                           model_info: Dict[str, str], additional_info: Optional[Dict] = None) -> str:
        """Save results to CSV with standardized format"""
        
        # Prepare data for CSV
        csv_data = {
            'exp_id': experiment_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **model_info,
            **results
        }
        
        # Add additional info if provided
        if additional_info:
            csv_data.update(additional_info)
        
        # Create CSV filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        csv_filename = f"{experiment_name}_{timestamp}.csv"
        csv_path = os.path.join(self.save_dir, csv_filename)
        
        # Convert to DataFrame and save
        df = pd.DataFrame([csv_data])
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to: {csv_path}")
        return csv_path
    
    def create_experiment_summary(self, all_results: List[Dict[str, float]], 
                                 experiment_name: str) -> str:
        """Create a summary CSV with all experiments"""
        
        # Prepare summary data
        summary_data = []
        for result in all_results:
            summary_data.append(result)
        
        # Create summary CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        summary_filename = f"summary_{experiment_name}_{timestamp}.csv"
        summary_path = os.path.join(self.save_dir, summary_filename)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_path, index=False)
        
        print(f"Summary saved to: {summary_path}")
        return summary_path
    
    def print_metrics_summary(self, results: Dict[str, float], model_name: str):
        """Print a formatted summary of metrics"""
        print(f"\n{'='*60}")
        print(f"METRICS SUMMARY: {model_name}")
        print(f"{'='*60}")
        
        # Main metrics
        print(f"mIoU: {results.get('mean_iou', 0):.4f}")
        print(f"Dice Coefficient: {results.get('mean_dice', 0):.4f}")
        print(f"F1-Score: {results.get('mean_f1', 0):.4f}")
        print(f"Precision: {results.get('mean_precision', 0):.4f}")
        print(f"Recall: {results.get('mean_recall', 0):.4f}")
        print(f"Accuracy: {results.get('mean_accuracy', 0):.4f}")
        
        # IoU@thresholds
        print(f"\nIoU@Thresholds:")
        for threshold in self.iou_thresholds:
            threshold_percentage = int(threshold * 100)
            key = f'iou_at_{threshold_percentage}'
            value = results.get(key, 0)
            print(f"  IoU@{threshold_percentage}: {value:.2f}%")
        
        # Benevolent metrics
        print(f"\nBenevolent Metrics:")
        print(f"  Coverage ≥75%: {results.get('mean_benevolent_75', 0):.4f}")
        print(f"  Any Overlap: {results.get('mean_benevolent_any', 0):.4f}")
        print(f"  Mean Coverage: {results.get('mean_coverage_ratio', 0):.4f}")
        
        print(f"{'='*60}")

# Example usage functions
def evaluate_single_prediction(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                             model_name: str = "unknown") -> Dict[str, float]:
    """Quick evaluation of a single prediction"""
    metrics = SegmentationMetrics()
    results = metrics.compute_pixel_metrics(pred_mask, gt_mask)
    benevolent = metrics.compute_benevolent_metrics(pred_mask, gt_mask)
    
    # Combine results
    combined_results = {**results, **benevolent}
    
    # Print summary
    metrics.print_metrics_summary(combined_results, model_name)
    
    return combined_results

def evaluate_batch_predictions(predictions: List[np.ndarray], 
                             ground_truths: List[np.ndarray],
                             experiment_name: str,
                             model_info: Dict[str, str],
                             save_results: bool = True) -> Dict[str, float]:
    """Evaluate a batch of predictions and optionally save results"""
    metrics = SegmentationMetrics()
    
    # Evaluate batch
    results = metrics.evaluate_batch(predictions, ground_truths)
    
    # Print summary
    model_name = model_info.get('model', 'Unknown')
    metrics.print_metrics_summary(results, model_name)
    
    # Save to CSV if requested
    if save_results:
        csv_path = metrics.save_results_to_csv(results, experiment_name, model_info)
        print(f"Results saved to: {csv_path}")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("SegmentationMetrics module loaded successfully!")
    print("Use the functions to evaluate your segmentation results.")
