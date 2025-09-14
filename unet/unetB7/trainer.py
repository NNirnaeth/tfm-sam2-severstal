"""
Training utilities for UNet with EfficientNet-B7
Handles training, validation, and model saving
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    ReduceLROnPlateau, 
    EarlyStopping,
    CSVLogger,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
from .model import UNetEfficientNetB7
from .data_utils import DataLoader, DataAugmentation, DataPreprocessor


class Trainer:
    """
    Trainer class for UNet with EfficientNet-B7
    Handles training, validation, and model management
    """
    
    def __init__(self, 
                 model: UNetEfficientNetB7,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 output_dir: str = "./outputs",
                 experiment_name: str = "unet_efficientnetb7"):
        """
        Initialize trainer
        
        Args:
            model: UNet model instance
            train_data: Training data loader
            val_data: Validation data loader
            output_dir: Output directory for saving models and logs
            experiment_name: Name of the experiment
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        # Create output directories
        self.model_dir = os.path.join(output_dir, experiment_name, "models")
        self.logs_dir = os.path.join(output_dir, experiment_name, "logs")
        self.plots_dir = os.path.join(output_dir, experiment_name, "plots")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Training history
        self.history = None
        
    def setup_callbacks(self, 
                       monitor: str = 'val_dice_coefficient',
                       mode: str = 'max',
                       patience: int = 20,
                       min_lr: float = 1e-7,
                       restore_best_weights: bool = True) -> List[tf.keras.callbacks.Callback]:
        """
        Setup training callbacks
        
        Args:
            monitor: Metric to monitor
            mode: 'max' or 'min' for the monitored metric
            patience: Patience for early stopping and LR reduction
            min_lr: Minimum learning rate
            restore_best_weights: Whether to restore best weights
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.model_dir, "best_model.h5")
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            save_freq='epoch'
        )
        callbacks.append(checkpoint)
        
        # Learning rate reduction
        lr_scheduler = ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=patience // 2,
            min_lr=min_lr,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # CSV logger
        csv_logger = CSVLogger(
            filename=os.path.join(self.logs_dir, "training_log.csv"),
            append=True
        )
        callbacks.append(csv_logger)
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=os.path.join(self.logs_dir, "tensorboard"),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def train(self, 
              epochs: int = 100,
              initial_learning_rate: float = 1e-4,
              loss: str = 'bce_dice',
              metrics: List[str] = None,
              class_weights: Optional[Dict[int, float]] = None,
              use_cosine_schedule: bool = True,
              warmup_epochs: int = 5) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            initial_learning_rate: Initial learning rate
            loss: Loss function to use
            metrics: List of metrics to track
            class_weights: Class weights for handling imbalance
            use_cosine_schedule: Whether to use cosine learning rate schedule
            warmup_epochs: Number of warmup epochs
            
        Returns:
            Training history dictionary
        """
        if metrics is None:
            metrics = ['accuracy']
        
        # Compile model
        self.model.compile_model(
            learning_rate=initial_learning_rate,
            loss=loss,
            metrics=metrics
        )
        
        # Setup learning rate schedule (matching IPYNB)
        if use_cosine_schedule:
            total_steps = len(self.train_data) * epochs
            
            lr_schedule = ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=total_steps // 4,
                decay_rate=0.9,
                staircase=True
            )
            
            # Override optimizer with scheduled learning rate
            self.model.model.optimizer = Adam(learning_rate=lr_schedule)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Print model summary
        print("Model Summary:")
        print(self.model.get_model_summary())
        
        # Print training configuration
        print(f"\nTraining Configuration:")
        print(f"Epochs: {epochs}")
        print(f"Initial LR: {initial_learning_rate}")
        print(f"Loss: {loss}")
        print(f"Metrics: {metrics}")
        print(f"Train samples: {len(self.train_data) * self.train_data.batch_size}")
        print(f"Val samples: {len(self.val_data) * self.val_data.batch_size}")
        
        # Train model
        print("\nStarting training...")
        # For segmentation tasks, we'll skip class weights as they can cause issues
        # The loss function (bce_dice) already handles class imbalance well
        self.history = self.model.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, "final_model.h5")
        self.model.save_model(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save training configuration
        config = {
            'epochs': epochs,
            'initial_learning_rate': initial_learning_rate,
            'loss': loss,
            'metrics': metrics,
            'class_weights': class_weights,
            'use_cosine_schedule': use_cosine_schedule,
            'warmup_epochs': warmup_epochs,
            'input_shape': self.model.input_shape,
            'num_classes': self.model.num_classes
        }
        
        config_path = os.path.join(self.output_dir, self.experiment_name, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return self.history.history
    
    def plot_training_history(self, save_plots: bool = True):
        """
        Plot training history
        
        Args:
            save_plots: Whether to save plots to disk
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        history = self.history.history
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot accuracy
        if 'accuracy' in history:
            axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history:
                axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot dice coefficient
        if 'dice_coefficient' in history:
            axes[1, 0].plot(history['dice_coefficient'], label='Training Dice')
            if 'val_dice_coefficient' in history:
                axes[1, 0].plot(history['val_dice_coefficient'], label='Validation Dice')
            axes[1, 0].set_title('Dice Coefficient')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Dice Coefficient')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot learning rate
        if hasattr(self.model.model.optimizer, 'learning_rate'):
            lr_history = []
            for epoch in range(len(history['loss'])):
                if hasattr(self.model.model.optimizer.learning_rate, 'numpy'):
                    lr = self.model.model.optimizer.learning_rate.numpy()
                else:
                    lr = self.model.model.optimizer.learning_rate
                lr_history.append(lr)
            
            axes[1, 1].plot(lr_history)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.plots_dir, "training_history.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to: {plot_path}")
        
        plt.show()
    
    def evaluate_model(self, test_data: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model on test data...")
        
        # Load best model
        best_model_path = os.path.join(self.model_dir, "best_model.h5")
        if os.path.exists(best_model_path):
            self.model.load_model(best_model_path)
            print(f"Loaded best model from: {best_model_path}")
        
        # Evaluate
        results = self.model.model.evaluate(test_data, verbose=1)
        
        # Create results dictionary
        metrics_names = self.model.model.metrics_names
        evaluation_results = dict(zip(metrics_names, results))
        
        print("\nEvaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}")
        
        return evaluation_results
    
    def predict_batch(self, data_loader: DataLoader, num_samples: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of data
        
        Args:
            data_loader: Data loader for prediction
            num_samples: Number of samples to predict
            
        Returns:
            Tuple of (images, true_masks, predicted_masks)
        """
        # Get a batch
        images, true_masks = data_loader[0]
        
        # Make predictions
        predictions = self.model.model.predict(images[:num_samples])
        
        return images[:num_samples], true_masks[:num_samples], predictions
    
    def save_predictions_visualization(self, 
                                     test_data: DataLoader, 
                                     num_samples: int = 8,
                                     threshold: float = 0.5):
        """
        Save visualization of predictions
        
        Args:
            test_data: Test data loader
            num_samples: Number of samples to visualize
            threshold: Threshold for binary predictions
        """
        # Get predictions
        images, true_masks, predictions = self.predict_batch(test_data, num_samples)
        
        # Apply threshold to predictions
        binary_predictions = (predictions > threshold).astype(np.float32)
        
        # Create visualization
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(images[i])
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            # Ground truth
            axes[1, i].imshow(true_masks[i].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Ground Truth {i+1}')
            axes[1, i].axis('off')
            
            # Prediction
            axes[2, i].imshow(binary_predictions[i].squeeze(), cmap='gray')
            axes[2, i].set_title(f'Prediction {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.plots_dir, "predictions_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to: {viz_path}")
        
        plt.show()
