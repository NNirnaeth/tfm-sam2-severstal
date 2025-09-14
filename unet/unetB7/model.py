"""
UNet with EfficientNet-B7 Architecture
Optimized for high-resolution images (256x1600)
Based on: https://github.com/vipinkarthikeyan/UNet-with-EfficientNet-Encoder/blob/main/EfficientNetB7_UNet.ipynb
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import Metric
import numpy as np


def mish(x):
    """Mish activation function"""
    return x * tf.nn.tanh(tf.nn.softplus(x))


def dice_coefficient(y_true, y_pred):
    """Dice coefficient metric for segmentation"""
    smooth = 1e-5
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)


class DiceCoefficient(Metric):
    """Dice coefficient metric for segmentation"""
    
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.smooth = 1e-5
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
    
    def result(self):
        return (2. * self.intersection + self.smooth) / (self.union + self.smooth)
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class UNetEfficientNetB7:
    """
    UNet model with EfficientNet-B7 as encoder
    Designed for high-resolution image segmentation (256x1600)
    Based on the original IPYNB implementation
    """
    
    def __init__(self, input_shape=(256, 1600, 3), num_classes=1, 
                 pretrained_weights='imagenet', freeze_encoder_layers=10):
        """
        Initialize UNet with EfficientNet-B7
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            pretrained_weights: Pretrained weights for EfficientNet
            freeze_encoder_layers: Number of encoder layers to freeze
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights
        self.freeze_encoder_layers = freeze_encoder_layers
        self.model = None
        
    def build_model(self):
        """Build the UNet model with EfficientNet-B7 encoder"""
        
        # Encoder (EfficientNet-B7)
        try:
            encoder = EfficientNetB7(
                weights=self.pretrained_weights,
                include_top=False,
                input_shape=self.input_shape
            )
        except ValueError as e:
            if "Shape mismatch" in str(e) and self.pretrained_weights == 'imagenet':
                print(f"Warning: Could not load ImageNet pretrained weights due to shape mismatch: {e}")
                print("Falling back to random initialization...")
                encoder = EfficientNetB7(
                    weights=None,
                    include_top=False,
                    input_shape=self.input_shape
                )
            else:
                raise e
        
        # Freeze early layers to prevent overfitting
        for layer in encoder.layers[:-self.freeze_encoder_layers]:
            layer.trainable = False
            
        # Get skip connections from different levels (matching IPYNB)
        skip_connections = []
        skip_layer_names = [
            'block2a_expand_activation',  # 1/4 resolution
            'block3a_expand_activation',  # 1/8 resolution  
            'block4a_expand_activation',  # 1/16 resolution
            'block5a_expand_activation',  # 1/32 resolution
            'block6a_expand_activation',  # 1/64 resolution
        ]
        
        for layer in encoder.layers:
            if layer.name in skip_layer_names:
                skip_connections.append(layer.output)
        
        # Decoder (UNet) - matching IPYNB implementation
        x = encoder.output
        
        # Upsampling and concatenation with skip connections
        decoder_filters = [512, 256, 128, 64, 32]
        
        for i, (skip, filters) in enumerate(zip(reversed(skip_connections), decoder_filters)):
            # Upsampling using Conv2DTranspose (matching IPYNB)
            x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
            
            # Skip connection
            if skip is not None:
                # Resize skip connection to match current upsampled feature map dimensions
                skip_resized = layers.Lambda(
                    lambda inputs: tf.image.resize(inputs[0], tf.shape(inputs[1])[1:3], method='bilinear')
                )([skip, x])
                x = layers.Concatenate()([x, skip_resized])
            
            # Convolutional blocks with Mish activation (matching IPYNB)
            x = layers.Conv2D(filters, 3, activation=mish, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, 3, activation=mish, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Dropout for regularization
            if i < len(decoder_filters) - 1:
                x = layers.Dropout(0.2)(x)
        
        # The decoder loop already handles the upsampling to reach the target resolution
        
        # Output layer
        if self.num_classes == 1:
            outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
        else:
            outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(x)
        
        self.model = Model(inputs=encoder.input, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=1e-4, loss='dice_loss', metrics=None):
        """
        Compile the model with optimizer, loss and metrics
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            loss: Loss function ('dice_loss', 'bce_dice', 'focal_loss')
            metrics: List of metrics to track
        """
        if self.model is None:
            self.build_model()
            
        if metrics is None:
            metrics = ['accuracy']
            
        # Custom loss function
        if loss == 'dice_loss':
            loss_fn = self._dice_loss
        elif loss == 'bce_dice':
            loss_fn = self._bce_dice_loss
        elif loss == 'focal_loss':
            loss_fn = self._focal_loss
        else:
            loss_fn = loss
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=metrics
        )
        
    def _dice_loss(self, y_true, y_pred):
        """Dice loss for segmentation"""
        smooth = 1e-5
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def _bce_dice_loss(self, y_true, y_pred):
        """Combined Binary Cross Entropy and Dice loss"""
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = self._dice_loss(y_true, y_pred)
        return bce + dice
    
    def _focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        focal_loss = focal_weight * tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(focal_loss)
    
    def _dice_coefficient(self, y_true, y_pred):
        """Dice coefficient metric for segmentation"""
        smooth = 1e-5
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def dice_coefficient(self, y_true, y_pred):
        """Dice coefficient metric"""
        smooth = 1e-5
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={
                'dice_loss': self._dice_loss,
                'bce_dice_loss': self._bce_dice_loss,
                'focal_loss': self._focal_loss,
                'dice_coefficient': self.dice_coefficient,
                'mish': mish
            }
        )
        return self.model
