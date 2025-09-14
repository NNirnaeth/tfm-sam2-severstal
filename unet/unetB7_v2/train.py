import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import Sequence

# ============================================================================
# 1. ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train UNet with EfficientNet-B7 for Steel Defect Detection')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./severstal-steel-defect-detection/',
                       help='Path to the dataset directory')
    parser.add_argument('--train_csv', type=str, default='train.csv',
                       help='Name of the training CSV file')
    parser.add_argument('--train_images_dir', type=str, default='train_images',
                       help='Name of the training images directory')
    
    # Model arguments
    parser.add_argument('--img_height', type=int, default=512,
                       help='Input image height')
    parser.add_argument('--img_width', type=int, default=512,
                       help='Input image width')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--experiment_name', type=str, default='steel_unet_efficientb7',
                       help='Name of the experiment')
    parser.add_argument('--model_name', type=str, default='best_steel_unet_efficientb7.h5',
                       help='Name for the saved model')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--test_images_dir', type=str, default='test_images',
                       help='Name of the test images directory')
    parser.add_argument('--submission_file', type=str, default='submission.csv',
                       help='Name of the submission file')
    
    return parser.parse_args()

# ============================================================================
# 2. CONFIGURACIÓN Y PARÁMETROS
# ============================================================================

# Configurar GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ============================================================================
# 3. FUNCIONES DE UTILIDAD PARA RLE (Run-Length Encoding)
# ============================================================================

def rle2mask(rle, shape):
    """Convierte RLE a máscara binaria"""
    if pd.isna(rle):
        return np.zeros(shape, dtype=np.uint8)
    
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask2rle(img):
    """Convierte máscara binaria a RLE"""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# ============================================================================
# 4. GENERADOR DE DATOS PERSONALIZADO
# ============================================================================

class SteelDataGenerator(Sequence):
    def __init__(self, df, data_path, batch_size=8, img_size=(512, 512), 
                 n_classes=4, shuffle=True, mode='train'):
        self.df = df
        self.data_path = data_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mode = mode
        self.indexes = np.arange(len(df))
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.df) // self.batch_size
        
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.df.iloc[k]['ImageId'] for k in indexes]
        
        X, y = self.__data_generation(batch_ids)
        return X, y
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, batch_ids):
        X = np.empty((self.batch_size, *self.img_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, *self.img_size, self.n_classes), dtype=np.float32)
        
        for i, img_id in enumerate(batch_ids):
            # Cargar imagen
            img_path = os.path.join(self.data_path, 'train_images', img_id)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            X[i,] = img
            
            # Crear máscaras para cada clase
            masks = np.zeros((*self.img_size, self.n_classes), dtype=np.float32)
            
            for class_id in range(1, self.n_classes + 1):
                class_df = self.df[(self.df['ImageId'] == img_id) & 
                                 (self.df['ClassId'] == class_id)]
                
                if len(class_df) > 0 and not pd.isna(class_df.iloc[0]['EncodedPixels']):
                    # Crear máscara desde RLE
                    rle = class_df.iloc[0]['EncodedPixels']
                    mask = rle2mask(rle, (256, 1600))  # Tamaño original
                    
                    # Redimensionar máscara
                    mask = cv2.resize(mask.astype(np.uint8), self.img_size, 
                                    interpolation=cv2.INTER_NEAREST)
                    masks[:, :, class_id - 1] = mask
            
            y[i,] = masks
            
        return X, y

# ============================================================================
# 5. MODELO U-NET CON EFFICIENTNET-B7
# ============================================================================

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet_efficientnetb7(input_shape=(512, 512, 3), num_classes=4):
    """Construir U-Net con EfficientNet-B7 como encoder"""
    
    inputs = Input(input_shape)
    
    # Encoder: EfficientNet-B7
    encoder = EfficientNetB7(
        input_tensor=inputs,
        weights='imagenet',
        include_top=False
    )
    
    # Extraer skip connections en diferentes niveles
    skip_connections = [
        encoder.get_layer('block2a_expand_activation').output,   # 128x128
        encoder.get_layer('block3a_expand_activation').output,   # 64x64
        encoder.get_layer('block4a_expand_activation').output,   # 32x32
        encoder.get_layer('block6a_expand_activation').output,   # 16x16
    ]
    
    # Bottleneck (punto más profundo)
    bottleneck = encoder.output  # 8x8
    
    # Decoder
    d1 = decoder_block(bottleneck, skip_connections[3], 512)  # 16x16
    d2 = decoder_block(d1, skip_connections[2], 256)          # 32x32  
    d3 = decoder_block(d2, skip_connections[1], 128)          # 64x64
    d4 = decoder_block(d3, skip_connections[0], 64)           # 128x128
    
    # Upsampling final
    d5 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(d4)  # 256x256
    
    # Si usas 512x512, necesitas otro upsampling
    if input_shape[0] == 512:
        d5 = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(d5)  # 512x512
    
    # Capa de salida
    outputs = Conv2D(num_classes, 1, activation='sigmoid', name='output')(d5)
    
    model = Model(inputs, outputs)
    return model

# ============================================================================
# 6. MÉTRICAS Y PÉRDIDAS
# ============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Coeficiente Dice para segmentación"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """BCE + Dice Loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# ============================================================================
# 7. PREPARACIÓN DE DATOS
# ============================================================================

def prepare_data(data_path, train_csv, val_split, random_seed):
    """Preparar datos del dataset Severstal"""
    
    # Leer CSV
    train_df = pd.read_csv(os.path.join(data_path, train_csv))
    
    # Llenar NaN con string vacío
    train_df['EncodedPixels'] = train_df['EncodedPixels'].fillna('')
    
    # Obtener lista única de imágenes
    unique_images = train_df['ImageId'].unique()
    
    # Crear DataFrame con una fila por imagen
    image_df = pd.DataFrame({'ImageId': unique_images})
    
    # Agregar información de si tiene defectos
    has_defect = train_df[train_df['EncodedPixels'] != '']['ImageId'].unique()
    image_df['has_defect'] = image_df['ImageId'].isin(has_defect).astype(int)
    
    # Split train/validation
    train_imgs, val_imgs = train_test_split(
        image_df, 
        test_size=val_split, 
        random_state=random_seed,
        stratify=image_df['has_defect']
    )
    
    # Crear DataFrames finales
    train_data = train_df[train_df['ImageId'].isin(train_imgs['ImageId'])]
    val_data = train_df[train_df['ImageId'].isin(val_imgs['ImageId'])]
    
    print(f"Imágenes de entrenamiento: {len(train_imgs)}")
    print(f"Imágenes de validación: {len(val_imgs)}")
    print(f"Total de anotaciones de entrenamiento: {len(train_data)}")
    print(f"Total de anotaciones de validación: {len(val_data)}")
    
    return train_data, val_data

# ============================================================================
# 8. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def train_model(args):
    """Función principal de entrenamiento"""
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Preparar datos
    print("Preparando datos...")
    train_data, val_data = prepare_data(
        args.data_path, 
        args.train_csv, 
        args.val_split, 
        args.random_seed
    )
    
    # Crear generadores
    print("Creando generadores de datos...")
    train_gen = SteelDataGenerator(
        train_data, 
        args.data_path,
        batch_size=args.batch_size,
        img_size=(args.img_height, args.img_width),
        n_classes=args.num_classes,
        mode='train'
    )
    
    val_gen = SteelDataGenerator(
        val_data,
        args.data_path, 
        batch_size=args.batch_size,
        img_size=(args.img_height, args.img_width),
        n_classes=args.num_classes,
        mode='val',
        shuffle=False
    )
    
    # Crear modelo
    print("Creando modelo U-Net con EfficientNet-B7...")
    model = build_unet_efficientnetb7(
        input_shape=(args.img_height, args.img_width, 3),
        num_classes=args.num_classes
    )
    
    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )
    
    # Mostrar resumen
    print(model.summary())
    
    # Callbacks
    model_path = os.path.join(experiment_dir, args.model_name)
    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor='val_dice_coefficient',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Entrenar
    print("Iniciando entrenamiento...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ============================================================================
# 9. PREDICCIÓN Y EVALUACIÓN
# ============================================================================

def predict_test_images(model, test_path, output_path, img_height, img_width, num_classes):
    """Generar predicciones para imágenes de test"""
    
    test_images = os.listdir(test_path)
    predictions = []
    
    for img_name in tqdm(test_images):
        # Cargar imagen
        img_path = os.path.join(test_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predicción
        pred_mask = model.predict(img)[0]
        
        # Procesar cada clase
        for class_id in range(1, num_classes + 1):
            mask = pred_mask[:, :, class_id - 1]
            
            # Aplicar threshold
            mask = (mask > 0.5).astype(np.uint8)
            
            # Redimensionar a tamaño original si es necesario
            if mask.shape != (256, 1600):
                mask = cv2.resize(mask, (1600, 256), interpolation=cv2.INTER_NEAREST)
            
            # Convertir a RLE
            rle = mask2rle(mask)
            
            predictions.append({
                'ImageId_ClassId': f"{img_name}_{class_id}",
                'EncodedPixels': rle if rle != '1 0' else ''
            })
    
    # Crear submission
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission guardado en {output_path}")

# ============================================================================
# 10. VISUALIZACIÓN
# ============================================================================

def plot_training_history(history, save_path):
    """Visualizar historial de entrenamiento"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0,0].plot(history.history['loss'], label='Train Loss')
    axes[0,0].plot(history.history['val_loss'], label='Val Loss')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()
    
    # Accuracy
    axes[0,1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0,1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0,1].set_title('Accuracy')
    axes[0,1].legend()
    
    # Dice Coefficient
    axes[1,0].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[1,0].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[1,0].set_title('Dice Coefficient')
    axes[1,0].legend()
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1,1].plot(history.history['lr'])
        axes[1,1].set_title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# ============================================================================
# 11. MAIN
# ============================================================================

def main():
    """Main function"""
    args = parse_args()
    
    # Print configuration
    print("=== Severstal Steel Defect Detection - U-Net EfficientNet-B7 ===")
    print("Training Configuration:")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("="*50)
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Entrenar modelo
    model, history = train_model(args)
    
    # Visualizar resultados
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    plot_path = os.path.join(experiment_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Opcional: Generar predicciones en test
    if args.evaluate:
        test_path = os.path.join(args.data_path, args.test_images_dir)
        if os.path.exists(test_path):
            submission_path = os.path.join(experiment_dir, args.submission_file)
            predict_test_images(
                model, 
                test_path, 
                submission_path,
                args.img_height,
                args.img_width,
                args.num_classes
            )
        else:
            print(f"Test directory not found: {test_path}")
    
    print(f"¡Entrenamiento completado! Resultados guardados en: {experiment_dir}")

if __name__ == "__main__":
    main()