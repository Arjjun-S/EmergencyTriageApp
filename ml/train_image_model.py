"""
Image Classification Model Training Script
Trains a CNN or Vision Transformer model on skin disease dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    'dataset_path': '../datasets/SkinDisease',
    'model_output_path': '../app/src/main/assets/',
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'model_type': 'cnn',  # 'cnn' or 'vit'
    'use_pretrained': True,
    'fine_tune_layers': 50
}

# Disease classes (22 categories)
DISEASE_CLASSES = [
    'Acne', 'Actinic_Keratosis', 'Benign_Tumor', 'Bullous', 'Candidiasis',
    'Drug_Eruption', 'Eczema', 'Infestation_Bite', 'Lichen', 'Lupus',
    'Moles', 'Psoriasis', 'Rosacea', 'Seborrh_Keratosis', 'Skin_Cancer',
    'Sunlight_Damage', 'Tinea', 'Vascular_Tumor', 'Vasculitis', 'Vitiligo', 'Warts'
]

def create_data_generators():
    """Create training and validation data generators with augmentation"""
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=CONFIG['validation_split']
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['validation_split']
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'train'),
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'train'),
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def create_cnn_model(num_classes):
    """Create a CNN model for skin disease classification"""
    
    if CONFIG['use_pretrained']:
        # Use pre-trained EfficientNetB0 as base
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(*CONFIG['image_size'], 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(num_classes, activation='softmax')
        ])
        
    else:
        # Custom CNN architecture
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*CONFIG['image_size'], 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    return model

def create_callbacks():
    """Create training callbacks"""
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks

def plot_training_history(history):
    """Plot training and validation metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_generator):
    """Evaluate model performance and generate reports"""
    
    # Predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # True labels
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    # Classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print("Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return report, cm

def fine_tune_model(model):
    """Fine-tune the pre-trained model"""
    
    if CONFIG['use_pretrained'] and hasattr(model.layers[0], 'trainable'):
        # Unfreeze the base model
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - CONFIG['fine_tune_layers']
        
        # Freeze all the layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use a lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(CONFIG['learning_rate'] / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Fine-tuning from layer {fine_tune_at} onwards")
    
    return model

def main():
    """Main training function"""
    
    print("Starting skin disease classification model training...")
    print(f"Configuration: {CONFIG}")
    
    # Create output directory
    os.makedirs(CONFIG['model_output_path'], exist_ok=True)
    
    # Create data generators
    print("Creating data generators...")
    train_generator, validation_generator = create_data_generators()
    
    # Create test generator (using test folder if available)
    test_path = os.path.join(CONFIG['dataset_path'], 'test')
    if os.path.exists(test_path):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=CONFIG['image_size'],
            batch_size=CONFIG['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
    else:
        test_generator = validation_generator
    
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Create model
    print("Creating model...")
    model = create_cnn_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=CONFIG['epochs'],
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tune if using pre-trained model
    if CONFIG['use_pretrained']:
        print("Starting fine-tuning...")
        model = fine_tune_model(model)
        
        # Continue training with fine-tuning
        fine_tune_epochs = 20
        history_fine = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=CONFIG['epochs'] + fine_tune_epochs,
            initial_epoch=len(history.history['loss']),
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=callbacks,
            verbose=1
        )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, test_generator)
    
    # Save model
    model_path = os.path.join(CONFIG['model_output_path'], 'skin_cnn.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save class labels
    labels_path = os.path.join(CONFIG['model_output_path'], 'rash_model_labels.txt')
    with open(labels_path, 'w') as f:
        for class_name in train_generator.class_indices.keys():
            f.write(f"{class_name}\n")
    print(f"Labels saved to: {labels_path}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()