"""
Text Classification Model Training Script
Trains a text classifier on symptom data for severity assessment
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import re
import string

# Configuration
CONFIG = {
    'dataset_path': '../datasets/DiseaseAndSymptoms.csv',
    'model_output_path': '../app/src/main/assets/',
    'max_sequence_length': 128,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 2e-5,
    'validation_split': 0.2,
    'model_type': 'distilbert',  # 'simple_nn', 'lstm', 'distilbert'
    'vocab_size': 10000,
    'embedding_dim': 128
}

def load_and_preprocess_data():
    """Load and preprocess the symptom dataset"""
    
    # Load the CSV file
    df = pd.read_csv(CONFIG['dataset_path'])
    
    # Combine all symptom columns into a single text column
    symptom_columns = [col for col in df.columns if col.startswith('Symptom_')]
    
    # Create text samples by combining symptoms
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        # Combine non-null symptoms into a single text
        symptoms = []
        for col in symptom_columns:
            if pd.notna(row[col]) and row[col].strip():
                symptoms.append(row[col].strip())
        
        if symptoms:  # Only include rows with at least one symptom
            text = ' '.join(symptoms)
            texts.append(text)
            labels.append(row['Severity'] if 'Severity' in df.columns else 'Moderate')
    
    # If no severity column, create synthetic labels based on keywords
    if 'Severity' not in df.columns:
        labels = create_synthetic_labels(texts)
    
    return texts, labels

def create_synthetic_labels(texts):
    """Create synthetic severity labels based on symptom keywords"""
    
    severe_keywords = [
        'high_fever', 'chest_pain', 'breathlessness', 'severe', 'acute', 'emergency',
        'blood', 'hemorrhage', 'coma', 'seizure', 'paralysis', 'heart_attack'
    ]
    
    mild_keywords = [
        'mild', 'slight', 'itching', 'rash', 'dry', 'minor', 'light',
        'skin_rash', 'mild_fever', 'fatigue'
    ]
    
    labels = []
    for text in texts:
        text_lower = text.lower()
        
        severe_count = sum(1 for keyword in severe_keywords if keyword in text_lower)
        mild_count = sum(1 for keyword in mild_keywords if keyword in text_lower)
        
        if severe_count > 0:
            labels.append('Severe')
        elif mild_count > 0:
            labels.append('Mild')
        else:
            labels.append('Moderate')
    
    return labels

def preprocess_text(texts):
    """Clean and preprocess text data"""
    
    processed_texts = []
    
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        
        # Replace underscores with spaces
        text = text.replace('_', ' ')
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        processed_texts.append(text)
    
    return processed_texts

def create_simple_nn_model(vocab_size, max_length, num_classes):
    """Create a simple neural network model"""
    
    model = models.Sequential([
        layers.Embedding(vocab_size, CONFIG['embedding_dim'], input_length=max_length),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_lstm_model(vocab_size, max_length, num_classes):
    """Create an LSTM model"""
    
    model = models.Sequential([
        layers.Embedding(vocab_size, CONFIG['embedding_dim'], input_length=max_length),
        layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
        layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_distilbert_model(num_classes):
    """Create a DistilBERT-based model"""
    
    model = TFDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_classes
    )
    
    return model

def tokenize_texts(texts, tokenizer_type='keras'):
    """Tokenize texts based on the specified tokenizer type"""
    
    if tokenizer_type == 'keras':
        # Use Keras tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=CONFIG['vocab_size'],
            oov_token="<OOV>"
        )
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=CONFIG['max_sequence_length'],
            padding='post'
        )
        
        return sequences, tokenizer
    
    elif tokenizer_type == 'distilbert':
        # Use DistilBERT tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=CONFIG['max_sequence_length'],
            return_tensors='tf'
        )
        
        return encoded, tokenizer

def plot_training_history(history):
    """Plot training and validation metrics"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('text_training_history.png')
    plt.show()

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model performance"""
    
    # Predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # True labels
    true_classes = np.argmax(y_test, axis=1)
    class_labels = label_encoder.classes_
    
    # Classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print("Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('text_confusion_matrix.png')
    plt.show()
    
    return report, cm

def save_tokenizer_vocab(tokenizer, output_path):
    """Save tokenizer vocabulary for mobile deployment"""
    
    if hasattr(tokenizer, 'word_index'):
        # Keras tokenizer
        vocab_path = os.path.join(output_path, 'vocab.txt')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word, index in sorted(tokenizer.word_index.items(), key=lambda x: x[1]):
                f.write(f"{word}\n")
        print(f"Vocabulary saved to: {vocab_path}")
    else:
        print("Tokenizer vocabulary not saved (DistilBERT uses built-in vocab)")

def main():
    """Main training function"""
    
    print("Starting text classification model training...")
    print(f"Configuration: {CONFIG}")
    
    # Create output directory
    os.makedirs(CONFIG['model_output_path'], exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    texts, labels = load_and_preprocess_data()
    texts = preprocess_text(texts)
    
    print(f"Total samples: {len(texts)}")
    print(f"Unique labels: {set(labels)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # Convert to categorical
    y_categorical = tf.keras.utils.to_categorical(encoded_labels, num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y_categorical, 
        test_size=CONFIG['validation_split'], 
        random_state=42,
        stratify=encoded_labels
    )
    
    # Tokenize texts
    if CONFIG['model_type'] == 'distilbert':
        X_train_encoded, tokenizer = tokenize_texts(X_train, 'distilbert')
        X_test_encoded, _ = tokenize_texts(X_test, 'distilbert')
        
        # Create model
        model = create_distilbert_model(num_classes)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(CONFIG['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    else:
        X_train_encoded, tokenizer = tokenize_texts(X_train, 'keras')
        X_test_encoded, _ = tokenize_texts(X_test, 'keras')
        
        # Create model based on type
        if CONFIG['model_type'] == 'simple_nn':
            model = create_simple_nn_model(CONFIG['vocab_size'], CONFIG['max_sequence_length'], num_classes)
        elif CONFIG['model_type'] == 'lstm':
            model = create_lstm_model(CONFIG['vocab_size'], CONFIG['max_sequence_length'], num_classes)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(CONFIG['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    model.summary()
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Train model
    print("Starting training...")
    if CONFIG['model_type'] == 'distilbert':
        # DistilBERT training
        history = model.fit(
            X_train_encoded['input_ids'],
            y_train,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['epochs'],
            validation_data=(X_test_encoded['input_ids'], y_test),
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Standard Keras training
        history = model.fit(
            X_train_encoded, y_train,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['epochs'],
            validation_data=(X_test_encoded, y_test),
            callbacks=callbacks,
            verbose=1
        )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    if CONFIG['model_type'] == 'distilbert':
        evaluate_model(model, X_test_encoded['input_ids'], y_test, label_encoder)
    else:
        evaluate_model(model, X_test_encoded, y_test, label_encoder)
    
    # Save model
    model_path = os.path.join(CONFIG['model_output_path'], 'text_classifier.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save labels
    labels_path = os.path.join(CONFIG['model_output_path'], 'text_classifier_labels.txt')
    with open(labels_path, 'w') as f:
        for label in label_encoder.classes_:
            f.write(f"{label}\n")
    print(f"Labels saved to: {labels_path}")
    
    # Save tokenizer vocabulary
    save_tokenizer_vocab(tokenizer, CONFIG['model_output_path'])
    
    print("Text classification training completed successfully!")

if __name__ == "__main__":
    main()