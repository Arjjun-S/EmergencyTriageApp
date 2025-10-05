"""
Convert trained text classification model to TensorFlow Lite format
for mobile deployment in Android app
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Configuration
CONFIG = {
    'model_path': '../app/src/main/assets/models/text_classifier.h5',
    'output_path': '../app/src/main/assets/models/text_classifier.tflite',
    'quantization': True,
    'max_sequence_length': 128,
    'vocab_size': 10000
}

def create_representative_dataset_for_text():
    """
    Create a representative dataset for text model quantization
    """
    def representative_data_gen():
        # Generate random sequences that match your text model's input shape
        for _ in range(100):
            # Random integer sequences representing tokenized text
            yield [np.random.randint(0, CONFIG['vocab_size'], 
                                   (1, CONFIG['max_sequence_length'])).astype(np.int32)]
    
    return representative_data_gen

def convert_text_model_to_tflite(
    model_path, 
    output_path, 
    quantization=True
):
    """
    Convert a text classification Keras model to TensorFlow Lite format
    
    Args:
        model_path: Path to the saved Keras model (.h5)
        output_path: Path where TFLite model will be saved (.tflite)
        quantization: Whether to apply quantization
    """
    
    try:
        print(f"Loading text model from: {model_path}")
        
        # Load the trained model
        model = load_model(model_path)
        
        # Print model info
        print("Model Summary:")
        model.summary()
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        if quantization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # For text models, we typically don't need aggressive quantization
            # as they're usually smaller than image models
            
            # Optional: Use representative dataset for better quantization
            # converter.representative_dataset = create_representative_dataset_for_text()
        
        # Text models often use embedding layers that work better with certain settings
        converter.allow_custom_ops = True
        
        # Convert the model
        print("Converting text model to TensorFlow Lite...")
        tflite_model = converter.convert()
        
        # Save the converted model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite text model saved to: {output_path}")
        
        # Get model size information
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)   # MB
        compression_ratio = original_size / tflite_size if tflite_size > 0 else 0
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        return tflite_model
        
    except Exception as e:
        print(f"Error converting text model: {e}")
        return None

def validate_text_tflite_model(tflite_model_path, original_model_path):
    """
    Validate the converted text TFLite model
    """
    
    try:
        print("Validating text TFLite model...")
        
        # Load original model
        original_model = load_model(original_model_path)
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Text TFLite Model Info:")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input dtype: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output dtype: {output_details[0]['dtype']}")
        
        # Create test input (random sequence of integers)
        input_shape = input_details[0]['shape']
        if input_shape[0] is None or input_shape[0] == -1:
            input_shape = (1,) + tuple(input_shape[1:])
        
        # Generate random integer sequence for text input
        test_input = np.random.randint(0, CONFIG['vocab_size'], input_shape).astype(np.int32)
        
        # Get predictions from original model
        original_prediction = original_model.predict(test_input)
        
        # Get predictions from TFLite model
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        tflite_prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Compare predictions
        max_diff = np.max(np.abs(original_prediction - tflite_prediction))
        mean_diff = np.mean(np.abs(original_prediction - tflite_prediction))
        
        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 0.01:  # Threshold for acceptable difference
            print("✅ Text TFLite model validation successful!")
        else:
            print("⚠️  Text TFLite model may have significant differences from original")
        
        return True
        
    except Exception as e:
        print(f"Error validating text model: {e}")
        return False

def benchmark_text_tflite_model(tflite_model_path, num_runs=1000):
    """
    Benchmark the text TFLite model inference time
    """
    
    try:
        print(f"Benchmarking text TFLite model with {num_runs} runs...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        if input_shape[0] is None or input_shape[0] == -1:
            input_shape = (1,) + tuple(input_shape[1:])
        
        # Create test input (random integer sequence)
        test_input = np.random.randint(0, CONFIG['vocab_size'], input_shape).astype(np.int32)
        
        # Warm up
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(num_runs):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
        
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_runs * 1000  # ms
        
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Throughput: {1000/avg_inference_time:.1f} inferences/second")
        
        return avg_inference_time
        
    except Exception as e:
        print(f"Error benchmarking text model: {e}")
        return None

def create_sample_tokenizer_output():
    """
    Create sample files that would be generated by the tokenizer
    for use in the Android app
    """
    
    # Sample vocabulary file
    vocab_path = os.path.join(os.path.dirname(CONFIG['output_path']), 'vocab.txt')
    
    # Common medical/symptom vocabulary
    sample_vocab = [
        '<PAD>', '<OOV>', 'pain', 'fever', 'headache', 'nausea', 'fatigue',
        'cough', 'rash', 'swelling', 'itching', 'burning', 'ache', 'sore',
        'dizziness', 'weakness', 'bleeding', 'numbness', 'tingling', 'stiff',
        'sharp', 'dull', 'throbbing', 'constant', 'mild', 'moderate', 'severe',
        'chest', 'back', 'stomach', 'head', 'neck', 'arm', 'leg', 'skin',
        'red', 'swollen', 'hot', 'cold', 'dry', 'wet', 'hard', 'soft'
    ] + [f'symptom_{i}' for i in range(100)]  # Placeholder symptoms
    
    try:
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word in sample_vocab:
                f.write(f"{word}\n")
        
        print(f"Sample vocabulary saved to: {vocab_path}")
        
    except Exception as e:
        print(f"Error creating sample vocabulary: {e}")

def main():
    """Main text model conversion function"""
    
    print("Starting text model conversion to TensorFlow Lite...")
    
    # Check if input model exists
    if not os.path.exists(CONFIG['model_path']):
        print(f"Error: Text model file not found at {CONFIG['model_path']}")
        print("Please train the text model first using train_text_model.py")
        
        # Create a dummy simple model for demonstration
        print("Creating a dummy text model for demonstration...")
        create_dummy_text_model()
    
    # Convert model
    tflite_model = convert_text_model_to_tflite(
        CONFIG['model_path'],
        CONFIG['output_path'],
        CONFIG['quantization']
    )
    
    if tflite_model is not None:
        # Validate converted model
        validate_text_tflite_model(CONFIG['output_path'], CONFIG['model_path'])
        
        # Benchmark converted model
        benchmark_text_tflite_model(CONFIG['output_path'])
        
        # Create sample tokenizer files
        create_sample_tokenizer_output()
        
        print("✅ Text model conversion completed successfully!")
    else:
        print("❌ Text model conversion failed!")

def create_dummy_text_model():
    """Create a dummy text model for demonstration purposes"""
    
    try:
        print("Creating dummy text classification model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(CONFIG['vocab_size'], 64, 
                                    input_length=CONFIG['max_sequence_length']),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: Mild, Moderate, Severe
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy training data
        X_dummy = np.random.randint(0, CONFIG['vocab_size'], 
                                  (100, CONFIG['max_sequence_length']))
        y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, 3, 100), 3)
        
        # Train for a few epochs
        model.fit(X_dummy, y_dummy, epochs=5, verbose=1)
        
        # Save the model
        os.makedirs(os.path.dirname(CONFIG['model_path']), exist_ok=True)
        model.save(CONFIG['model_path'])
        
        print(f"Dummy text model saved to: {CONFIG['model_path']}")
        
    except Exception as e:
        print(f"Error creating dummy model: {e}")

if __name__ == "__main__":
    main()