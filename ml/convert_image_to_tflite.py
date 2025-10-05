"""
Convert trained Keras/TensorFlow models to TensorFlow Lite format
for mobile deployment in Android app
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Configuration
CONFIG = {
    'model_path': '../app/src/main/assets/models/skin_cnn.h5',
    'output_path': '../app/src/main/assets/models/rash_model.tflite',
    'quantization': True,  # Enable quantization for smaller model size
    'representative_dataset': None,  # Path to representative dataset for quantization
    'optimization_level': 'DEFAULT'  # 'DEFAULT', 'OPTIMIZE_FOR_SIZE', 'OPTIMIZE_FOR_LATENCY'
}

def create_representative_dataset():
    """
    Create a representative dataset for quantization
    This helps the converter understand the range of input values
    """
    def representative_data_gen():
        # Generate random sample data that matches your model's input shape
        # In production, use real sample data from your training set
        for _ in range(100):
            # Assuming input shape is (224, 224, 3) for image classification
            yield [np.random.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)]
    
    return representative_data_gen

def convert_model_to_tflite(
    model_path, 
    output_path, 
    quantization=True, 
    optimization_level='DEFAULT'
):
    """
    Convert a Keras model to TensorFlow Lite format
    
    Args:
        model_path: Path to the saved Keras model (.h5)
        output_path: Path where TFLite model will be saved (.tflite)
        quantization: Whether to apply quantization
        optimization_level: Level of optimization to apply
    """
    
    try:
        print(f"Loading model from: {model_path}")
        
        # Load the trained model
        model = load_model(model_path)
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        if quantization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # For more aggressive quantization, you can use:
            # converter.target_spec.supported_types = [tf.float16]
            
            # For int8 quantization (requires representative dataset)
            if CONFIG['representative_dataset']:
                converter.representative_dataset = create_representative_dataset()
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
        
        # Additional optimization settings
        if optimization_level == 'OPTIMIZE_FOR_SIZE':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        elif optimization_level == 'OPTIMIZE_FOR_LATENCY':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        
        # Convert the model
        print("Converting model to TensorFlow Lite...")
        tflite_model = converter.convert()
        
        # Save the converted model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved to: {output_path}")
        
        # Get model size information
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)   # MB
        compression_ratio = original_size / tflite_size
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        return tflite_model
        
    except Exception as e:
        print(f"Error converting model: {e}")
        return None

def validate_tflite_model(tflite_model_path, original_model_path):
    """
    Validate the converted TFLite model by comparing outputs
    with the original Keras model
    """
    
    try:
        print("Validating TFLite model...")
        
        # Load original model
        original_model = load_model(original_model_path)
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("TFLite Model Info:")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input dtype: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output dtype: {output_details[0]['dtype']}")
        
        # Create test input
        input_shape = input_details[0]['shape']
        if input_shape[0] is None or input_shape[0] == -1:
            input_shape = (1,) + tuple(input_shape[1:])
        
        test_input = np.random.uniform(0, 1, input_shape).astype(np.float32)
        
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
            print("✅ TFLite model validation successful!")
        else:
            print("⚠️  TFLite model may have significant differences from original")
        
        return True
        
    except Exception as e:
        print(f"Error validating model: {e}")
        return False

def benchmark_tflite_model(tflite_model_path, num_runs=100):
    """
    Benchmark the TFLite model inference time
    """
    
    try:
        print(f"Benchmarking TFLite model with {num_runs} runs...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        if input_shape[0] is None or input_shape[0] == -1:
            input_shape = (1,) + tuple(input_shape[1:])
        
        # Create test input
        test_input = np.random.uniform(0, 1, input_shape).astype(np.float32)
        
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
        print(f"Throughput: {1000/avg_inference_time:.1f} FPS")
        
        return avg_inference_time
        
    except Exception as e:
        print(f"Error benchmarking model: {e}")
        return None

def main():
    """Main conversion function"""
    
    print("Starting image model conversion to TensorFlow Lite...")
    
    # Check if input model exists
    if not os.path.exists(CONFIG['model_path']):
        print(f"Error: Model file not found at {CONFIG['model_path']}")
        print("Please train the model first using train_image_model.py")
        return
    
    # Convert model
    tflite_model = convert_model_to_tflite(
        CONFIG['model_path'],
        CONFIG['output_path'],
        CONFIG['quantization'],
        CONFIG['optimization_level']
    )
    
    if tflite_model is not None:
        # Validate converted model
        validate_tflite_model(CONFIG['output_path'], CONFIG['model_path'])
        
        # Benchmark converted model
        benchmark_tflite_model(CONFIG['output_path'])
        
        print("✅ Image model conversion completed successfully!")
    else:
        print("❌ Image model conversion failed!")

if __name__ == "__main__":
    main()