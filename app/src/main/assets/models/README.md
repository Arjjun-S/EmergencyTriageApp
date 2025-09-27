# Model Placeholder Files

This folder contains placeholder files for the TensorFlow Lite models that will be generated after training.

## Required Model Files:

### 1. Image Classification Model
- **rash_model.tflite** - TensorFlow Lite model for skin disease classification
- **rash_model_labels.txt** - Class labels for the 22 skin disease categories
- **Model Input**: 224x224x3 RGB images
- **Model Output**: Probabilities for 22 disease classes

### 2. Text Classification Model  
- **text_classifier.tflite** - TensorFlow Lite model for symptom severity classification
- **text_classifier_labels.txt** - Severity labels (Mild, Moderate, Severe)
- **vocab.txt** - Vocabulary file for text tokenization
- **Model Input**: Tokenized text sequences (max length 128)
- **Model Output**: Probabilities for 3 severity levels

## Training Models

To generate the actual TFLite models:

1. **Train Image Model**:
   ```bash
   cd ml/
   python train_image_model.py
   python convert_image_to_tflite.py
   ```

2. **Train Text Model**:
   ```bash
   cd ml/
   python train_text_model.py
   python convert_text_to_tflite.py
   ```

## Model Requirements

- **Image Model Size**: Target < 50MB for mobile deployment
- **Text Model Size**: Target < 10MB for mobile deployment
- **Inference Time**: Target < 500ms per prediction on mobile devices
- **Accuracy**: Target > 80% for both models

## Note

The current label files are provided as placeholders. The actual TFLite model files (.tflite) need to be generated through the training process described above.