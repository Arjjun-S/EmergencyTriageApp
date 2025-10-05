## Emergency AI Triage App (Android)

Talk. Show. Get help — in minutes. This Android app performs on-device triage by combining voice (symptoms), text, and skin-image analysis with TensorFlow Lite. It is designed to run smoothly on everyday devices and keep data private on your phone.

Team Blue Orchids — SRM Institute of Science and Technology (KTR)

Members: Arjjun S, B B Hemanth, H A Pranav Jai, S M Sudharshan, Bala Tharun R

---

## Table of Contents

1. Submissions
2. Overview and Key Features
3. App Interface (Workflow)
4. Project Structure
5. Getting Started (Android)
6. ML Assets Placement
7. Datasets and Data Preparation
8. Training and Converting Models
9. AI/ML Pipeline and How It Works
10. Configuration (Android + ML)
11. Performance and Optimizations
12. Troubleshooting
13. Testing
14. Make the Repository Public
15. License, Ethics, and Data
16. Acknowledgments and Support
17. Medical Disclaimer

---

## 1) Submissions

Video demo:

https://github.com/user-attachments/assets/9fec213d-3475-4a4a-a04e-689a87045274


- Detailed documentation: https://github.com/Arjjun-S/EmergencyTriageApp/blob/767614fcddc16df7869cb860f068a1fa86f7f719/BLUE%20ORCHIDS.pdf

Replace the placeholders with your actual URLs. Keep this section even if assets are already uploaded to GitHub.

---

## 2) Overview and Key Features

The app combines voice recognition, image analysis, and text processing to provide intelligent medical triage decisions. Using on-device ML (TensorFlow Lite), it classifies likely conditions, computes urgency, and suggests recommended actions.

- Voice-to-Text: Capture symptoms via Android SpeechRecognizer
- Image Classification: AI-powered skin condition detection using a CNN
- Text Classification: Estimate severity from typed/dictated symptoms
- Multimodal Fusion: Weighted fusion of image + text predictions
- Urgency Flags: Green / Yellow / Red
- Disease Precautions: Maps conditions to recommended actions
- Telehealth Integration: Hooks to call/connect to care
- Offline-First: All ML inference on-device (privacy-friendly)

Why it runs well on-device:
- Mobile-friendly TFLite models; quantization enabled where possible
- Efficient pre/post-processing for low latency
- Graceful fallbacks: app remains usable even if models are missing

---

## 3) App Interface (Workflow)

1. Launch the app and grant Camera/Microphone permissions
2. Optional: capture or select a skin image
3. Press "Record" to dictate symptoms; press "Stop" when done
4. Tap "Analyze" to run multimodal inference
5. Review urgency flag, predicted condition(s), confidence, and precautions
6. Use telehealth shortcuts if the case is urgent

The app displays a small banner if ML assets are missing, and uses rule-based fallback logic so you can still test end-to-end flows.

---

## 4) Project Structure

```bash
EmergencyTriageApp/
├── app/                           # Android application (Kotlin, MVVM)
│   ├── src/main/
│   │   ├── java/com/example/emergencytriage/
│   │   │   ├── MainActivity.kt
│   │   │   ├── ml/                # On-device inference (TFLite)
│   │   │   ├── data/              # Repositories, models, cache
│   │   │   └── ui/                # Screens and UI
│   │   ├── assets/models/         # Place your .tflite and label files here
│   │   └── res/                   # Layouts, drawables, icons
│   └── build.gradle
├── ml/                            # Python training & conversion scripts
│   ├── train_image_model.py
│   ├── train_text_model.py
│   ├── convert_image_to_tflite.py
│   └── convert_text_to_tflite.py
├── datasets/                      # Sample CSVs & example images
│   ├── DiseaseAndSymptoms.csv
│   ├── DiseasePrecaution.csv
│   └── SkinDisease/
├── requirements.txt               # Python deps for model training
├── ML_TRAINING_SUMMARY.md         # Training notes/results
└── README.md
```

---

## 5) Getting Started (Android)

Prerequisites
- Android Studio (Giraffe/Flamingo or newer)
- JDK 17 (project targets Java/Kotlin 17)

Steps (Windows PowerShell)

```bash
# Clone and open in Android Studio
git clone https://github.com/Arjjun-S/EmergencyTriageApp.git
cd EmergencyTriageApp

# Open the folder in Android Studio and let Gradle sync
# Run on a device/emulator (Android 8.0 / API 26+ recommended)
```

---

## 6) ML Assets Placement

Place these files in `app/src/main/assets/models/`:

- Image model: `rash_model.tflite`, `rash_model_labels.txt`
- Text model: `text_classifier.tflite`, `text_classifier_labels.txt`, `vocab.txt`
- Optional data: `DiseasePrecaution.csv`

Final asset layout:

```bash
app/src/main/assets/models/
  rash_model.tflite
  rash_model_labels.txt
  text_classifier.tflite
  text_classifier_labels.txt
  vocab.txt
  DiseasePrecaution.csv
```

If any assets are missing, the app still runs with safe fallbacks.

---

## 7) Datasets and Data Preparation

1) Image dataset (skin/rash):

- Recommended Keras directory layout:
  ```bash
  datasets/SkinDisease/
    train/
      Acne/ ...jpg
      Eczema/ ...jpg
      ...
    test/  (optional)
  ```
- Use public datasets (e.g., HAM10000, ISIC) where licenses permit.
- Keep classes balanced and remove any personal identifiers/EXIF.

2) Symptoms dataset (text):

- Use `datasets/DiseaseAndSymptoms.csv` with `Symptom_1..Symptom_n` columns.
- Optional `Severity` column; if missing, the training script infers labels from keywords.

---

## 8) Training and Converting Models

Environment setup (Windows PowerShell):

```bash
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train and convert image model:

```bash
python .\ml\train_image_model.py
python .\ml\convert_image_to_tflite.py
```

Train and convert text model:

```bash
python .\ml\train_text_model.py
python .\ml\convert_text_to_tflite.py
```

Notes
- For a lighter text model suitable for mobile, set `model_type` to `simple_nn` or `lstm` in `train_text_model.py`.
- The converters already enable quantization to reduce model size.

---

## 9) AI/ML Pipeline and How It Works

Image Classification (Skin Disease)
- Architecture: EfficientNetB0 base + custom dense head
- Input: 224×224 RGB images
- Output: class probabilities for skin conditions

Text Classification (Symptom Severity)
- Options: simple NN, LSTM, or DistilBERT
- Input: tokenized symptom descriptions (max ~128 tokens)
- Output: severity/condition probabilities

Multimodal Fusion
- Fusion: weighted average (default Image 0.7, Text 0.3)
- Urgency logic: conservative mapping to Green/Yellow/Red with precautions

On-Device Inference
- Entire inference pipeline runs via TFLite; telehealth is optional.

Expected (with full datasets; indicative)
- Image: ~84–92% accuracy
- Text: ~88–95% accuracy

---

## 10) Configuration (Android + ML)

Android app
- Min SDK: 26 (Android 8.0) or as configured in `build.gradle`
- Target SDK: 34
- Permissions: CAMERA, RECORD_AUDIO, INTERNET (for telehealth links)

Training defaults
- Image size: 224×224×3
- Batch size: 32
- Epochs: 50–100
- Optimizer: Adam (0.001)

---

## 11) Performance and Optimizations

- Quantize models (float16/int8) for better speed/size
- Use NNAPI or GPU delegate if available on device
- Keep input images clear, centered, and well-lit to improve accuracy
- Ensure label order matches between training and mobile

---

## 12) Troubleshooting

| Issue | Resolution |
|-------|------------|
| "Models missing" banner | Confirm files under `app/src/main/assets/models/` |
| Build errors | Let Gradle sync; Build > Clean Project; restart Android Studio |
| Poor image results | Improve lighting/quality, expand training data |
| Speech not captured | Check microphone permission; try in quiet area |

---

## 13) Testing

Android tests
```bash
./gradlew test
./gradlew connectedAndroidTest
```

ML tests (optional, if you add Python tests)
```bash
cd ml
python -m pytest tests/
```

---

## 14) Make the Repository Public

If this repository is not public:
1. Open the GitHub repo page
2. Settings → General → Danger Zone
3. "Change repository visibility" → Public

---

## 15) License, Ethics, and Data

- Use only datasets you have rights to and respect their licenses
- Do not log PHI; keep data local and private by default

---

## 16) Acknowledgments and Support

Acknowledgments
- HAM10000 Dataset (Harvard Dataverse)
- TensorFlow / Android teams for on-device ML and Speech APIs
- Medical advisors for domain guidance

Support
- Issues: https://github.com/Arjjun-S/EmergencyTriageApp/issues
- Discussions: https://github.com/Arjjun-S/EmergencyTriageApp/discussions

---

## 17) Medical Disclaimer

This app provides informational triage assistance only and does not replace professional medical advice, diagnosis, or treatment. For emergencies, call your local emergency number immediately.

---

## Alternative Datasets for Enhanced Training

**ISIC 2019 Challenge Dataset:**
- **URL**: https://challenge.isic-archive.com/data/
- **Size**: 25,331 images
- **Classes**: 8 diagnostic categories
- **Use Case**: More diverse skin lesion types

**DermNet Dataset:**
- **URL**: http://www.dermnet.com/
- **Size**: 23,000+ images
- **Classes**: 23 skin condition classes
- **Use Case**: Broader skin condition coverage

**PH2 Dataset:**
- **URL**: https://www.fc.up.pt/addi/ph2%20database.html
- **Size**: 200 dermoscopic images
- **Use Case**: Melanoma detection focus

## How to Use the App

### **1. Launch the App**
- Open the Emergency Triage App on your Android device
- Grant microphone and camera permissions when prompted

### **2. Record Symptoms (Voice Input)**
- Tap the **Record** button
- Speak clearly about your symptoms
- Example: *"I have a red rash on my arm with itching and swelling"*
- Tap **Stop** when finished

### **3. Capture/Upload Image (Optional)**
- Tap **Capture** to take a photo with camera
- Or tap **Gallery** to select an existing image
- Ensure the image is clear and well-lit

### **4. Get AI Analysis**
- Tap **Analyze** to start processing
- The AI will analyze both text and image (if provided)
- Wait for results (usually 2-5 seconds)

### **5. Review Results**
- **Urgency Level**: Green (Low) / Yellow (Medium) / Red (High)
- **Confidence Score**: AI's confidence in the diagnosis
- **Recommended Actions**: Specific steps to take
- **Precautions**: Disease-specific precautionary measures

### **6. Take Action**
- Follow the recommended actions
- Use **Call Doctor** for telehealth if urgent
- Save results for medical consultation

## Dataset Information

### **Disease and Symptoms Dataset**
- **File**: `datasets/DiseaseAndSymptoms.csv`
- **Records**: ~40 diseases with associated symptoms
- **Columns**: Disease, Symptom_1 to Symptom_17, Severity Level

### **Disease Precautions Dataset**
- **File**: `datasets/DiseasePrecaution.csv`
- **Records**: Precautionary measures for each disease
- **Columns**: Disease, Precaution_1 to Precaution_4

### **Skin Disease Image Dataset**
- **Location**: `datasets/SkinDisease/`
- **Format**: JPEG images, 224×224 recommended
- **Classes**: 22 common skin conditions
- **Note**: Sample images included, download full dataset for training

### Training Setup with Complete Datasets

#### Step 1: Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Configure Training Data

```bash
# Ensure datasets are in correct structure:
datasets/
├── DiseaseAndSymptoms.csv
├── DiseasePrecaution.csv
└── SkinDisease/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/  # 5,000+ images
    └── HAM10000_images_part_2/  # 5,000+ images
```

#### Step 3: Train Models

```bash
cd ml/

# Train image classification model
python train_image_model.py

# Train text classification model
python train_text_model.py

# Convert models to TensorFlow Lite
python convert_image_to_tflite.py
python convert_text_to_tflite.py
```

### Expected Performance with Full Datasets

- **Skin Disease Classification**: 85-92% accuracy
- **Text Symptom Analysis**: 88-95% accuracy
- **Training Time**: 2-4 hours on GPU, 8-12 hours on CPU
- **Model Size**: 15-25 MB per model (optimized for mobile)

**Note**: Training with complete datasets significantly improves model accuracy and real-world performance compared to sample data.

## Common Issues

| Issue | Solution |
|-------|----------|
| **App crashes on startup** | Check permissions in Settings > Apps > Emergency Triage |
| **Models not loading** | Ensure TensorFlow Lite files are in `app/src/main/assets/` |
| **Poor image classification** | Use well-lit, clear images; retrain with more data |
| **Voice recognition fails** | Check microphone permissions and speak clearly |
| **Build errors** | Clean project: `Build > Clean Project` in Android Studio |

### **Performance Optimization**
- **RAM Usage**: ~200-300MB typical usage
- **Storage**: ~100MB for app + models
- **Battery**: Minimal impact with on-device processing
- **Network**: Only required for telehealth features

## Contributing

We welcome contributions! Areas for contribution:
- Additional disease categories
- UI/UX improvements  
- ML model enhancements
- Internationalization
- iOS version
- Accessibility features

## Support

- **Issues**: [GitHub Issues](https://github.com/Arjjun-S/EmergencyTriageApp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Arjjun-S/EmergencyTriageApp/discussions)

---

**Medical Disclaimer**: This app is for informational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical emergencies.
