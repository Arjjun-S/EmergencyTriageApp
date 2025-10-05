## 🏥 Emergency AI Triage App (Android)

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


Uploading Explanation_Video.mp4…


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
3. Press “Record” to dictate symptoms; press “Stop” when done
4. Tap “Analyze” to run multimodal inference
5. Review urgency flag, predicted condition(s), confidence, and precautions
6. Use telehealth shortcuts if the case is urgent

The app displays a small banner if ML assets are missing, and uses rule-based fallback logic so you can still test end-to-end flows.

---

## 4) Project Structure

```
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

```powershell
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

```
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
  ```
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

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train and convert image model:

```powershell
python .\ml\train_image_model.py
python .\ml\convert_image_to_tflite.py
```

Train and convert text model:

```powershell
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
- Image: ~85–92% accuracy
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
|------|------------|
| “Models missing” banner | Confirm files under `app/src/main/assets/models/` |
| Build errors | Let Gradle sync; Build > Clean Project; restart Android Studio |
| Poor image results | Improve lighting/quality, expand training data |
| Speech not captured | Check microphone permission; try in quiet area |

---

## 13) Testing

Android tests
```powershell
./gradlew test
./gradlew connectedAndroidTest
```

ML tests (optional, if you add Python tests)
```powershell
cd ml
python -m pytest tests/
```

---

## 14) Make the Repository Public

If this repository is not public:
1. Open the GitHub repo page
2. Settings → General → Danger Zone
3. “Change repository visibility” → Public

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

   datasets/SkinDisease/HAM10000_images_part_2/

   ```#### **Step 3: Train Models**

```bash

#### 2. Alternative Datasets for Enhanced Trainingcd ml/



**ISIC 2019 Challenge Dataset:**# Train image classification model

- **URL**: https://challenge.isic-archive.com/data/python train_image_model.py

- **Size**: 25,331 images

- **Classes**: 8 diagnostic categories# Train text classification model  

- **Use Case**: More diverse skin lesion typespython train_text_model.py



**DermNet Dataset:**# Convert models to TensorFlow Lite

- **URL**: http://www.dermnet.com/python convert_image_to_tflite.py

- **Size**: 23,000+ imagespython convert_text_to_tflite.py

- **Classes**: 23 skin condition classes```

- **Use Case**: Broader skin condition coverage

## 🎮 How to Use the App

**PH2 Dataset:**

- **URL**: https://www.fc.up.pt/addi/ph2%20database.html### **1. Launch the App**

- **Size**: 200 dermoscopic images- Open the Emergency Triage App on your Android device

- **Use Case**: Melanoma detection focus- Grant microphone and camera permissions when prompted



### Training Setup with Complete Datasets### **2. Record Symptoms (Voice Input)**

- Tap the **🎤 Record** button

#### Step 1: Python Environment Setup- Speak clearly about your symptoms

```bash- Example: *"I have a red rash on my arm with itching and swelling"*

# Create virtual environment- Tap **🔇 Stop** when finished

python -m venv venv

### **3. Capture/Upload Image (Optional)**

# Activate environment- Tap **📸 Capture** to take a photo with camera

# Windows:- Or tap **🖼️ Gallery** to select an existing image

venv\Scripts\activate- Ensure the image is clear and well-lit

# Linux/Mac:

source venv/bin/activate### **4. Get AI Analysis**

- Tap **🔍 Analyze** to start processing

# Install dependencies- The AI will analyze both text and image (if provided)

pip install -r requirements.txt- Wait for results (usually 2-5 seconds)

```

### **5. Review Results**

#### Step 2: Configure Training Data- **Urgency Level**: 🟢 Green (Low) / 🟡 Yellow (Medium) / 🔴 Red (High)

```bash- **Confidence Score**: AI's confidence in the diagnosis

# Ensure datasets are in correct structure:- **Recommended Actions**: Specific steps to take

datasets/- **Precautions**: Disease-specific precautionary measures

├── DiseaseAndSymptoms.csv

├── DiseasePrecaution.csv### **6. Take Action**

└── SkinDisease/- Follow the recommended actions

    ├── HAM10000_metadata.csv- Use **🏥 Call Doctor** for telehealth if urgent

    ├── HAM10000_images_part_1/  # 5,000+ images- Save results for medical consultation

    └── HAM10000_images_part_2/  # 5,000+ images

```## � Dataset Information



#### Step 3: Train High-Accuracy Models### **Disease and Symptoms Dataset**

```bash- **File**: `datasets/DiseaseAndSymptoms.csv`

cd ml/- **Records**: ~40 diseases with associated symptoms

- **Columns**: Disease, Symptom_1 to Symptom_17, Severity Level

# Train image classification model with full dataset

python train_image_model.py### **Disease Precautions Dataset**  

- **File**: `datasets/DiseasePrecaution.csv`

# Train text classification model- **Records**: Precautionary measures for each disease

python train_text_model.py- **Columns**: Disease, Precaution_1 to Precaution_4



# Convert trained models to TensorFlow Lite### **Skin Disease Image Dataset**

python convert_image_to_tflite.py- **Location**: `datasets/SkinDisease/`

python convert_text_to_tflite.py- **Format**: JPEG images, 224×224 recommended

```- **Classes**: 22 common skin conditions

- **Note**: Sample images included, download full dataset for training

### Expected Performance with Full Datasets

- **Skin Disease Classification**: 85-92% accuracy## 🔧 Configuration

- **Text Symptom Analysis**: 88-95% accuracy

- **Training Time**: 2-4 hours on GPU, 8-12 hours on CPU### **Android App Configuration**

- **Model Size**: 15-25 MB per model (optimized for mobile)- **Minimum SDK**: API 21 (Android 5.0)

- **Target SDK**: API 34 (Android 14)

**Note**: Training with complete datasets significantly improves model accuracy and real-world performance compared to sample data.- **Permissions**: CAMERA, RECORD_AUDIO, INTERNET

- **Model Size**: ~50MB total (optimized for mobile)

## How to Use the App

### **ML Training Configuration**

### 1. Launch the App- **Image Size**: 224×224×3

- Open the Emergency Triage App on your Android device- **Batch Size**: 32

- Grant microphone and camera permissions when prompted- **Epochs**: 50-100

- **Learning Rate**: 0.001

### 2. Record Symptoms (Voice Input)- **Optimization**: Adam optimizer

- Tap the **Record** button

- Speak clearly about your symptoms## 🧪 Testing

- Example: *"I have a red rash on my arm with itching and swelling"*

- Tap **Stop** when finished### **Unit Tests**

```bash

### 3. Capture/Upload Image (Optional)# Run Android unit tests

- Tap **Capture** to take a photo with camera./gradlew test

- Or tap **Gallery** to select an existing image

- Ensure the image is clear and well-lit# Run instrumented tests  

./gradlew connectedAndroidTest

### 4. Get AI Analysis```

- Tap **Analyze** to start processing

- The AI will analyze both text and image (if provided)### **ML Model Tests**

- Wait for results (usually 2-5 seconds)```bash

cd ml/

### 5. Review Resultspython -m pytest tests/

- **Urgency Level**: Green (Low) / Yellow (Medium) / Red (High)```

- **Confidence Score**: AI's confidence in the diagnosis

- **Recommended Actions**: Specific steps to take## � Troubleshooting

- **Precautions**: Disease-specific precautionary measures

### **Common Issues**

### 6. Take Action

- Follow the recommended actions| Issue | Solution |

- Use **Call Doctor** for telehealth if urgent|-------|----------|

- Save results for medical consultation| **App crashes on startup** | Check permissions in Settings > Apps > Emergency Triage |

| **Models not loading** | Ensure TensorFlow Lite files are in `app/src/main/assets/` |

## Configuration| **Poor image classification** | Use well-lit, clear images; retrain with more data |

| **Voice recognition fails** | Check microphone permissions and speak clearly |

### Android App Configuration| **Build errors** | Clean project: `Build > Clean Project` in Android Studio |

- **Minimum SDK**: API 21 (Android 5.0)

- **Target SDK**: API 34 (Android 14)### **Performance Optimization**

- **Permissions**: CAMERA, RECORD_AUDIO, INTERNET- **RAM Usage**: ~200-300MB typical usage

- **Model Size**: ~50MB total (optimized for mobile)- **Storage**: ~100MB for app + models

- **Battery**: Minimal impact with on-device processing

### ML Training Configuration- **Network**: Only required for telehealth features

- **Image Size**: 224×224×3

- **Batch Size**: 32## 🤝 Contributing

- **Epochs**: 50-100

- **Learning Rate**: 0.001We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

- **Optimization**: Adam optimizer

### **Areas for Contribution**

## Testing- 🔬 Additional disease categories

- 🎨 UI/UX improvements  

### Unit Tests- 🧠 ML model enhancements

```bash- 🌐 Internationalization

# Run Android unit tests- � iOS version

./gradlew test- ♿ Accessibility features



# Run instrumented tests  ## 📄 License

./gradlew connectedAndroidTest

```This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



### ML Model Tests## 🙏 Acknowledgments

```bash

cd ml/- **HAM10000 Dataset**: Harvard Dataverse

python -m pytest tests/- **TensorFlow Team**: For mobile ML framework

```- **Android Team**: For Speech Recognition APIs

- **Medical Consultants**: For domain expertise

## Troubleshooting

## 📞 Support

### Common Issues

- **Issues**: [GitHub Issues](https://github.com/yourusername/EmergencyTriageApp/issues)

| Issue | Solution |- **Discussions**: [GitHub Discussions](https://github.com/yourusername/EmergencyTriageApp/discussions)

|-------|----------|- **Email**: support@emergencytriage.app

| **App crashes on startup** | Check permissions in Settings > Apps > Emergency Triage |

| **Models not loading** | Ensure TensorFlow Lite files are in `app/src/main/assets/` |---

| **Poor image classification** | Use well-lit, clear images; retrain with more data |

| **Voice recognition fails** | Check microphone permissions and speak clearly |⚠️ **Medical Disclaimer**: This app is for informational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical emergencies.

| **Build errors** | Clean project: `Build > Clean Project` in Android Studio |

### Performance Optimization
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
