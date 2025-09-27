# 🏥 Emergency AI Triage App

[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org/)
[![TensorFlow Lite](https://img.shields.io/badge/ML-TensorFlow%20Lite-orange.svg)](https://www.tensorflow.org/lite)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Complete Multimodal AI-Powered Emergency Medical Triage System for Android**

## 🎯 Overview

The Emergency AI Triage App is a cutting-edge Android application that combines **voice recognition**, **image analysis**, and **text processing** to provide intelligent medical triage decisions. Using on-device machine learning, the app classifies medical emergencies and provides urgency flags with recommended actions.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎤 **Voice-to-Text** | Record symptoms using Android's SpeechRecognizer |
| 📸 **Image Classification** | AI-powered skin disease detection using CNN |
| 🧠 **Multimodal Fusion** | Combines image and text analysis for comprehensive triage |
| 🚨 **Urgency Classification** | 🟢 Green / 🟡 Yellow / 🔴 Red urgency levels |
| 💊 **Disease Precautions** | Automated precaution recommendations |
| 🏥 **Telehealth Integration** | Direct connection to telehealth services |
| 📱 **Offline-First** | All AI processing happens on-device using TensorFlow Lite |

## 📱 App Screenshots

```
[Main Screen]     [Recording]      [Analysis]       [Results]
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  🎤 Record  │   │ 🔴 Speaking │   │ ⏳ Analyzing│   │ 🔴 URGENT   │
│  📸 Capture │   │ "I have..." │   │   Please    │   │ Skin Cancer │
│  🖼️ Gallery  │   │             │   │   wait...   │   │ 92% Conf.   │
│             │   │ 🔇 Stop     │   │             │   │ 📋 Actions  │
│ 🔍 Analyze  │   │             │   │             │   │ 🏥 Call Dr  │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

## 🏗️ Project Structure

```
EmergencyTriageApp/
├── 📁 app/                        # Android application module
│   ├── src/main/
│   │   ├── java/com/example/emergencytriage/
│   │   │   ├── MainActivity.kt    # Main app entry point
│   │   │   ├── ui/screens/        # UI components
│   │   │   ├── ml/                # Machine learning processors
│   │   │   ├── data/models/       # Data classes and models
│   │   │   └── utils/             # Utility classes
│   │   ├── assets/                # TensorFlow Lite models & resources
│   │   └── res/                   # Android resources (layouts, icons, etc.)
│   │       └── mipmap-*/          # App icons (multiple resolutions)
│   └── build.gradle               # App-level build configuration
├── 📁 ml/                         # Python ML training scripts
│   ├── train_image_model.py       # CNN training for skin diseases
│   ├── train_text_model.py        # Text classifier for symptom severity
│   ├── convert_image_to_tflite.py # Image model conversion utilities
│   └── convert_text_to_tflite.py  # Text model conversion utilities
├── 📁 datasets/                   # Medical datasets and training data
│   ├── DiseaseAndSymptoms.csv     # Disease-symptom mappings with severity
│   ├── DiseasePrecaution.csv      # Disease-precaution recommendations
│   └── SkinDisease/               # Image classification dataset (samples)
├── 📁 models/                     # Pre-trained model files
├── 📁 docs/                       # Documentation and architecture diagrams
├── � build.gradle                # Project-level build configuration
├── 📄 requirements.txt            # Python dependencies for ML training
└── 📄 README.md                   # This file
```

## 🤖 AI/ML Pipeline

### 1. **Image Classification (Skin Disease Detection)**
- **Model**: EfficientNetB0 + Custom CNN layers
- **Input**: 224×224 RGB images
- **Output**: 22 skin disease categories with confidence scores
- **Classes**: Acne, Skin Cancer, Eczema, Psoriasis, Melanoma, etc.
- **Accuracy Target**: >85% on test set

### 2. **Text Classification (Symptom Severity)**
- **Model**: LSTM + Dense layers OR DistilBERT
- **Input**: Tokenized symptom descriptions (max 128 tokens)
- **Output**: Severity levels (Mild, Moderate, Severe)
- **Features**: Rule-based + ML-based classification

### 3. **Multimodal Fusion Engine**
- **Approach**: Weighted fusion of image and text predictions
- **Weights**: Image (70%) + Text (30%)
- **Output**: Final urgency level (Green/Yellow/Red)
- **Decision Logic**: Conservative approach (takes higher urgency)

## � Quick Start Guide

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| **Android Studio** | Arctic Fox+ | Android development |
| **Kotlin** | 1.5.0+ | App programming language |
| **Android SDK** | API Level 21+ | Android 5.0+ support |
| **Python** | 3.8+ | ML training (optional) |
| **TensorFlow** | 2.8+ | ML training (optional) |
| **Device RAM** | 4GB+ | Smooth app performance |

### 🔧 Installation Steps

#### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/EmergencyTriageApp.git
cd EmergencyTriageApp
```

#### **Step 2: Open in Android Studio**
1. Launch Android Studio
2. Click "Open an existing Android Studio project"
3. Navigate to the `EmergencyTriageApp` folder and select it
4. Wait for Gradle sync to complete

#### **Step 3: Add App Icon (Optional)**
**📱 Create a Professional App Icon:**

Your app needs an icon to look professional on users' devices. I've created the folder structure for you:

```
app/src/main/res/
├── mipmap-mdpi/     → ic_launcher.png (48×48 px)
├── mipmap-hdpi/     → ic_launcher.png (72×72 px)  
├── mipmap-xhdpi/    → ic_launcher.png (96×96 px)
├── mipmap-xxhdpi/   → ic_launcher.png (144×144 px)
└── mipmap-xxxhdpi/  → ic_launcher.png (192×192 px)
```

**🎨 Quick Icon Creation:**
1. **Use Android Asset Studio** (Recommended): https://romannurik.github.io/AndroidAssetStudio/icons-launcher.html
2. **Upload a 512×512** base design (medical theme: red cross, stethoscope, etc.)
3. **Download all sizes** and place them in the respective mipmap folders
4. **Name each file**: `ic_launcher.png`

**💡 Icon Ideas:** Medical cross + AI circuit, stethoscope + phone, emergency star + mobile device

*See detailed guide: [`docs/APP_ICON_GUIDE.md`](docs/APP_ICON_GUIDE.md)*

#### **Step 4: Build and Run**
1. Connect an Android device (API 21+) or start an emulator
2. Click the "Run" button (▶️) in Android Studio
3. Select your target device
4. The app will install and launch automatically

### 📊 Dataset Setup (Optional - For Training)

If you want to train your own models:

#### **Step 1: Set Up Python Environment**
```bash
# Create virtual environment
python -m venv venv
source venv\Scripts\activate  # On Windows
# source venv/bin/activate    # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

#### **Step 2: Download Full Dataset**
The repository includes sample data only. For full training:

1. **Get HAM10000 Dataset**:
   - Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
   - Download `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip`

2. **Extract Images**:
   ```bash
   # Extract to datasets/SkinDisease/HAM10000_images_part_1/
   # Extract to datasets/SkinDisease/HAM10000_images_part_2/
   ```

#### **Step 3: Train Models**
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

## 🎮 How to Use the App

### **1. Launch the App**
- Open the Emergency Triage App on your Android device
- Grant microphone and camera permissions when prompted

### **2. Record Symptoms (Voice Input)**
- Tap the **🎤 Record** button
- Speak clearly about your symptoms
- Example: *"I have a red rash on my arm with itching and swelling"*
- Tap **🔇 Stop** when finished

### **3. Capture/Upload Image (Optional)**
- Tap **📸 Capture** to take a photo with camera
- Or tap **🖼️ Gallery** to select an existing image
- Ensure the image is clear and well-lit

### **4. Get AI Analysis**
- Tap **🔍 Analyze** to start processing
- The AI will analyze both text and image (if provided)
- Wait for results (usually 2-5 seconds)

### **5. Review Results**
- **Urgency Level**: 🟢 Green (Low) / 🟡 Yellow (Medium) / 🔴 Red (High)
- **Confidence Score**: AI's confidence in the diagnosis
- **Recommended Actions**: Specific steps to take
- **Precautions**: Disease-specific precautionary measures

### **6. Take Action**
- Follow the recommended actions
- Use **🏥 Call Doctor** for telehealth if urgent
- Save results for medical consultation

## � Dataset Information

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

## 🔧 Configuration

### **Android App Configuration**
- **Minimum SDK**: API 21 (Android 5.0)
- **Target SDK**: API 34 (Android 14)
- **Permissions**: CAMERA, RECORD_AUDIO, INTERNET
- **Model Size**: ~50MB total (optimized for mobile)

### **ML Training Configuration**
- **Image Size**: 224×224×3
- **Batch Size**: 32
- **Epochs**: 50-100
- **Learning Rate**: 0.001
- **Optimization**: Adam optimizer

## 🧪 Testing

### **Unit Tests**
```bash
# Run Android unit tests
./gradlew test

# Run instrumented tests  
./gradlew connectedAndroidTest
```

### **ML Model Tests**
```bash
cd ml/
python -m pytest tests/
```

## � Troubleshooting

### **Common Issues**

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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### **Areas for Contribution**
- 🔬 Additional disease categories
- 🎨 UI/UX improvements  
- 🧠 ML model enhancements
- 🌐 Internationalization
- � iOS version
- ♿ Accessibility features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HAM10000 Dataset**: Harvard Dataverse
- **TensorFlow Team**: For mobile ML framework
- **Android Team**: For Speech Recognition APIs
- **Medical Consultants**: For domain expertise

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/EmergencyTriageApp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/EmergencyTriageApp/discussions)
- **Email**: support@emergencytriage.app

---

⚠️ **Medical Disclaimer**: This app is for informational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical emergencies.
