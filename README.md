# Emergency Triage App# 🏥 Emergency AI Triage App



[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)

[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org/)[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org/)

[![TensorFlow Lite](https://img.shields.io/badge/ML-TensorFlow%20Lite-orange.svg)](https://www.tensorflow.org/lite)[![TensorFlow Lite](https://img.shields.io/badge/ML-TensorFlow%20Lite-orange.svg)](https://www.tensorflow.org/lite)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A Complete Multimodal AI-Powered Emergency Medical Triage System for Android**

> **A Complete Multimodal AI-Powered Emergency Medical Triage System for Android**

## Overview

## 🎯 Overview

The Emergency AI Triage App is a cutting-edge Android application that combines voice recognition, image analysis, and text processing to provide intelligent medical triage decisions. Using on-device machine learning, the app classifies medical emergencies and provides urgency flags with recommended actions.

The Emergency AI Triage App is a cutting-edge Android application that combines **voice recognition**, **image analysis**, and **text processing** to provide intelligent medical triage decisions. Using on-device machine learning, the app classifies medical emergencies and provides urgency flags with recommended actions.

### Key Features

### ✨ Key Features

| Feature | Description |

|---------|-------------|| Feature | Description |

| **Voice-to-Text** | Record symptoms using Android's SpeechRecognizer ||---------|-------------|

| **Image Classification** | AI-powered skin disease detection using CNN || 🎤 **Voice-to-Text** | Record symptoms using Android's SpeechRecognizer |

| **Multimodal Fusion** | Combines image and text analysis for comprehensive triage || 📸 **Image Classification** | AI-powered skin disease detection using CNN |

| **Urgency Classification** | Green / Yellow / Red urgency levels || 🧠 **Multimodal Fusion** | Combines image and text analysis for comprehensive triage |

| **Disease Precautions** | Automated precaution recommendations || 🚨 **Urgency Classification** | 🟢 Green / 🟡 Yellow / 🔴 Red urgency levels |

| **Telehealth Integration** | Direct connection to telehealth services || 💊 **Disease Precautions** | Automated precaution recommendations |

| **Offline-First** | All AI processing happens on-device using TensorFlow Lite || 🏥 **Telehealth Integration** | Direct connection to telehealth services |

| 📱 **Offline-First** | All AI processing happens on-device using TensorFlow Lite |

## App Interface

## 📱 App Screenshots

```

[Main Screen]     [Recording]      [Analysis]       [Results]```

┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐[Main Screen]     [Recording]      [Analysis]       [Results]

│  Record     │   │  Speaking   │   │  Analyzing  │   │  URGENT     │┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐

│  Capture    │   │ "I have..." │   │   Please    │   │ Skin Cancer ││  🎤 Record  │   │ 🔴 Speaking │   │ ⏳ Analyzing│   │ 🔴 URGENT   │

│  Gallery    │   │             │   │   wait...   │   │ 92% Conf.   ││  📸 Capture │   │ "I have..." │   │   Please    │   │ Skin Cancer │

│             │   │  Stop       │   │             │   │  Actions    ││  🖼️ Gallery  │   │             │   │   wait...   │   │ 92% Conf.   │

│  Analyze    │   │             │   │             │   │  Call Dr    ││             │   │ 🔇 Stop     │   │             │   │ 📋 Actions  │

└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘│ 🔍 Analyze  │   │             │   │             │   │ 🏥 Call Dr  │

```└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘

```

## Project Structure

## 🏗️ Project Structure

```

EmergencyTriageApp/```

├── app/                           # Android application moduleEmergencyTriageApp/

│   ├── src/main/├── 📁 app/                        # Android application module

│   │   ├── java/com/example/emergencytriage/│   ├── src/main/

│   │   │   ├── MainActivity.kt    # Main app entry point│   │   ├── java/com/example/emergencytriage/

│   │   │   ├── ui/screens/        # UI components│   │   │   ├── MainActivity.kt    # Main app entry point

│   │   │   ├── ml/                # Machine learning processors│   │   │   ├── ui/screens/        # UI components

│   │   │   ├── data/models/       # Data classes and models│   │   │   ├── ml/                # Machine learning processors

│   │   │   └── utils/             # Utility classes│   │   │   ├── data/models/       # Data classes and models

│   │   ├── assets/                # TensorFlow Lite models & resources│   │   │   └── utils/             # Utility classes

│   │   └── res/                   # Android resources (layouts, icons, etc.)│   │   ├── assets/                # TensorFlow Lite models & resources

│   │       └── mipmap-*/          # App icons (multiple resolutions)│   │   └── res/                   # Android resources (layouts, icons, etc.)

│   └── build.gradle               # App-level build configuration│   │       └── mipmap-*/          # App icons (multiple resolutions)

├── ml/                            # Python ML training scripts│   └── build.gradle               # App-level build configuration

│   ├── train_image_model.py       # CNN training for skin diseases├── 📁 ml/                         # Python ML training scripts

│   ├── train_text_model.py        # Text classifier for symptom severity│   ├── train_image_model.py       # CNN training for skin diseases

│   ├── convert_image_to_tflite.py # Image model conversion utilities│   ├── train_text_model.py        # Text classifier for symptom severity

│   └── convert_text_to_tflite.py  # Text model conversion utilities│   ├── convert_image_to_tflite.py # Image model conversion utilities

├── datasets/                      # Medical datasets and training data│   └── convert_text_to_tflite.py  # Text model conversion utilities

│   ├── DiseaseAndSymptoms.csv     # Disease-symptom mappings with severity├── 📁 datasets/                   # Medical datasets and training data

│   ├── DiseasePrecaution.csv      # Disease-precaution recommendations│   ├── DiseaseAndSymptoms.csv     # Disease-symptom mappings with severity

│   └── SkinDisease/               # Image classification dataset (samples)│   ├── DiseasePrecaution.csv      # Disease-precaution recommendations

├── models/                        # Pre-trained model files│   └── SkinDisease/               # Image classification dataset (samples)

├── docs/                          # Documentation and architecture diagrams├── 📁 models/                     # Pre-trained model files

├── build.gradle                   # Project-level build configuration├── 📁 docs/                       # Documentation and architecture diagrams

├── requirements.txt               # Python dependencies for ML training├── � build.gradle                # Project-level build configuration

└── README.md                      # This file├── 📄 requirements.txt            # Python dependencies for ML training

```└── 📄 README.md                   # This file

```

## AI/ML Pipeline

## 🤖 AI/ML Pipeline

### 1. Image Classification (Skin Disease Detection)

- **Model**: EfficientNetB0 + Custom CNN layers### 1. **Image Classification (Skin Disease Detection)**

- **Input**: 224×224 RGB images- **Model**: EfficientNetB0 + Custom CNN layers

- **Output**: 22 skin disease categories with confidence scores- **Input**: 224×224 RGB images

- **Classes**: Acne, Skin Cancer, Eczema, Psoriasis, Melanoma, etc.- **Output**: 22 skin disease categories with confidence scores

- **Accuracy Target**: >85% on test set- **Classes**: Acne, Skin Cancer, Eczema, Psoriasis, Melanoma, etc.

- **Accuracy Target**: >85% on test set

### 2. Text Classification (Symptom Severity)

- **Model**: LSTM + Dense layers OR DistilBERT### 2. **Text Classification (Symptom Severity)**

- **Input**: Tokenized symptom descriptions (max 128 tokens)- **Model**: LSTM + Dense layers OR DistilBERT

- **Output**: Severity levels (Mild, Moderate, Severe)- **Input**: Tokenized symptom descriptions (max 128 tokens)

- **Features**: Rule-based + ML-based classification- **Output**: Severity levels (Mild, Moderate, Severe)

- **Features**: Rule-based + ML-based classification

### 3. Multimodal Fusion Engine

- **Approach**: Weighted fusion of image and text predictions### 3. **Multimodal Fusion Engine**

- **Weights**: Image (70%) + Text (30%)- **Approach**: Weighted fusion of image and text predictions

- **Output**: Final urgency level (Green/Yellow/Red)- **Weights**: Image (70%) + Text (30%)

- **Decision Logic**: Conservative approach (takes higher urgency)- **Output**: Final urgency level (Green/Yellow/Red)

- **Decision Logic**: Conservative approach (takes higher urgency)

## Quick Start Guide

## � Quick Start Guide

### Prerequisites

### Prerequisites

| Requirement | Version | Purpose |

|-------------|---------|---------|| Requirement | Version | Purpose |

| **Android Studio** | Arctic Fox+ | Android development ||-------------|---------|---------|

| **Kotlin** | 1.5.0+ | App programming language || **Android Studio** | Arctic Fox+ | Android development |

| **Android SDK** | API Level 21+ | Android 5.0+ support || **Kotlin** | 1.5.0+ | App programming language |

| **Python** | 3.8+ | ML training (optional) || **Android SDK** | API Level 21+ | Android 5.0+ support |

| **TensorFlow** | 2.8+ | ML training (optional) || **Python** | 3.8+ | ML training (optional) |

| **Device RAM** | 4GB+ | Smooth app performance || **TensorFlow** | 2.8+ | ML training (optional) |

| **Device RAM** | 4GB+ | Smooth app performance |

### Installation Steps

### 🔧 Installation Steps

#### Step 1: Clone the Repository

```bash#### **Step 1: Clone the Repository**

git clone https://github.com/Arjjun-S/EmergencyTriageApp.git```bash

cd EmergencyTriageAppgit clone https://github.com/yourusername/EmergencyTriageApp.git

```cd EmergencyTriageApp

```

#### Step 2: Open in Android Studio

1. Launch Android Studio#### **Step 2: Open in Android Studio**

2. Click "Open an existing Android Studio project"1. Launch Android Studio

3. Navigate to the `EmergencyTriageApp` folder and select it2. Click "Open an existing Android Studio project"

4. Wait for Gradle sync to complete3. Navigate to the `EmergencyTriageApp` folder and select it

4. Wait for Gradle sync to complete

#### Step 3: Add App Icon (Optional)

**Create a Professional App Icon:**#### **Step 3: Add App Icon (Optional)**

**📱 Create a Professional App Icon:**

Your app needs an icon to look professional on users' devices. The folder structure is ready:

Your app needs an icon to look professional on users' devices. I've created the folder structure for you:

```

app/src/main/res/```

├── mipmap-mdpi/     → ic_launcher.png (48×48 px)app/src/main/res/

├── mipmap-hdpi/     → ic_launcher.png (72×72 px)  ├── mipmap-mdpi/     → ic_launcher.png (48×48 px)

├── mipmap-xhdpi/    → ic_launcher.png (96×96 px)├── mipmap-hdpi/     → ic_launcher.png (72×72 px)  

├── mipmap-xxhdpi/   → ic_launcher.png (144×144 px)├── mipmap-xhdpi/    → ic_launcher.png (96×96 px)

└── mipmap-xxxhdpi/  → ic_launcher.png (192×192 px)├── mipmap-xxhdpi/   → ic_launcher.png (144×144 px)

```└── mipmap-xxxhdpi/  → ic_launcher.png (192×192 px)

```

**Quick Icon Creation:**

1. **Use Android Asset Studio** (Recommended): https://romannurik.github.io/AndroidAssetStudio/icons-launcher.html**🎨 Quick Icon Creation:**

2. **Upload a 512×512** base design (medical theme: red cross, stethoscope, etc.)1. **Use Android Asset Studio** (Recommended): https://romannurik.github.io/AndroidAssetStudio/icons-launcher.html

3. **Download all sizes** and place them in the respective mipmap folders2. **Upload a 512×512** base design (medical theme: red cross, stethoscope, etc.)

4. **Name each file**: `ic_launcher.png`3. **Download all sizes** and place them in the respective mipmap folders

4. **Name each file**: `ic_launcher.png`

**Icon Ideas:** Medical cross + AI circuit, stethoscope + phone, emergency star + mobile device

**💡 Icon Ideas:** Medical cross + AI circuit, stethoscope + phone, emergency star + mobile device

*See detailed guide: [`docs/APP_ICON_GUIDE.md`](docs/APP_ICON_GUIDE.md)*

*See detailed guide: [`docs/APP_ICON_GUIDE.md`](docs/APP_ICON_GUIDE.md)*

#### Step 4: Build and Run

1. Connect an Android device (API 21+) or start an emulator#### **Step 4: Build and Run**

2. Click the "Run" button in Android Studio1. Connect an Android device (API 21+) or start an emulator

3. Select your target device2. Click the "Run" button (▶️) in Android Studio

4. The app will install and launch automatically3. Select your target device

4. The app will install and launch automatically

## Dataset Setup for High-Accuracy Model Training

### 📊 Dataset Setup (Optional - For Training)

The repository includes only sample data for demonstration purposes. To train high-accuracy custom models, you need to download complete datasets.

If you want to train your own models:

### Current Sample Data

- **Disease Symptoms**: 40+ diseases with symptom mappings#### **Step 1: Set Up Python Environment**

- **Precautions**: Disease-specific precautionary measures```bash

- **Skin Images**: 2 sample images from HAM10000 dataset# Create virtual environment

python -m venv venv

### Download Complete Datasets for Production Modelssource venv\Scripts\activate  # On Windows

# source venv/bin/activate    # On Linux/Mac

#### 1. HAM10000 Skin Lesion Dataset (Recommended)

**For high-accuracy skin disease detection:**# Install dependencies

pip install -r requirements.txt

- **Source**: Harvard Dataverse```

- **URL**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

- **Size**: ~10,000 dermatoscopic images#### **Step 2: Download Full Dataset**

- **Classes**: 7 skin lesion typesThe repository includes sample data only. For full training:

- **Format**: JPEG images with metadata

1. **Get HAM10000 Dataset**:

**Download Steps:**   - Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

1. Visit the Harvard Dataverse link above   - Download `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip`

2. Download `HAM10000_images_part_1.zip` (5.5 GB)

3. Download `HAM10000_images_part_2.zip` (5.0 GB)2. **Extract Images**:

4. Download `HAM10000_metadata.csv`   ```bash

5. Extract images to:   # Extract to datasets/SkinDisease/HAM10000_images_part_1/

   ```   # Extract to datasets/SkinDisease/HAM10000_images_part_2/

   datasets/SkinDisease/HAM10000_images_part_1/   ```

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