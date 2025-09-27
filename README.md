# Emergency Triage App# Emergency Triage App# 🏥 Emergency AI Triage App



[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)

[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org/)

[![TensorFlow Lite](https://img.shields.io/badge/ML-TensorFlow%20Lite-orange.svg)](https://www.tensorflow.org/lite)[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)



**A Complete Multimodal AI-Powered Emergency Medical Triage System for Android**[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org/)[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org/)



## Overview[![TensorFlow Lite](https://img.shields.io/badge/ML-TensorFlow%20Lite-orange.svg)](https://www.tensorflow.org/lite)[![TensorFlow Lite](https://img.shields.io/badge/ML-TensorFlow%20Lite-orange.svg)](https://www.tensorflow.org/lite)



The Emergency AI Triage App is a cutting-edge Android application that combines voice recognition, image analysis, and text processing to provide intelligent medical triage decisions. Using on-device machine learning, the app classifies medical emergencies and provides urgency flags with recommended actions.[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



### Key Features**A Complete Multimodal AI-Powered Emergency Medical Triage System for Android**



| Feature | Description |> **A Complete Multimodal AI-Powered Emergency Medical Triage System for Android**

|---------|-------------|

| **Voice-to-Text** | Record symptoms using Android's SpeechRecognizer |## Overview

| **Image Classification** | AI-powered skin disease detection using CNN |

| **Multimodal Fusion** | Combines image and text analysis for comprehensive triage |## 🎯 Overview

| **Urgency Classification** | Green / Yellow / Red urgency levels |

| **Disease Precautions** | Automated precaution recommendations |The Emergency AI Triage App is a cutting-edge Android application that combines voice recognition, image analysis, and text processing to provide intelligent medical triage decisions. Using on-device machine learning, the app classifies medical emergencies and provides urgency flags with recommended actions.

| **Telehealth Integration** | Direct connection to telehealth services |

| **Offline-First** | All AI processing happens on-device using TensorFlow Lite |The Emergency AI Triage App is a cutting-edge Android application that combines **voice recognition**, **image analysis**, and **text processing** to provide intelligent medical triage decisions. Using on-device machine learning, the app classifies medical emergencies and provides urgency flags with recommended actions.



## App Interface### Key Features



```### ✨ Key Features

[Main Screen]     [Recording]      [Analysis]       [Results]

┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐| Feature | Description |

│  Record     │   │  Speaking   │   │  Analyzing  │   │  URGENT     │

│  Capture    │   │ "I have..." │   │   Please    │   │ Skin Cancer │|---------|-------------|| Feature | Description |

│  Gallery    │   │             │   │   wait...   │   │ 92% Conf.   │

│             │   │  Stop       │   │             │   │  Actions    │| **Voice-to-Text** | Record symptoms using Android's SpeechRecognizer ||---------|-------------|

│  Analyze    │   │             │   │             │   │  Call Dr    │

└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘| **Image Classification** | AI-powered skin disease detection using CNN || 🎤 **Voice-to-Text** | Record symptoms using Android's SpeechRecognizer |

```

| **Multimodal Fusion** | Combines image and text analysis for comprehensive triage || 📸 **Image Classification** | AI-powered skin disease detection using CNN |

## Project Structure

| **Urgency Classification** | Green / Yellow / Red urgency levels || 🧠 **Multimodal Fusion** | Combines image and text analysis for comprehensive triage |

```

EmergencyTriageApp/| **Disease Precautions** | Automated precaution recommendations || 🚨 **Urgency Classification** | 🟢 Green / 🟡 Yellow / 🔴 Red urgency levels |

├── app/                           # Android application module

│   ├── src/main/| **Telehealth Integration** | Direct connection to telehealth services || 💊 **Disease Precautions** | Automated precaution recommendations |

│   │   ├── java/com/example/emergencytriage/

│   │   │   ├── MainActivity.kt    # Main app entry point| **Offline-First** | All AI processing happens on-device using TensorFlow Lite || 🏥 **Telehealth Integration** | Direct connection to telehealth services |

│   │   │   ├── ui/screens/        # UI components

│   │   │   ├── ml/                # Machine learning processors| 📱 **Offline-First** | All AI processing happens on-device using TensorFlow Lite |

│   │   │   ├── data/models/       # Data classes and models

│   │   │   └── utils/             # Utility classes## App Interface

│   │   ├── assets/                # TensorFlow Lite models & resources

│   │   └── res/                   # Android resources (layouts, icons, etc.)## 📱 App Screenshots

│   │       └── mipmap-*/          # App icons (multiple resolutions)

│   └── build.gradle               # App-level build configuration```

├── ml/                            # Python ML training scripts

│   ├── train_image_model.py       # CNN training for skin diseases[Main Screen]     [Recording]      [Analysis]       [Results]```

│   ├── train_text_model.py        # Text classifier for symptom severity

│   ├── convert_image_to_tflite.py # Image model conversion utilities┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐[Main Screen]     [Recording]      [Analysis]       [Results]

│   └── convert_text_to_tflite.py  # Text model conversion utilities

├── datasets/                      # Medical datasets and training data│  Record     │   │  Speaking   │   │  Analyzing  │   │  URGENT     │┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐

│   ├── DiseaseAndSymptoms.csv     # Disease-symptom mappings with severity

│   ├── DiseasePrecaution.csv      # Disease-precaution recommendations│  Capture    │   │ "I have..." │   │   Please    │   │ Skin Cancer ││  🎤 Record  │   │ 🔴 Speaking │   │ ⏳ Analyzing│   │ 🔴 URGENT   │

│   └── SkinDisease/               # Image classification dataset (samples)

├── models/                        # Pre-trained model files│  Gallery    │   │             │   │   wait...   │   │ 92% Conf.   ││  📸 Capture │   │ "I have..." │   │   Please    │   │ Skin Cancer │

├── docs/                          # Documentation and architecture diagrams

├── build.gradle                   # Project-level build configuration│             │   │  Stop       │   │             │   │  Actions    ││  🖼️ Gallery  │   │             │   │   wait...   │   │ 92% Conf.   │

├── requirements.txt               # Python dependencies for ML training

└── README.md                      # This file│  Analyze    │   │             │   │             │   │  Call Dr    ││             │   │ 🔇 Stop     │   │             │   │ 📋 Actions  │

```

└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘│ 🔍 Analyze  │   │             │   │             │   │ 🏥 Call Dr  │

## AI/ML Pipeline

```└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘

### 1. Image Classification (Skin Disease Detection)

- **Model**: EfficientNetB0 + Custom CNN layers```

- **Input**: 224×224 RGB images

- **Output**: 22 skin disease categories with confidence scores## Project Structure

- **Classes**: Acne, Skin Cancer, Eczema, Psoriasis, Melanoma, etc.

- **Accuracy Target**: >85% on test set## 🏗️ Project Structure



### 2. Text Classification (Symptom Severity)```

- **Model**: LSTM + Dense layers OR DistilBERT

- **Input**: Tokenized symptom descriptions (max 128 tokens)EmergencyTriageApp/```

- **Output**: Severity levels (Mild, Moderate, Severe)

- **Features**: Rule-based + ML-based classification├── app/                           # Android application moduleEmergencyTriageApp/



### 3. Multimodal Fusion Engine│   ├── src/main/├── 📁 app/                        # Android application module

- **Approach**: Weighted fusion of image and text predictions

- **Weights**: Image (70%) + Text (30%)│   │   ├── java/com/example/emergencytriage/│   ├── src/main/

- **Output**: Final urgency level (Green/Yellow/Red)

- **Decision Logic**: Conservative approach (takes higher urgency)│   │   │   ├── MainActivity.kt    # Main app entry point│   │   ├── java/com/example/emergencytriage/



## Quick Start Guide│   │   │   ├── ui/screens/        # UI components│   │   │   ├── MainActivity.kt    # Main app entry point



### Prerequisites│   │   │   ├── ml/                # Machine learning processors│   │   │   ├── ui/screens/        # UI components



| Requirement | Version | Purpose |│   │   │   ├── data/models/       # Data classes and models│   │   │   ├── ml/                # Machine learning processors

|-------------|---------|---------|

| **Android Studio** | Arctic Fox+ | Android development |│   │   │   └── utils/             # Utility classes│   │   │   ├── data/models/       # Data classes and models

| **Kotlin** | 1.5.0+ | App programming language |

| **Android SDK** | API Level 21+ | Android 5.0+ support |│   │   ├── assets/                # TensorFlow Lite models & resources│   │   │   └── utils/             # Utility classes

| **Python** | 3.8+ | ML training (optional) |

| **TensorFlow** | 2.8+ | ML training (optional) |│   │   └── res/                   # Android resources (layouts, icons, etc.)│   │   ├── assets/                # TensorFlow Lite models & resources

| **Device RAM** | 4GB+ | Smooth app performance |

│   │       └── mipmap-*/          # App icons (multiple resolutions)│   │   └── res/                   # Android resources (layouts, icons, etc.)

### Installation Steps

│   └── build.gradle               # App-level build configuration│   │       └── mipmap-*/          # App icons (multiple resolutions)

#### Step 1: Clone the Repository

```bash├── ml/                            # Python ML training scripts│   └── build.gradle               # App-level build configuration

git clone https://github.com/Arjjun-S/EmergencyTriageApp.git

cd EmergencyTriageApp│   ├── train_image_model.py       # CNN training for skin diseases├── 📁 ml/                         # Python ML training scripts

```

│   ├── train_text_model.py        # Text classifier for symptom severity│   ├── train_image_model.py       # CNN training for skin diseases

#### Step 2: Open in Android Studio

1. Launch Android Studio│   ├── convert_image_to_tflite.py # Image model conversion utilities│   ├── train_text_model.py        # Text classifier for symptom severity

2. Click "Open an existing Android Studio project"

3. Navigate to the `EmergencyTriageApp` folder and select it│   └── convert_text_to_tflite.py  # Text model conversion utilities│   ├── convert_image_to_tflite.py # Image model conversion utilities

4. Wait for Gradle sync to complete

├── datasets/                      # Medical datasets and training data│   └── convert_text_to_tflite.py  # Text model conversion utilities

#### Step 3: Add App Icon (Optional)

**Create a Professional App Icon:**│   ├── DiseaseAndSymptoms.csv     # Disease-symptom mappings with severity├── 📁 datasets/                   # Medical datasets and training data



Your app needs an icon to look professional on users' devices. The folder structure is ready:│   ├── DiseasePrecaution.csv      # Disease-precaution recommendations│   ├── DiseaseAndSymptoms.csv     # Disease-symptom mappings with severity



```│   └── SkinDisease/               # Image classification dataset (samples)│   ├── DiseasePrecaution.csv      # Disease-precaution recommendations

app/src/main/res/

├── mipmap-mdpi/     → ic_launcher.png (48×48 px)├── models/                        # Pre-trained model files│   └── SkinDisease/               # Image classification dataset (samples)

├── mipmap-hdpi/     → ic_launcher.png (72×72 px)  

├── mipmap-xhdpi/    → ic_launcher.png (96×96 px)├── docs/                          # Documentation and architecture diagrams├── 📁 models/                     # Pre-trained model files

├── mipmap-xxhdpi/   → ic_launcher.png (144×144 px)

└── mipmap-xxxhdpi/  → ic_launcher.png (192×192 px)├── build.gradle                   # Project-level build configuration├── 📁 docs/                       # Documentation and architecture diagrams

```

├── requirements.txt               # Python dependencies for ML training├── � build.gradle                # Project-level build configuration

**Quick Icon Creation:**

1. **Use Android Asset Studio** (Recommended): https://romannurik.github.io/AndroidAssetStudio/icons-launcher.html└── README.md                      # This file├── 📄 requirements.txt            # Python dependencies for ML training

2. **Upload a 512×512** base design (medical theme: red cross, stethoscope, etc.)

3. **Download all sizes** and place them in the respective mipmap folders```└── 📄 README.md                   # This file

4. **Name each file**: `ic_launcher.png`

```

**Icon Ideas:** Medical cross + AI circuit, stethoscope + phone, emergency star + mobile device

## AI/ML Pipeline

*See detailed guide: [`docs/APP_ICON_GUIDE.md`](docs/APP_ICON_GUIDE.md)*

## 🤖 AI/ML Pipeline

#### Step 4: Build and Run

1. Connect an Android device (API 21+) or start an emulator### 1. Image Classification (Skin Disease Detection)

2. Click the "Run" button in Android Studio

3. Select your target device- **Model**: EfficientNetB0 + Custom CNN layers### 1. **Image Classification (Skin Disease Detection)**

4. The app will install and launch automatically

- **Input**: 224×224 RGB images- **Model**: EfficientNetB0 + Custom CNN layers

## Dataset Setup for High-Accuracy Model Training

- **Output**: 22 skin disease categories with confidence scores- **Input**: 224×224 RGB images

The repository includes only sample data for demonstration purposes. To train high-accuracy custom models, you need to download complete datasets.

- **Classes**: Acne, Skin Cancer, Eczema, Psoriasis, Melanoma, etc.- **Output**: 22 skin disease categories with confidence scores

### Current Sample Data

- **Disease Symptoms**: 40+ diseases with symptom mappings- **Accuracy Target**: >85% on test set- **Classes**: Acne, Skin Cancer, Eczema, Psoriasis, Melanoma, etc.

- **Precautions**: Disease-specific precautionary measures

- **Skin Images**: 2 sample images from HAM10000 dataset- **Accuracy Target**: >85% on test set



### Download Complete Datasets for Production Models### 2. Text Classification (Symptom Severity)



#### 1. HAM10000 Skin Lesion Dataset (Recommended)- **Model**: LSTM + Dense layers OR DistilBERT### 2. **Text Classification (Symptom Severity)**

**For high-accuracy skin disease detection:**

- **Input**: Tokenized symptom descriptions (max 128 tokens)- **Model**: LSTM + Dense layers OR DistilBERT

- **Source**: Harvard Dataverse

- **URL**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T- **Output**: Severity levels (Mild, Moderate, Severe)- **Input**: Tokenized symptom descriptions (max 128 tokens)

- **Size**: ~10,000 dermatoscopic images

- **Classes**: 7 skin lesion types- **Features**: Rule-based + ML-based classification- **Output**: Severity levels (Mild, Moderate, Severe)

- **Format**: JPEG images with metadata

- **Features**: Rule-based + ML-based classification

**Download Steps:**

1. Visit the Harvard Dataverse link above### 3. Multimodal Fusion Engine

2. Download `HAM10000_images_part_1.zip` (5.5 GB)

3. Download `HAM10000_images_part_2.zip` (5.0 GB)- **Approach**: Weighted fusion of image and text predictions### 3. **Multimodal Fusion Engine**

4. Download `HAM10000_metadata.csv`

5. Extract images to:- **Weights**: Image (70%) + Text (30%)- **Approach**: Weighted fusion of image and text predictions

   ```

   datasets/SkinDisease/HAM10000_images_part_1/- **Output**: Final urgency level (Green/Yellow/Red)- **Weights**: Image (70%) + Text (30%)

   datasets/SkinDisease/HAM10000_images_part_2/

   ```- **Decision Logic**: Conservative approach (takes higher urgency)- **Output**: Final urgency level (Green/Yellow/Red)



#### 2. Alternative Datasets for Enhanced Training- **Decision Logic**: Conservative approach (takes higher urgency)



**ISIC 2019 Challenge Dataset:**## Quick Start Guide

- **URL**: https://challenge.isic-archive.com/data/

- **Size**: 25,331 images## � Quick Start Guide

- **Classes**: 8 diagnostic categories

- **Use Case**: More diverse skin lesion types### Prerequisites



**DermNet Dataset:**### Prerequisites

- **URL**: http://www.dermnet.com/

- **Size**: 23,000+ images| Requirement | Version | Purpose |

- **Classes**: 23 skin condition classes

- **Use Case**: Broader skin condition coverage|-------------|---------|---------|| Requirement | Version | Purpose |



**PH2 Dataset:**| **Android Studio** | Arctic Fox+ | Android development ||-------------|---------|---------|

- **URL**: https://www.fc.up.pt/addi/ph2%20database.html

- **Size**: 200 dermoscopic images| **Kotlin** | 1.5.0+ | App programming language || **Android Studio** | Arctic Fox+ | Android development |

- **Use Case**: Melanoma detection focus

| **Android SDK** | API Level 21+ | Android 5.0+ support || **Kotlin** | 1.5.0+ | App programming language |

### Training Setup with Complete Datasets

| **Python** | 3.8+ | ML training (optional) || **Android SDK** | API Level 21+ | Android 5.0+ support |

#### Step 1: Python Environment Setup

```bash| **TensorFlow** | 2.8+ | ML training (optional) || **Python** | 3.8+ | ML training (optional) |

# Create virtual environment

python -m venv venv| **Device RAM** | 4GB+ | Smooth app performance || **TensorFlow** | 2.8+ | ML training (optional) |



# Activate environment| **Device RAM** | 4GB+ | Smooth app performance |

# Windows:

venv\Scripts\activate### Installation Steps

# Linux/Mac:

source venv/bin/activate### 🔧 Installation Steps



# Install dependencies#### Step 1: Clone the Repository

pip install -r requirements.txt

``````bash#### **Step 1: Clone the Repository**



#### Step 2: Configure Training Datagit clone https://github.com/Arjjun-S/EmergencyTriageApp.git```bash

```bash

# Ensure datasets are in correct structure:cd EmergencyTriageAppgit clone https://github.com/yourusername/EmergencyTriageApp.git

datasets/

├── DiseaseAndSymptoms.csv```cd EmergencyTriageApp

├── DiseasePrecaution.csv

└── SkinDisease/```

    ├── HAM10000_metadata.csv

    ├── HAM10000_images_part_1/  # 5,000+ images#### Step 2: Open in Android Studio

    └── HAM10000_images_part_2/  # 5,000+ images

```1. Launch Android Studio#### **Step 2: Open in Android Studio**



#### Step 3: Train High-Accuracy Models2. Click "Open an existing Android Studio project"1. Launch Android Studio

```bash

cd ml/3. Navigate to the `EmergencyTriageApp` folder and select it2. Click "Open an existing Android Studio project"



# Train image classification model with full dataset4. Wait for Gradle sync to complete3. Navigate to the `EmergencyTriageApp` folder and select it

python train_image_model.py

4. Wait for Gradle sync to complete

# Train text classification model

python train_text_model.py#### Step 3: Add App Icon (Optional)



# Convert trained models to TensorFlow Lite**Create a Professional App Icon:**#### **Step 3: Add App Icon (Optional)**

python convert_image_to_tflite.py

python convert_text_to_tflite.py**📱 Create a Professional App Icon:**

```

Your app needs an icon to look professional on users' devices. The folder structure is ready:

### Expected Performance with Full Datasets

- **Skin Disease Classification**: 85-92% accuracyYour app needs an icon to look professional on users' devices. I've created the folder structure for you:

- **Text Symptom Analysis**: 88-95% accuracy

- **Training Time**: 2-4 hours on GPU, 8-12 hours on CPU```

- **Model Size**: 15-25 MB per model (optimized for mobile)

app/src/main/res/```

**Note**: Training with complete datasets significantly improves model accuracy and real-world performance compared to sample data.

├── mipmap-mdpi/     → ic_launcher.png (48×48 px)app/src/main/res/

## How to Use the App

├── mipmap-hdpi/     → ic_launcher.png (72×72 px)  ├── mipmap-mdpi/     → ic_launcher.png (48×48 px)

### 1. Launch the App

- Open the Emergency Triage App on your Android device├── mipmap-xhdpi/    → ic_launcher.png (96×96 px)├── mipmap-hdpi/     → ic_launcher.png (72×72 px)  

- Grant microphone and camera permissions when prompted

├── mipmap-xxhdpi/   → ic_launcher.png (144×144 px)├── mipmap-xhdpi/    → ic_launcher.png (96×96 px)

### 2. Record Symptoms (Voice Input)

- Tap the **Record** button└── mipmap-xxxhdpi/  → ic_launcher.png (192×192 px)├── mipmap-xxhdpi/   → ic_launcher.png (144×144 px)

- Speak clearly about your symptoms

- Example: *"I have a red rash on my arm with itching and swelling"*```└── mipmap-xxxhdpi/  → ic_launcher.png (192×192 px)

- Tap **Stop** when finished

```

### 3. Capture/Upload Image (Optional)

- Tap **Capture** to take a photo with camera**Quick Icon Creation:**

- Or tap **Gallery** to select an existing image

- Ensure the image is clear and well-lit1. **Use Android Asset Studio** (Recommended): https://romannurik.github.io/AndroidAssetStudio/icons-launcher.html**🎨 Quick Icon Creation:**



### 4. Get AI Analysis2. **Upload a 512×512** base design (medical theme: red cross, stethoscope, etc.)1. **Use Android Asset Studio** (Recommended): https://romannurik.github.io/AndroidAssetStudio/icons-launcher.html

- Tap **Analyze** to start processing

- The AI will analyze both text and image (if provided)3. **Download all sizes** and place them in the respective mipmap folders2. **Upload a 512×512** base design (medical theme: red cross, stethoscope, etc.)

- Wait for results (usually 2-5 seconds)

4. **Name each file**: `ic_launcher.png`3. **Download all sizes** and place them in the respective mipmap folders

### 5. Review Results

- **Urgency Level**: Green (Low) / Yellow (Medium) / Red (High)4. **Name each file**: `ic_launcher.png`

- **Confidence Score**: AI's confidence in the diagnosis

- **Recommended Actions**: Specific steps to take**Icon Ideas:** Medical cross + AI circuit, stethoscope + phone, emergency star + mobile device

- **Precautions**: Disease-specific precautionary measures

**💡 Icon Ideas:** Medical cross + AI circuit, stethoscope + phone, emergency star + mobile device

### 6. Take Action

- Follow the recommended actions*See detailed guide: [`docs/APP_ICON_GUIDE.md`](docs/APP_ICON_GUIDE.md)*

- Use **Call Doctor** for telehealth if urgent

- Save results for medical consultation*See detailed guide: [`docs/APP_ICON_GUIDE.md`](docs/APP_ICON_GUIDE.md)*



## Configuration#### Step 4: Build and Run



### Android App Configuration1. Connect an Android device (API 21+) or start an emulator#### **Step 4: Build and Run**

- **Minimum SDK**: API 21 (Android 5.0)

- **Target SDK**: API 34 (Android 14)2. Click the "Run" button in Android Studio1. Connect an Android device (API 21+) or start an emulator

- **Permissions**: CAMERA, RECORD_AUDIO, INTERNET

- **Model Size**: ~50MB total (optimized for mobile)3. Select your target device2. Click the "Run" button (▶️) in Android Studio



### ML Training Configuration4. The app will install and launch automatically3. Select your target device

- **Image Size**: 224×224×3

- **Batch Size**: 324. The app will install and launch automatically

- **Epochs**: 50-100

- **Learning Rate**: 0.001## Dataset Setup for High-Accuracy Model Training

- **Optimization**: Adam optimizer

### 📊 Dataset Setup (Optional - For Training)

## Testing

The repository includes only sample data for demonstration purposes. To train high-accuracy custom models, you need to download complete datasets.

### Unit Tests

```bashIf you want to train your own models:

# Run Android unit tests

./gradlew test### Current Sample Data



# Run instrumented tests  - **Disease Symptoms**: 40+ diseases with symptom mappings#### **Step 1: Set Up Python Environment**

./gradlew connectedAndroidTest

```- **Precautions**: Disease-specific precautionary measures```bash



### ML Model Tests- **Skin Images**: 2 sample images from HAM10000 dataset# Create virtual environment

```bash

cd ml/python -m venv venv

python -m pytest tests/

```### Download Complete Datasets for Production Modelssource venv\Scripts\activate  # On Windows



## Troubleshooting# source venv/bin/activate    # On Linux/Mac



### Common Issues#### 1. HAM10000 Skin Lesion Dataset (Recommended)



| Issue | Solution |**For high-accuracy skin disease detection:**# Install dependencies

|-------|----------|

| **App crashes on startup** | Check permissions in Settings > Apps > Emergency Triage |pip install -r requirements.txt

| **Models not loading** | Ensure TensorFlow Lite files are in `app/src/main/assets/` |

| **Poor image classification** | Use well-lit, clear images; retrain with more data |- **Source**: Harvard Dataverse```

| **Voice recognition fails** | Check microphone permissions and speak clearly |

| **Build errors** | Clean project: `Build > Clean Project` in Android Studio |- **URL**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T



### Performance Optimization- **Size**: ~10,000 dermatoscopic images#### **Step 2: Download Full Dataset**

- **RAM Usage**: ~200-300MB typical usage

- **Storage**: ~100MB for app + models- **Classes**: 7 skin lesion typesThe repository includes sample data only. For full training:

- **Battery**: Minimal impact with on-device processing

- **Network**: Only required for telehealth features- **Format**: JPEG images with metadata



## Contributing1. **Get HAM10000 Dataset**:



We welcome contributions! Areas for contribution:**Download Steps:**   - Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

- Additional disease categories

- UI/UX improvements  1. Visit the Harvard Dataverse link above   - Download `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip`

- ML model enhancements

- Internationalization2. Download `HAM10000_images_part_1.zip` (5.5 GB)

- iOS version

- Accessibility features3. Download `HAM10000_images_part_2.zip` (5.0 GB)2. **Extract Images**:



## Support4. Download `HAM10000_metadata.csv`   ```bash



- **Issues**: [GitHub Issues](https://github.com/Arjjun-S/EmergencyTriageApp/issues)5. Extract images to:   # Extract to datasets/SkinDisease/HAM10000_images_part_1/

- **Discussions**: [GitHub Discussions](https://github.com/Arjjun-S/EmergencyTriageApp/discussions)

   ```   # Extract to datasets/SkinDisease/HAM10000_images_part_2/

---

   datasets/SkinDisease/HAM10000_images_part_1/   ```

**Medical Disclaimer**: This app is for informational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical emergencies.
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