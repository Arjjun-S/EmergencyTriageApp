# Emergency Triage App# Emergency Triage App# 🏥 Emergency AI Triage App



## Overview


The Emergency AI Triage App is a cutting-edge Android application that combines voice recognition, image analysis, and text processing to provide intelligent medical triage decisions. Using on-device machine learning, the app classifies medical emergencies and provides urgency flags with recommended actions.


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
# Emergency Triage App

**A Multimodal AI-Powered Emergency Medical Triage System for Android**

## Overview

The Emergency Triage App combines voice input, image analysis, and structured symptom text to assist with preliminary medical triage. All inference runs on-device using TensorFlow Lite for privacy and low-latency operation. The system produces an urgency level (Green / Yellow / Red) and suggested next-step actions.

## Key Features

| Feature | Description |
|---------|-------------|
| Voice-to-Text | Capture spoken symptoms via Android speech recognition |
| Image Classification | Skin condition analysis using a CNN model |
| Multimodal Fusion | Weighted combination of image + text model outputs |
| Urgency Classification | Three-tier urgency result (Green / Yellow / Red) |
| Precaution Lookup | Maps predicted condition to precaution suggestions |
| Telehealth Hook | Placeholder for integration with remote care services |
| Offline Execution | All ML inference happens fully on-device |

## Project Structure

```
EmergencyTriageApp/
├── app/
│   ├── src/main/
│   │   ├── java/com/example/emergencytriage/
│   │   │   ├── MainActivity.kt
│   │   │   ├── ui/screens/
│   │   │   ├── ml/
│   │   │   ├── data/models/
│   │   │   └── utils/
│   │   ├── assets/               # TFLite models & label files
│   │   └── res/                  # Layouts, drawables, mipmap icons
│   └── build.gradle
├── ml/                           # Python model training & conversion
│   ├── train_image_model.py
│   ├── train_text_model.py
│   ├── convert_image_to_tflite.py
│   └── convert_text_to_tflite.py
├── datasets/                     # Sample CSVs + sample images
│   ├── DiseaseAndSymptoms.csv
│   ├── DiseasePrecaution.csv
│   └── SkinDisease/
├── models/                       # (Optional) exported .h5 or .tflite
├── docs/                         # Additional documentation
├── requirements.txt              # Python dependencies
└── README.md
```

## ML Pipeline Summary

### 1. Image Model
- Architecture: EfficientNetB0 (base) + custom dense head
- Input: 224x224 RGB
- Output: Condition class probabilities
- Goal Accuracy: >85% with full dataset

### 2. Text Model
- Architecture: LSTM (or optional transformer) over tokenized symptom text
- Output: Severity / condition category probability distribution

### 3. Fusion
- Weighted blend (default Image 0.7, Text 0.3)
- Produces final urgency level + top condition

## Quick Start (App)

1. Install / open Android Studio (Arctic Fox or later)
2. Clone repository:
    ```bash
    git clone https://github.com/Arjjun-S/EmergencyTriageApp.git
    cd EmergencyTriageApp
    ```
3. Open the project root in Android Studio
4. Let Gradle sync finish
5. (Optional) Add launcher icons (see below)
6. Run on a device/emulator (API 21+)

## Launcher Icon (Optional)

Place appropriately sized `ic_launcher.png` files in:
```
app/src/main/res/
   mipmap-mdpi/
   mipmap-hdpi/
   mipmap-xhdpi/
   mipmap-xxhdpi/
   mipmap-xxxhdpi/
```
Use Android Asset Studio to generate sizes from a 512x512 base graphic.

## High-Accuracy Model Training

The repository only includes minimal sample data. For production-quality accuracy you must download full datasets.

### Recommended Skin Dataset (HAM10000)
Source: Harvard Dataverse  
URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Download:
1. `HAM10000_images_part_1.zip`
2. `HAM10000_images_part_2.zip`
3. `HAM10000_metadata.csv`

Extract into:
```
datasets/SkinDisease/HAM10000_images_part_1/
datasets/SkinDisease/HAM10000_images_part_2/
datasets/SkinDisease/HAM10000_metadata.csv
```

### Alternative Datasets
- ISIC 2019 Challenge: https://challenge.isic-archive.com/data/
- DermNet: http://www.dermnet.com/
- PH2 (melanoma focus): https://www.fc.up.pt/addi/ph2%20database.html

### Python Environment Setup
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Training Commands
```bash
cd ml
python train_image_model.py
python train_text_model.py
python convert_image_to_tflite.py
python convert_text_to_tflite.py
```

### Expected (Full Dataset) Performance
- Image classification: 85–92% accuracy
- Text classification: 88–95% accuracy
- Training time: GPU 2–4h, CPU 8–12h

## Using the App (Workflow)
1. Launch app and grant Camera / Microphone permissions
2. (Optional) Capture or select a skin image
3. Press record to dictate symptoms, then stop
4. Tap analyze to run multimodal inference
5. Review urgency, top condition, and precaution suggestions

## Configuration Reference
- Min SDK: 21
- Target SDK: 34
- Permissions: CAMERA, RECORD_AUDIO, INTERNET
- Image input size: 224x224
- Default batch size (training): 32

## Troubleshooting
| Issue | Resolution |
|-------|-----------|
| App crash on launch | Verify permissions granted in system settings |
| Model not found | Confirm TFLite files exist under `app/src/main/assets/models` |
| Poor image results | Use clear, well-lit, centered images; expand training data |
| Speech not captured | Check microphone permission and retry in quiet area |
| Build failure | Clean/Rebuild project; ensure Gradle sync completed |

## Contributing
Suggestions / PRs welcome (UI polish, model improvements, accessibility, localization).

## Support
- Issues: https://github.com/Arjjun-S/EmergencyTriageApp/issues
- Discussions: (enable in GitHub repository settings if needed)

---
Medical Disclaimer: This application provides informational triage assistance only and is not a substitute for professional medical advice, diagnosis, or treatment.

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