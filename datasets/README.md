# Datasets Folder

This folder contains the training datasets for the Emergency AI Triage App.

## 📊 Available Datasets

### 1. Disease and Symptoms Dataset
- **File**: `DiseaseAndSymptoms.csv`
- **Purpose**: Text classification training for symptom severity assessment  
- **Records**: ~40 diseases with associated symptoms
- **Columns**: Disease, Symptom_1 to Symptom_17, Severity Level

### 2. Disease Precautions Dataset  
- **File**: `DiseasePrecaution.csv`
- **Purpose**: Automated precaution recommendations
- **Records**: Precautionary measures for each disease
- **Columns**: Disease, Precaution_1 to Precaution_4

### 3. Skin Disease Image Dataset
- **Location**: `SkinDisease/HAM10000_images_part_1/`
- **Purpose**: Image classification training for skin condition detection
- **Format**: JPEG images (224×224 recommended)
- **Note**: Only 2 sample images included as placeholders

## 🚀 Getting Full Dataset

The skin disease dataset contains sample images only. For training:

1. **Download HAM10000 Dataset**: 
   - Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
   - Download HAM10000_images_part_1.zip and HAM10000_images_part_2.zip

2. **Extract Images**:
   ```bash
   # Extract to datasets/SkinDisease/HAM10000_images_part_1/
   # Extract to datasets/SkinDisease/HAM10000_images_part_2/
   ```

3. **Organize by Disease Categories** (for training):
   ```
   SkinDisease/
   ├── train/
   │   ├─ melanoma/
   │   ├─ basal_cell_carcinoma/  
   │   ├─ actinic_keratosis/
   │   └─ ... (other categories)
   └── test/
       ├─ melanoma/
       └─ ... (same categories)
   ```

## 📝 Dataset Statistics

- **Total Diseases**: 40+
- **Skin Conditions**: 22 categories
- **Image Resolution**: 224×224 (recommended)
- **Training Split**: 80/20 (train/validation)

## 🔧 Usage in Training

Refer to the training scripts in the `ml/` folder:
- `train_text_model.py` - Uses CSV datasets
- `train_image_model.py` - Uses image dataset