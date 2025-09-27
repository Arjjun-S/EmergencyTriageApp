# Image Classification Examples

This folder contains sample images demonstrating the Emergency AI Triage App's image classification capabilities.

## 📸 Sample Classifications

### Skin Disease Detection Examples

The app can classify various skin conditions. Here are some example categories:

1. **Melanoma** (High Priority - 🔴 Red)
   - Dark, irregular moles
   - Asymmetric shape
   - Color variations
   - Immediate medical attention required

2. **Basal Cell Carcinoma** (Medium Priority - 🟡 Yellow)
   - Pink or red patches
   - Shiny bumps
   - Sores that don't heal
   - Dermatologist consultation recommended

3. **Acne** (Low Priority - 🟢 Green)
   - Pimples, blackheads, whiteheads
   - Inflammatory lesions
   - Common skin condition
   - Over-the-counter treatments available

4. **Eczema/Dermatitis** (Medium Priority - 🟡 Yellow)
   - Red, itchy, inflamed patches
   - Dry, scaly skin
   - May require prescription treatment

## 🔍 Image Quality Guidelines

For best results when using the app:

- ✅ **Good lighting** - Natural daylight preferred
- ✅ **Clear focus** - Sharp, not blurry images
- ✅ **Close-up view** - Fill most of the frame
- ✅ **Minimal background** - Focus on affected area
- ✅ **Multiple angles** - If possible, take several photos

## 🚫 Image Quality Issues to Avoid

- ❌ **Poor lighting** - Too dark or harsh shadows
- ❌ **Blurry images** - Camera shake or poor focus
- ❌ **Too far away** - Affected area is too small
- ❌ **Obstructed view** - Hair, clothing blocking the area
- ❌ **Extreme angles** - Very tilted or distorted perspective

## 🎯 Accuracy Notes

- The AI model achieves >85% accuracy on validation datasets
- Confidence scores help users understand prediction reliability
- Always consult healthcare professionals for definitive diagnosis
- This tool is for screening and triage purposes only

## 📊 Supported Skin Conditions

The current model can classify 22 different skin conditions:

1. Acne
2. Basal Cell Carcinoma
3. Melanoma
4. Benign Keratosis
5. Dermatofibroma
6. Melanocytic Nevus
7. Pyogenic Granuloma
8. Seborrheic Keratosis
9. Squamous Cell Carcinoma
10. Vascular Lesion
11. Eczema/Dermatitis
12. Psoriasis
13. Rosacea
14. Vitiligo
15. Alopecia
16. Cellulitis
17. Herpes
18. Impetigo
19. Ringworm
20. Scabies
21. Warts
22. Other/Unknown

## 🔬 Model Information

- **Architecture**: EfficientNetB0 + Custom CNN layers
- **Input Size**: 224×224×3 RGB images
- **Model Size**: ~45MB (optimized for mobile)
- **Inference Time**: <500ms on modern Android devices
- **Framework**: TensorFlow Lite

---

**Note**: Actual demo images will be added when this becomes a public repository. For privacy and copyright reasons, medical images are not included in this documentation version.