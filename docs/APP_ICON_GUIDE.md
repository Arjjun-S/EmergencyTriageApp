# Android App Icon Guide

## Where to Save Your App Icon Images

Your Emergency AI Triage App icon should be saved in the following directories with these exact sizes:

```
app/src/main/res/
├── mipmap-mdpi/
│   └── ic_launcher.png          (48×48 pixels)
├── mipmap-hdpi/
│   └── ic_launcher.png          (72×72 pixels)
├── mipmap-xhdpi/
│   └── ic_launcher.png          (96×96 pixels)
├── mipmap-xxhdpi/
│   └── ic_launcher.png          (144×144 pixels)
└── mipmap-xxxhdpi/
    └── ic_launcher.png          (192×192 pixels)
```

## Icon Design Requirements

### Design Guidelines
- **Theme**: Medical/Healthcare related
- **Style**: Modern, clean, professional
- **Colors**: Consider using medical colors (red cross, blue, green)
- **Symbol**: Could include:
  - Medical cross
  - Magnifying glass (for AI analysis)
  - Phone with medical symbol
  - Brain icon (for AI)
  - Heart with pulse line

### Technical Requirements
- **Format**: PNG (recommended) or WebP
- **Background**: Should work on light and dark backgrounds
- **Shape**: Square (Android will apply appropriate masking)
- **Padding**: Leave 10% padding around the icon content
- **No text**: Icons should be symbol-based, not text-based

## Icon Size Chart

| Density | Directory | Size (px) | Use Case |
|---------|-----------|-----------|----------|
| mdpi    | mipmap-mdpi    | 48×48   | Medium density screens |
| hdpi    | mipmap-hdpi    | 72×72   | High density screens |
| xhdpi   | mipmap-xhdpi   | 96×96   | Extra high density |
| xxhdpi  | mipmap-xxhdpi  | 144×144 | Extra extra high density |
| xxxhdpi | mipmap-xxxhdpi | 192×192 | Extra extra extra high density |


## How to Create Icons

### Option 1: Online Icon Generators
1. **Android Asset Studio** (Recommended)
   - Visit: https://romannurik.github.io/AndroidAssetStudio/icons-launcher.html
   - Upload your base icon (512×512 recommended)
   - Download all sizes automatically

2. **Icon Kitchen**
   - Visit: https://icon.kitchen/
   - Upload your design
   - Generate all Android sizes

### Option 2: Design Tools
1. **Figma/Adobe XD**
   - Create 512×512 base design
   - Export multiple sizes
   
2. **Canva**
   - Use medical icon templates
   - Resize for each density

### Option 3: AI Generation
Use AI tools like:
- DALL-E
- Midjourney  
- Stable Diffusion

**Prompt suggestion**: 
*"Modern medical app icon, red cross with AI circuit pattern, clean minimalist design, square format, professional healthcare branding"*

## Suggested Icon Concepts

### Concept 1: Medical Cross + AI
- Red medical cross
- Circuit board pattern overlay
- Clean white background

### Concept 2: Stethoscope + Phone
- Stylized stethoscope wrapped around phone
- Blue and white color scheme
- Modern gradient

### Concept 3: Heart + Pulse + Brain
- Heart symbol with EKG line
- Brain icon indicating AI
- Green accent color

### Concept 4: Emergency Symbol
- Emergency medical star (six-pointed)
- Smartphone silhouette
- Red and blue colors

## Implementation Steps

1. **Create your icon** in 512×512 pixels
2. **Generate all sizes** using Android Asset Studio
3. **Save files** in the respective mipmap folders as `ic_launcher.png`
4. **Test the icon** by building and installing the app
5. **Verify** the icon appears correctly on different devices

## Testing Your Icon

After adding the icon:
1. Build your app (`./gradlew assembleDebug`)
2. Install on device/emulator
3. Check home screen appearance
4. Test on different Android versions
5. Verify icon appears in:
   - App drawer
   - Recent apps
   - Settings > Apps
   - Notification area (if applicable)

## Current Status

**Completed:**
- Mipmap directories created

**Pending:**
- Icon files need to be added
- App manifest has icon reference ready

## Next Steps

1. Create or generate your app icon
2. Save the 5 different sizes in their respective folders
3. The app will automatically use the appropriate size for each device

---

**Note**: Once you add the icon files, your Emergency AI Triage App will have a professional appearance on users' devices!