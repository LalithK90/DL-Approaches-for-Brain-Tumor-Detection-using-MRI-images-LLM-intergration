# 📱 Brain Tumor Identification Mobile App

> **Cross-Platform Ionic React Application for AI-Powered Brain Tumor Diagnosis**

## 🎯 What's Inside

This is the **front-end mobile/web application** that provides an intuitive interface for:
- ✅ **Upload MRI scans** (camera or file picker)
- ✅ **Select AI models** (6 CNN architectures)
- ✅ **View diagnoses** with confidence scores
- ✅ **Visualize XAI** (Grad-CAM, LIME, Saliency heatmaps)
- ✅ **Read 12-section reports** with clinical recommendations
- ✅ **Chat with RAG bot** for medical Q&A

## 📂 Project Structure

```
braintumoridentificationapp/
├── package.json              # Node dependencies
├── capacitor.config.ts       # Native app configuration
├── vite.config.ts           # Build configuration
├── ionic.config.json        # Ionic framework config
├── index.html               # Entry point HTML
├── src/                     # Source code
│   ├── main.tsx            # React app bootstrap
│   ├── App.tsx             # Main app component
│   ├── pages/              # Screen components
│   │   ├── Home.tsx
│   │   ├── Login.tsx
│   │   ├── Analysis.tsx
│   │   ├── Chat.tsx
│   │   └── History.tsx
│   ├── components/         # Reusable UI components
│   │   ├── ImageUploader.tsx
│   │   ├── ModelSelector.tsx
│   │   ├── ResultCard.tsx
│   │   └── XAIVisualizer.tsx
│   └── theme/              # Ionic styling
├── android/                # Android native project
├── cypress/                # E2E testing
└── public/                 # Static assets
    ├── favicon.ico
    ├── favicon.png
    └── manifest.json       # PWA manifest
```

## 🚀 Quick Start

### Prerequisites

- **Node.js 18+** and npm
- **Backend API running** (see `brain_tumor_identification_api/`)
- **Android Studio** (for Android builds)
- **Xcode** (for iOS builds - macOS only)

### Installation Steps

```bash
# 1. Navigate to the mobile app directory
cd braintumoridentificationapp

# 2. Install dependencies
npm install

# 3. Configure backend URL
# Edit API_BASE_URL in your code to point to backend:
# const API_BASE_URL = "http://localhost:5001"

# 4. Run in browser (development)
npm run dev
# Opens at http://localhost:5173

# 5. Build for production
npm run build

# 6. Preview production build
npm run preview
```

### Mobile Deployment

#### Android

```bash
# 1. Build the web app
npm run build

# 2. Sync with Capacitor
npx cap add android
npx cap sync android

# 3. Open in Android Studio
npx cap open android

# 4. Build APK in Android Studio
# Build > Build Bundle(s) / APK(s) > Build APK
```

#### iOS (macOS only)

```bash
# 1. Build the web app
npm run build

# 2. Sync with Capacitor
npx cap add ios
npx cap sync ios

# 3. Open in Xcode
npx cap open ios

# 4. Build in Xcode
# Product > Build
```

### PWA Installation

The app can be installed as a Progressive Web App:
1. Open in browser (Chrome, Edge, Safari)
2. Click "Install App" in address bar
3. Access offline capabilities

## 🔧 Key Features

### 1. **MRI Image Upload**

Multiple upload methods:
- 📷 **Camera**: Take photo directly (mobile only)
- 📁 **File Picker**: Select from gallery
- 🖱️ **Drag & Drop**: Desktop convenience

Supported formats: **JPEG, PNG, JPG**

### 2. **AI Model Selection**

Choose from 6 CNN architectures:
- **VGG16** - Deep homogeneous architecture
- **VGG19** - Extended VGG variant
- **ResNet50** - Residual connections
- **MobileNetV2** - Lightweight mobile-optimized
- **GoogleLeNet** - Inception modules
- **Proposed** - Custom research model

**Each model available in**:
- ✅ Balanced dataset version
- ✅ Imbalanced dataset version

### 3. **Diagnosis Results**

Real-time display of:
- **Predicted Class**: Glioma / Meningioma / Pituitary / No Tumor
- **Confidence Score**: Percentage with color-coded badge
- **Processing Time**: Inference duration
- **Uncertainty Metrics**: MC Dropout variance, entropy

### 4. **XAI Visualizations**

Three explainability techniques displayed side-by-side:

**Grad-CAM** (Gradient-weighted Class Activation Maps)
- Heatmap overlay showing important regions
- Class-specific activation

**LIME** (Local Interpretable Model-agnostic Explanations)
- Segmented feature importance
- Superpixel-based explanation

**Saliency Maps**
- Pixel-level gradient attribution
- Fine-grained sensitivity analysis

**XAI Metrics**:
- Comprehensiveness Score
- Sufficiency Score
- Dice Coefficient (inter-method agreement)

### 5. **12-Section Clinical Report**

Structured medical report with:

| Section | Content |
|---------|---------|
| **0** | Quick Clinical Decision (30-sec summary) |
| **1** | Executive Summary |
| **2** | Clinical Presentation |
| **3** | Imaging Findings |
| **4** | AI Classification Results |
| **5** | XAI Interpretation |
| **6** | Diagnostic Synthesis |
| **7** | Differential Diagnoses |
| **8** | Management Recommendations |
| **9** | Prognosis & Follow-up |
| **10** | Educational Insights (🎓 Teaching points) |
| **11** | References & Guidelines |
| **11.5** | ⭐ **External Learning Resources** (NEW) |
| **12** | Technical AI Appendix |

**NEW Features**:
- 🌐 **Direct links** to Neurosurgical Atlas, Radiopaedia, NCCN
- 🔍 **Specific search terms** for deeper study
- 📚 **Related conditions** beyond brain tumors

### 6. **RAG-Enhanced Medical Chatbot**

Ask questions and get evidence-based answers:
- Query medical knowledge base (ChromaDB)
- Retrieve similar clinical cases
- Search clinical guidelines
- Patient-specific Q&A

**Sample Questions**:
- "What are the symptoms of glioma?"
- "Explain the difference between meningioma and glioma"
- "What imaging findings suggest pituitary adenoma?"

### 7. **Analysis History**

Track previous scans:
- View past diagnoses
- Compare results across models
- Export reports as PDF
- Share with healthcare providers

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Ionic Framework | 8.5.0 | UI components |
| React | 19.0.0 | Frontend framework |
| TypeScript | Latest | Type safety |
| Capacitor | 7.4.0 | Native bridge |
| Vite | Latest | Build tool |
| Axios | Latest | HTTP client |

## 📦 Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm test             # Run unit tests
npm run lint         # Lint TypeScript/React code
npx cap sync         # Sync web code with native platforms
```

## 🎨 Customization

### Theme Configuration

Edit `src/theme/variables.css`:

```css
:root {
  --ion-color-primary: #1f2937;      /* App primary color */
  --ion-color-secondary: #3b82f6;    /* Secondary accent */
  --ion-color-success: #10b981;      /* Success states */
  --ion-color-danger: #ef4444;       /* Error states */
}
```

### App Branding

Update `public/manifest.json`:

```json
{
  "short_name": "Brain Tumor ID",
  "name": "Brain Tumor Identification with XAI",
  "theme_color": "#1f2937"
}
```

Update `capacitor.config.ts`:

```typescript
const config: CapacitorConfig = {
  appId: 'com.braintumor.identification',
  appName: 'Brain Tumor ID'
};
```

## 🔐 Security

- ✅ HTTPS-only API communication
- ✅ Secure token storage
- ✅ Input sanitization
- ✅ CORS protection
- ✅ Client-side validation

## 🐛 Troubleshooting

**Backend connection failed?**
```typescript
// For mobile testing, use your computer's IP:
const API_BASE_URL = "http://192.168.1.100:5001";
```

**Android build errors?**
```bash
cd android
./gradlew clean
./gradlew build
```

**npm install fails?**
```bash
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

## 📱 Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| **Web** | ✅ Full support | Chrome, Firefox, Safari, Edge |
| **Android** | ✅ Full support | Android 5.0+ (API 21+) |
| **iOS** | ✅ Configurable | iOS 13.0+ |
| **PWA** | ✅ Installable | Offline capabilities |

## 📊 Performance

- ⚡ **Lazy loading** for components
- 🗜️ **Code splitting** via Vite
- 📦 **Optimized bundles** (<500KB gzipped)
- 🚀 **Fast page transitions**
- 💾 **Caching** for API responses

## 🧪 Testing

```bash
# Unit tests
npm test

# E2E tests (Cypress)
npx cypress open

# Coverage report
npm run test:coverage
```

## 🤝 Development Workflow

1. **Run backend**: Start Flask API (`python app.py` in backend folder)
2. **Run frontend**: Start dev server (`npm run dev`)
3. **Make changes**: Edit components in `src/`
4. **Test**: Check browser at `http://localhost:5173`
5. **Build**: `npm run build` for production
6. **Deploy**: Sync with Capacitor for mobile (`npx cap sync`)

## ⚠️ Clinical Disclaimer

> This application is designed exclusively for **academic research and educational purposes**. It is **NOT** a medical device and has not been evaluated by regulatory authorities (FDA, CE, etc.). **MUST NOT** be used for clinical diagnosis or patient care decisions.

## 📄 License

See [LICENSE](../LICENSE) in the root directory.

---

**Quick Launch**: `npm install && npm run dev` (make sure backend is running!)
