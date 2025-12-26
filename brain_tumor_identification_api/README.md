# Brain Tumor Identification API - Backend Computational Framework

## Research Context

This directory constitutes the **backend computational infrastructure** for an integrated Explainable Artificial Intelligence (XAI) system designed to facilitate automated brain tumor classification from magnetic resonance imaging (MRI) scans. The framework represents the core research implementation component, integrating state-of-the-art deep learning architectures with interpretability mechanisms to enhance clinical transparency and trustworthiness in AI-assisted medical diagnostics.

This work was developed as the primary technical contribution for the Master of Science in Computer Science (MSc in CS - SLQF Level 10) dissertation research at the Postgraduate Institute of Science (PGIS), University of Peradeniya, focusing on **Explainable Deep Learning Approaches for Medical Image Analysis**.

## Academic Significance

The platform addresses critical research questions in medical AI:

1. **Model Interpretability**: How can deep neural networks' decision-making processes be rendered comprehensible to clinical practitioners?
2. **Trustworthiness Quantification**: What metrics effectively measure the reliability and certainty of automated diagnostic predictions?
3. **Multi-Method Explanation Coherence**: Do different explainability techniques converge on consistent interpretations?
4. **Knowledge Synthesis**: Can large language models effectively integrate multimodal medical data into clinically meaningful narratives?

---

## ✨ Core Research Contributions

### 1. Multi-Architecture Comparative Framework

**Model Ensemble**: Implementation of six distinct convolutional neural network architectures with dual training strategies (balanced/imbalanced datasets):
- **VGG16**: 16-layer Visual Geometry Group architecture with homogeneous 3×3 convolutions
- **VGG19**: Extended 19-layer variant with deeper feature hierarchies
- **ResNet50**: 50-layer Residual Network employing skip connections and identity mappings
- **MobileNetV2**: Lightweight architecture with depthwise separable convolutions for computational efficiency
- **GoogleLeNet (Inception)**: Multi-scale feature extraction via inception modules
- **Proposed Architecture**: Novel custom CNN design optimized for neuroimaging classification

**Research Objective**: Empirical determination of optimal architectural paradigms for brain tumor classification across varying data distribution scenarios.

### 2. Multi-Modal Explainable AI Framework

**Integrated XAI Techniques**:

#### Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Mechanism**: Utilizes gradient information flowing into final convolutional layer
- **Output**: Class-discriminative localization maps highlighting salient regions
- **Research Value**: Identifies anatomical structures influencing classification decisions
- **Implementation**: `tf-keras-vis` library for TensorFlow/Keras integration

#### LIME (Local Interpretable Model-agnostic Explanations)
- **Mechanism**: Perturbs input via superpixel masking, observing prediction changes
- **Output**: Feature importance weights for interpretable image segments
- **Research Value**: Model-agnostic local fidelity assessment
- **Implementation**: `lime` library with custom image segmentation

#### Saliency Maps
- **Mechanism**: Computes gradient of output class with respect to input pixels
- **Output**: Pixel-level attribution map
- **Research Value**: Fine-grained visualization of decision-making sensitivity
- **Implementation**: `tf-explain` library for gradient-based attribution

**Novel Contribution**: Cross-method validation via quantitative agreement metrics (Dice coefficient, IoU) to assess explanation consistency and trustworthiness.

### 3. Comprehensive Uncertainty Quantification

**Epistemic Uncertainty Estimation**:
- **Monte Carlo Dropout (MC Dropout)**: Stochastic forward passes with activated dropout at inference
- **Metrics**: Prediction variance, confidence intervals, coefficient of variation
- **Research Significance**: Distinguishes model uncertainty from data uncertainty

**Aleatoric Uncertainty Measures**:
- **Softmax Entropy**: Quantifies prediction distribution spread
- **Margin Score**: Distance between top-2 class probabilities
- **Brier Score**: Calibration metric measuring probabilistic prediction accuracy

**Clinical Impact**: Enables risk stratification and flagging of uncertain cases requiring expert review.

### 4. AI-Synthesized Medical Reporting Pipeline

**Multi-LLM Orchestration Framework**:

1. **Vision-Language Model (VLM) - Llama 3.2 Vision**:
   - **Input**: Raw MRI image
   - **Output**: Anatomical description, preliminary observations
   - **Capability**: Multimodal understanding of medical imagery

2. **Medical Domain Specialist - MedGemma 4B**:
   - **Input**: VLM description + classification results + patient demographics
   - **Output**: Clinically grounded interpretation with medical terminology
   - **Capability**: Domain-specific knowledge integration

3. **Report Synthesizer - DeepSeek-R1 14B**:
   - **Input**: All previous outputs + XAI visualizations + quantitative metrics
   - **Output**: Comprehensive, structured medical report in standardized format
   - **Capability**: Coherent narrative generation, uncertainty communication

**Innovation**: Sequential LLM chaining leveraging specialized capabilities for enhanced report quality and clinical relevance.

### 5. Retrieval-Augmented Generation (RAG) Chatbot

**Architecture**:
- **Vector Database**: ChromaDB for semantic embedding storage
- **Embedding Model**: Domain-adapted text embeddings
- **Retrieval Mechanism**: Cosine similarity-based context extraction
- **Generation Model**: Ollama-served LLM with augmented prompts

**Functionality**:
- Context-aware follow-up question answering
- Medical terminology clarification
- Methodology explanation
- Educational support for non-expert users

**Research Contribution**: Demonstrates practical implementation of RAG for clinical decision support, enhancing user interaction and understanding.

### 6. Secure Multi-User System

**Authentication & Authorization**:
- **Framework**: Flask-Login with session management
- **Credential Security**: Bcrypt password hashing
- **Session Persistence**: Secure cookie-based authentication
- **Access Control**: Role-based permissions (extensible)

## 🛠️ Tech Stack

*   **Backend**: Python, Flask, Flask-Login
*   **Machine Learning**: TensorFlow, Keras
*   **XAI Libraries**: `tf-keras-vis`, `lime`, `tf-explain`
*   **LLM/VLM Orchestration**: Ollama
*   **Vector Database (for RAG)**: ChromaDB
*   **Frontend**: HTML, CSS, JavaScript, Bootstrap, jQuery, SweetAlert2

## 🚀 Setup and Installation

Follow these steps to get the application running locally.

### 1. Prerequisites

*   **Python 3.10+**
*   **Conda**: For managing the Python environment.
*   **Git**: For cloning the repository.
*   **Ollama**: The application relies on a locally running Ollama instance to serve the LLMs. Install Ollama from the official website.

### 2. Clone the Repository

```bash
git clone https://github.com/LalithK90/Deep-Learning-Approaches-for-Brain-Tumor-Detection-using-MRI-WebApp.git
cd Deep-Learning-Approaches-for-Brain-Tumor-Detection-using-MRI-WebApp
```

### 3. Set Up the Conda Environment

Create and activate a new conda environment for the project.

```bash
conda create --name mri_xai python=3.10 -y
conda activate mri_xai
```

### 4. Install Python Dependencies

Install all the required Python packages using pip.

```bash
pip install Flask Flask-Login Flask-Bcrypt Werkzeug numpy opencv-python scikit-image matplotlib scipy tensorflow tf-keras-vis lime tf-explain ollama chromadb
```

### 5. Set Up Ollama Models

After installing Ollama, pull the required models from the command line. These models are used for image analysis, report generation, and chat.

```bash
ollama pull llama3.2-vision:latest
ollama pull edwardlo12/medgemma-4b-it-Q4_K_M
ollama pull deepseek-r1:14b
```

Ensure the Ollama application is running in the background before starting the Flask app.

### 6. Place ML Models

This project uses several Keras (`.h5`) models for tumor classification.

1.  Create a directory named `models` in the project root.
2.  Download or create your trained model files (e.g., `propose_balance.h5`, `vgg19_balance.h5`, etc.).
3.  Place all `.h5` files inside the `models/` directory.

### 7. Create `run.py`

The project is missing a main entry point to start the server. Create a file named `run.py` in the project's root directory and add the following code:

```python
# run.py
from flask import Flask
from flask_bcrypt import Bcrypt
import os

# Import blueprints
from src.routes.routes import main_bp
from src.auth.auth import auth_bp, init_auth

def create_app():
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')

    # Configuration
    app.config['SECRET_KEY'] = 'a_very_secret_key_that_you_should_change'
    app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
    app.config['VISUALIZATION_FOLDER'] = os.path.join(app.static_folder, 'visualizations')
    # Note: Ensure you have a 'patient_data.json' file in 'src/data/'
    app.config['PATIENT_DATA_PATH'] = os.path.join(os.path.dirname(__file__), 'src', 'data', 'patient_data.json')
    app.config['MODELS_DIR'] = 'models'

    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['VISUALIZATION_FOLDER'], exist_ok=True)
    os.makedirs(os.path.dirname(app.config['PATIENT_DATA_PATH']), exist_ok=True)

    # Initialize extensions
    bcrypt = Bcrypt(app)
    init_auth(app, bcrypt) # Initialize auth routes and user store

    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
```

### 8. Run the Application

With the conda environment active and Ollama running, start the Flask server.

```bash
python run.py
```

Navigate to `http://127.0.0.1:5000` in your web browser.

## 🕹️ How to Use

1.  **Login**: Use one of the default credentials to log in (e.g., `Username: doctor`, `Password: doctor`).
2.  **Select Model**: Choose a classification model from the dropdown menu.
3.  **Upload Image**: Upload an MRI scan of a brain.
4.  **Analyze**: Click "Upload & Analyze" to start the process.
5.  **Review Results**:
    *   View the original image and the generated XAI visualizations.
    *   Examine the quantitative metrics in the "Prediction & Explanation Metrics" table.
    *   Read the AI-synthesized report in the chat window.
6.  **Chat with AI**: Use the chat box to ask follow-up questions about the report and analysis.

---

## System Architecture and Component Integration

### Directory Structure

```
brain_tumor_identification_api/
├── app.py                          # Flask application factory and configuration
├── requirements.txt                # Python dependencies specification
├── api_key.env                     # Environment variables (API keys, secrets)
├── patient data json.json          # Sample patient demographic data
├── models/                         # Trained deep learning models (.h5 files)
│   ├── vgg16_balance.h5
│   ├── vgg19_imbalanced.h5
│   └── ... (10 total models)
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── auth/                       # Authentication and authorization logic
│   │   ├── auth.py
│   │   └── __init__.py
│   ├── LLM/                        # Large Language Model integration
│   │   ├── ollama_client.py
│   │   └── __init__.py
│   ├── models/                     # Model loading and inference logic
│   │   ├── models.py
│   │   └── __init__.py
│   ├── routes/                     # API endpoint definitions
│   │   ├── routes.py
│   │   └── __init__.py
│   └── utils/                      # Utility functions (XAI, metrics, preprocessing)
│       ├── utils.py
│       └── __init__.py
├── templates/                      # HTML templates (Jinja2)
│   ├── index.html                  # Main application interface
│   └── login.html                  # Authentication page
├── static/                         # Static assets
│   ├── css/                        # Stylesheets
│   ├── js/                         # Client-side JavaScript
│   ├── img/                        # Images and icons
│   ├── uploads/                    # User-uploaded MRI images
│   └── visualizations/             # Generated XAI heatmaps
│       ├── glioma/
│       ├── meningioma/
│       ├── no tumor/
│       └── pituitary/
└── chroma_data/                    # ChromaDB vector store persistence
    ├── chroma.sqlite3
    └── [vector embeddings]
```

### Data Flow Architecture

```
User Request (MRI Upload)
    ↓
Flask API Endpoint (/api/analyze)
    ↓
Image Preprocessing (Resize, Normalize)
    ↓
Model Inference (Selected Architecture)
    ↓
[Parallel Execution]
    ├── Grad-CAM Generation
    ├── LIME Explanation
    ├── Saliency Map Computation
    ├── MC Dropout Uncertainty
    └── Quantitative Metrics Calculation
    ↓
XAI Agreement Analysis (Dice, IoU)
    ↓
Multi-LLM Report Generation Pipeline
    ├── Llama 3.2 Vision (Image Description)
    ├── MedGemma (Medical Interpretation)
    └── DeepSeek-R1 (Report Synthesis)
    ↓
ChromaDB Indexing (Report Embedding)
    ↓
Response to Client (JSON + Visualizations)
```

### Technology Stack Details

**Backend Framework**:
- **Flask 2.3+**: Lightweight WSGI web application framework
- **Flask-Login**: User session management
- **Flask-Bcrypt**: Password hashing and verification
- **Werkzeug**: WSGI utility library for routing and request handling

**Deep Learning & XAI**:
- **TensorFlow 2.10+**: Deep learning framework
- **Keras**: High-level neural network API
- **tf-keras-vis**: Visualization toolkit for Keras models
- **LIME 0.2+**: Model-agnostic interpretability
- **tf-explain**: TensorFlow-specific explainability library

**LLM Integration**:
- **Ollama**: Local LLM deployment and orchestration
- **ChromaDB**: Vector database for RAG implementation
- **NumPy, SciPy**: Numerical computing

**Image Processing**:
- **OpenCV (cv2)**: Computer vision operations
- **scikit-image**: Advanced image manipulation
- **Matplotlib**: Visualization generation
- **Pillow (PIL)**: Image I/O operations

---

## Related Research Components

This backend API integrates with the broader research ecosystem:

- **Dataset**: Utilizes data from [Brain Tumor Dataset](../brain%20tumor%20dataset/README.md)
- **Model Training**: Deploys models trained in [Model Training Notebook](../model_training_notebook/README.md)
- **Frontend Interface**: Serves [Mobile/Web Application](../braintumoridentificationapp/README.md)
- **Research Methodology**: Implements protocols documented in [Data Collection Sheet](../data%20collection%20sheet/README.md)

---

## ⚠️ Research Ethics and Limitations

### Ethical Considerations

**Scope of Application**:
- **For**: Academic research, algorithm development, educational demonstrations
- **Not For**: Clinical diagnosis, treatment decisions, patient care

**Data Privacy**:
- All datasets are publicly available and de-identified
- No Protected Health Information (PHI) is processed or stored
- HIPAA compliance not applicable (non-clinical research)

**Transparency Commitment**:
- Open methodology documentation
- Reproducible experimental protocols
- Acknowledgment of limitations and biases

### Technical Limitations

1. **Model Generalization**: Performance dependent on training data distribution
2. **Dataset Bias**: Kaggle dataset may not represent global population diversity
3. **Class Imbalance**: Natural distribution skews toward certain tumor types
4. **Resolution Constraints**: Fixed input size may lose fine-grained details
5. **Computational Resources**: Inference time varies with hardware capabilities

### Clinical Limitations

1. **Regulatory Status**: Not FDA-approved or CE-marked medical device
2. **Validation**: Lacks clinical trials or radiologist benchmarking
3. **Multimodal Integration**: Single MRI slice vs. volumetric 3D analysis
4. **Temporal Dynamics**: No longitudinal progression tracking
5. **Rare Variants**: Limited representation of uncommon tumor subtypes

---

## Future Research Directions

### Technical Enhancements

1. **3D Volumetric Analysis**: Full MRI scan processing with 3D CNNs
2. **Ensemble Methods**: Model combination for improved robustness
3. **Active Learning**: Iterative refinement with expert feedback
4. **Federated Learning**: Privacy-preserving multi-institutional training
5. **Neural Architecture Search**: Automated optimal architecture discovery

### Clinical Validation

1. **Prospective Studies**: Real-world deployment in research settings
2. **Expert Benchmarking**: Comparison with board-certified radiologists
3. **Multi-Reader Studies**: Inter-rater reliability assessment
4. **Outcome Correlation**: Prediction validation against histopathology

### Explainability Advances

1. **Counterfactual Explanations**: "What-if" scenario generation
2. **Concept Activation Vectors**: High-level feature interpretation
3. **Attention Mechanisms**: Transformer-based spatial attention
4. **Human-AI Collaboration**: Interactive refinement of explanations

---

## 🎓 Academic Acknowledgments

This research is conducted as part of the **SC 699 - Level 10 Research** module in the Master of Science in Computer Science (SLQF Level 10) program at:

**Postgraduate Institute of Science (PGIS)**  
University of Peradeniya, Sri Lanka

**Department of Statistics & Computer Science**  
Faculty of Science, University of Peradeniya

**Supervision**: Under the guidance of academic supervisors and research committee members.

---

## References and Further Reading

1. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "'Why Should I Trust You?': Explaining the Predictions of Any Classifier." KDD.
3. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML.
4. Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps." arXiv.
5. Topol, E. J. (2019). "High-performance medicine: the convergence of human and artificial intelligence." Nature Medicine.

---

**Repository Type**: Research Implementation Artifact  
**License**: Academic Use (See LICENSE file)  
**Last Updated**: December 2025  
**MSc CS Research Project** | Postgraduate Institute of Science, University of Peradeniya
