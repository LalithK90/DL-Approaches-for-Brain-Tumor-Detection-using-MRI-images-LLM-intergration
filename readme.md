# Explainable Deep Learning Framework for Automated Brain Tumor Classification

## Abstract

This repository constitutes a comprehensive research implementation investigating **Explainable Artificial Intelligence (XAI) methodologies for automated brain tumor classification from magnetic resonance imaging (MRI) scans**. The work represents a Master of Science in Computer Science (MSc CS - SLQF Level 10) dissertation research conducted at the Postgraduate Institute of Science (PGIS), University of Peradeniya, Sri Lanka.

The research addresses a critical challenge in medical AI: the interpretability and trustworthiness of deep learning models in clinical decision-making contexts. By integrating multiple state-of-the-art convolutional neural network (CNN) architectures with advanced explainability techniques and large language model (LLM)-based report generation, this platform demonstrates a holistic approach to transparent, accountable, and clinically viable AI-assisted diagnostics.

## Research Motivation

### Clinical Context

Brain tumors represent a heterogeneous group of neoplastic conditions with significant morbidity and mortality implications. Early and accurate classification is paramount for:

- **Treatment Stratification**: Determining surgical, radiotherapeutic, or chemotherapeutic interventions
- **Prognostic Assessment**: Estimating patient survival and quality of life outcomes
- **Resource Allocation**: Optimizing healthcare delivery in resource-constrained settings
- **Clinical Decision Support**: Augmenting radiologist expertise with computational assistance

### The Black Box Problem

Despite achieving impressive classification accuracies, deep learning models remain opaque "black boxes," limiting clinical adoption due to:

1. **Lack of Interpretability**: Inability to understand why a model made a specific prediction
2. **Trust Deficit**: Clinicians' reluctance to rely on unexplained automated decisions
3. **Liability Concerns**: Medicolegal implications of algorithmic errors
4. **Bias Detection**: Difficulty identifying dataset or model biases without transparency

### Research Contribution

This work pioneers an **integrated XAI framework** that:

- Implements and compares six distinct CNN architectures under balanced and imbalanced data scenarios
- Generates multi-modal explainability visualizations (Grad-CAM, LIME, Saliency Maps)
- Quantifies prediction uncertainty via epistemic and aleatoric measures
- Assesses explanation coherence through cross-method agreement metrics (Dice coefficient, IoU)
- Synthesizes comprehensive medical reports via multi-LLM orchestration
- Provides interactive knowledge retrieval through RAG-based chatbot interface

---

## System Architecture

The research platform comprises five interconnected components, each addressing distinct aspects of the medical AI workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Research Ecosystem Overview                   │
├─────────────────────────────────────────────────────────────────┤
│  [Dataset] → [Training] → [Backend API] → [Frontend App]        │
│       ↑                        ↓                                 │
│       └──────── [Research Methodology] ──────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

Each component is documented comprehensively in its respective directory:

#### 1. [Brain Tumor MRI Dataset](brain%20tumor%20dataset/README.md)
**Purpose**: Curated neuroimaging data repository  
**Content**: 7,023 MRI images across four pathological classes  
**Source**: Kaggle aggregation (figshare, SARTAJ, Br35H datasets)  
**Classes**: Glioma, Meningioma, Pituitary Adenoma, No Tumor  
**Research Role**: Empirical foundation for model training and evaluation

#### 2. [Model Training Notebook](model_training_notebook/README.md)
**Purpose**: Experimental deep learning pipeline  
**Content**: Jupyter notebook implementing comparative CNN analysis  
**Architectures**: VGG16, VGG19, ResNet50, MobileNetV2, GoogleLeNet, Proposed  
**Methodologies**: Transfer learning, hyperparameter optimization, cross-validation  
**Research Role**: Systematic architecture benchmarking and ablation studies

#### 3. [Brain Tumor Identification API](brain_tumor_identification_api/README.md)
**Purpose**: Backend computational framework and inference engine  
**Content**: Flask-based RESTful API with XAI and LLM integration  
**Capabilities**:
- Multi-model inference (10 trained models: balanced/imbalanced variants)
- Explainability generation (Grad-CAM, LIME, Saliency Maps)
- Uncertainty quantification (MC Dropout, entropy, Brier score)
- Multi-LLM report synthesis (Llama 3.2 Vision, MedGemma, DeepSeek-R1)
- RAG chatbot with ChromaDB vector store  
**Research Role**: Core XAI implementation and knowledge synthesis

#### 4. [Brain Tumor Identification Mobile/Web Application](braintumoridentificationapp/README.md)
**Purpose**: Cross-platform presentation layer  
**Content**: Ionic React application for user interaction  
**Features**: Image upload, model selection, results visualization, report viewing, AI chat  
**Platforms**: Web (PWA), Android (Capacitor), iOS-ready  
**Research Role**: Human-computer interaction and clinical usability demonstration

#### 5. [Data Collection and Research Methodology](data%20collection%20sheet/README.md)
**Purpose**: Experimental protocols and research documentation  
**Content**: Structured data collection instruments, methodology documentation  
**Components**: Experimental design, variable definitions, statistical protocols  
**Research Role**: Methodological rigor, reproducibility, and transparency

---

## Research Methodology

### Experimental Design

**Research Type**: Quantitative comparative experimental study  
**Paradigm**: Empirical evaluation of multiple deep learning architectures  
**Data**: Secondary (publicly available MRI datasets)

### Independent Variables

1. **Model Architecture**: VGG16, VGG19, ResNet50, MobileNetV2, GoogleLeNet, Custom
2. **Data Distribution**: Balanced vs. Imbalanced class representation
3. **Hyperparameters**: Learning rate, batch size, optimization algorithm
4. **Training Strategy**: Transfer learning vs. from-scratch training

### Dependent Variables

1. **Classification Performance**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
2. **Uncertainty Metrics**: Softmax entropy, margin score, Brier score, MC Dropout variance
3. **Explainability Coherence**: Dice coefficient, Intersection over Union (IoU)
4. **Computational Efficiency**: Training time, inference latency, model size

### Evaluation Protocol

- **Data Partitioning**: Stratified 70-15-15 train-validation-test split
- **Cross-Validation**: K-fold (K=5) for robust performance estimation
- **Statistical Testing**: Paired t-tests, confidence intervals (95% CI)
- **Baseline Comparison**: State-of-the-art benchmarks from literature

---

## Key Technologies

### Deep Learning Stack

- **Framework**: TensorFlow 2.10+ / Keras
- **Architectures**: VGG, ResNet, MobileNet, Inception
- **Training**: Adam optimizer, categorical cross-entropy, early stopping
- **Regularization**: Dropout, L2 weight decay, data augmentation

### Explainability Techniques

- **Grad-CAM**: Class-discriminative localization via gradients
- **LIME**: Superpixel-based model-agnostic explanations
- **Saliency Maps**: Gradient-based pixel attribution
- **Agreement Metrics**: Dice coefficient, IoU for cross-method validation

### Large Language Models

- **Vision-Language Model**: Llama 3.2 Vision (multimodal image understanding)
- **Medical Specialist**: MedGemma 4B (domain-specific knowledge)
- **Report Synthesizer**: DeepSeek-R1 14B (coherent narrative generation)
- **Infrastructure**: Ollama for local LLM deployment

### Retrieval-Augmented Generation

- **Vector Database**: ChromaDB for semantic embedding storage
- **Embedding**: Domain-adapted text representations
- **Retrieval**: Cosine similarity-based context extraction
- **Application**: Context-aware chatbot for follow-up queries

### Full-Stack Development

- **Backend**: Python, Flask, Flask-Login, Bcrypt
- **Frontend**: TypeScript, React, Ionic Framework
- **Mobile**: Capacitor for native Android/iOS deployment
- **Database**: SQLite (ChromaDB), JSON (patient data mock)

---

## Project Structure

```
brain_tumor_app/
│
├── readme.md                           # This research overview (main documentation)
│
├── brain tumor dataset/                # MRI neuroimaging data repository
│   ├── README.md                       # Dataset documentation
│   ├── dataset_details.txt             # Provenance and metadata
│   └── [7,023 MRI images]              # Organized by tumor class
│
├── model_training_notebook/            # Experimental deep learning pipeline
│   ├── README.md                       # Training methodology documentation
│   └── enhanced-dl-techniques-for-brain-tumor-identificat.ipynb
│
├── brain_tumor_identification_api/     # Backend computational framework
│   ├── README.md                       # API and XAI documentation
│   ├── app.py                          # Flask application entry point
│   ├── requirements.txt                # Python dependencies
│   ├── models/                         # Trained CNN models (10 .h5 files)
│   ├── src/                            # Source code modules
│   │   ├── auth/                       # Authentication logic
│   │   ├── LLM/                        # Ollama client integration
│   │   ├── models/                     # Model inference
│   │   ├── routes/                     # API endpoints
│   │   └── utils/                      # XAI and metrics utilities
│   ├── templates/                      # HTML templates (Jinja2)
│   ├── static/                         # CSS, JS, uploaded images, visualizations
│   └── chroma_data/                    # ChromaDB vector store
│
├── braintumoridentificationapp/        # Cross-platform mobile/web frontend
│   ├── README.md                       # Frontend application documentation
│   ├── package.json                    # Node.js dependencies
│   ├── src/                            # React/TypeScript source code
│   │   ├── components/                 # Reusable UI components
│   │   ├── pages/                      # Application screens
│   │   └── theme/                      # Ionic styling
│   ├── android/                        # Native Android build artifacts
│   └── public/                         # Static web assets
│
└── data collection sheet/              # Research methodology documentation
    ├── README.md                       # Methodology documentation
    └── Level 10 Research.xlsx          # Experimental protocols and data collection
```

---

## Installation and Setup

### Prerequisites

1. **Python 3.10+** with Conda environment manager
2. **Node.js 18+** and npm for frontend development
3. **Ollama** for local LLM inference ([Download](https://ollama.ai))
4. **Git** for repository cloning
5. **CUDA-capable GPU** (recommended for model training)

### Quick Start: Backend API

```bash
# Navigate to API directory
cd brain_tumor_identification_api

# Create and activate conda environment
conda create --name mri_xai python=3.10 -y
conda activate mri_xai

# Install Python dependencies
pip install -r requirements.txt

# Pull required LLM models
ollama pull llama3.2-vision:latest
ollama pull edwardlo12/medgemma-4b-it-Q4_K_M
ollama pull deepseek-r1:14b

# Ensure Ollama is running, then start Flask server
python app.py

# Access at http://localhost:5000
```

### Quick Start: Frontend Application

```bash
# Navigate to frontend directory
cd braintumoridentificationapp

# Install dependencies
npm install

# Run development server
npm run dev

# Access at http://localhost:5173
```

**Detailed setup instructions** are available in each component's README.

---

## Usage Workflow

### For Researchers and Developers

1. **Data Exploration**: Review [Dataset README](brain%20tumor%20dataset/README.md) for data characteristics
2. **Model Training**: Execute [Training Notebook](model_training_notebook/README.md) to reproduce experiments
3. **Backend Deployment**: Launch [API Backend](brain_tumor_identification_api/README.md) for inference
4. **Frontend Testing**: Run [Mobile/Web App](braintumoridentificationapp/README.md) for end-to-end testing

### For End Users (Educational/Research Context)

1. **Access Application**: Navigate to deployed frontend URL
2. **Authentication**: Log in with research credentials
3. **Image Upload**: Select brain MRI scan (JPEG/PNG)
4. **Model Selection**: Choose CNN architecture and data balance configuration
5. **Analysis Execution**: Trigger classification and XAI generation
6. **Results Review**:
   - Classification prediction with confidence scores
   - Grad-CAM, LIME, and Saliency Map visualizations
   - Uncertainty metrics and explanation agreement scores
   - AI-synthesized comprehensive medical report
7. **Interactive Exploration**: Ask follow-up questions via RAG chatbot

---

## Research Findings and Contributions

### Expected Outcomes

1. **Comparative Performance Analysis**: Identification of optimal CNN architecture for brain tumor classification
2. **Data Balance Impact**: Quantification of performance differences between balanced and imbalanced training
3. **Explainability Coherence**: Statistical validation of agreement between multiple XAI methods
4. **Uncertainty Characterization**: Empirical assessment of confidence calibration and reliability
5. **LLM Report Quality**: Evaluation of multi-LLM pipeline for medical narrative synthesis

### Novel Contributions

- **Integrated XAI Framework**: Holistic platform combining multiple explainability techniques
- **Cross-Method Validation**: Quantitative assessment of explanation consistency via Dice/IoU
- **Multi-LLM Orchestration**: Sequential chaining of specialized models for enhanced report quality
- **RAG-Enhanced Chatbot**: Context-aware conversational interface for medical AI education
- **Full-Stack Implementation**: End-to-end deployable system demonstrating research translation

---

## Limitations and Future Work

### Current Limitations

1. **2D Single-Slice Analysis**: Does not leverage full 3D volumetric MRI information
2. **Dataset Constraints**: Limited to Kaggle dataset; may not generalize to all populations
3. **Non-Clinical Validation**: Lacks prospective clinical trials and expert radiologist benchmarking
4. **Computational Requirements**: GPU-dependent; limited accessibility in resource-poor settings
5. **Regulatory Status**: Not approved as a medical device; restricted to research/education

### Future Research Directions

1. **3D Volumetric CNNs**: Processing entire MRI scans for enhanced spatial context
2. **Federated Learning**: Privacy-preserving multi-institutional model training
3. **Active Learning**: Iterative refinement with expert radiologist feedback
4. **Clinical Deployment**: Prospective studies in research hospital settings
5. **Multimodal Integration**: Combining MRI with CT, PET, and clinical metadata
6. **Longitudinal Analysis**: Tracking tumor progression over time
7. **Rare Variant Detection**: Expanding to uncommon tumor subtypes

---

## Ethical Considerations and Responsible AI

### Research Ethics

- **Institutional Approval**: Conducted under academic research ethics guidelines
- **Data Privacy**: Exclusive use of de-identified, publicly available datasets
- **Transparency**: Open documentation of methodologies and limitations
- **No Clinical Use**: Explicit restriction to academic research and education

### Responsible AI Principles

1. **Explainability**: Commitment to interpretable AI through multiple XAI techniques
2. **Uncertainty Communication**: Honest reporting of model confidence and limitations
3. **Bias Awareness**: Acknowledgment of potential dataset and model biases
4. **Human-in-the-Loop**: Positioning AI as augmentative, not replacement for expert judgment

---

## Scholarly Dissemination

### Publications and Presentations

*This section will be updated upon completion of the research with conference/journal submissions*

### Code and Data Availability

- **Code Repository**: [GitHub Repository Link - To be made public post-defense]
- **Trained Models**: Available upon request for academic research
- **Dataset**: Publicly accessible via [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{brain_tumor_xai_2025,
  author       = {[Author Name]},
  title        = {Explainable Deep Learning Framework for Automated Brain Tumor Classification from MRI Scans},
  school       = {Postgraduate Institute of Science, University of Peradeniya},
  year         = {2025},
  type         = {MSc Computer Science Dissertation (SLQF Level 10)},
  note         = {SC 699 - Level 10 Research}
}
```

---

## Acknowledgments

### Academic Supervision

- **Primary Supervisor**: [Supervisor Name], Department of Statistics & Computer Science, University of Peradeniya
- **Co-Supervisor**: [Co-Supervisor Name], [Affiliation]
- **Research Committee**: [Committee Members]

### Institutional Support

- **Postgraduate Institute of Science (PGIS)**, University of Peradeniya
- **Department of Statistics & Computer Science**, Faculty of Science, University of Peradeniya

### Technical Acknowledgments

- **Dataset Contributors**: Masoud Nickparvar (Kaggle), figshare, SARTAJ, Br35H dataset curators
- **Open-Source Communities**: TensorFlow, Keras, Flask, React, Ionic, Ollama development teams
- **Computing Resources**: [Institution/Lab providing GPU infrastructure]

---

## Disclaimer

> **⚠️ IMPORTANT NOTICE - NOT FOR CLINICAL USE**  
>   
> This platform is developed exclusively for **academic research and educational purposes** in the field of artificial intelligence and medical imaging. It is **NOT** a medical device and has **NOT** been evaluated, cleared, or approved by any regulatory authority (FDA, EMA, CE, etc.).  
>   
> **The system MUST NOT be used for**:
> - Clinical diagnosis of patients
> - Treatment planning or clinical decision-making
> - Patient care or medical advice
> - Any application affecting human health without appropriate clinical validation  
>   
> The predictions, explanations, and reports generated by this system are **research demonstrations only** and may contain errors, biases, or inaccuracies. **Always consult qualified healthcare professionals** for medical diagnosis and treatment.  
>   
> The developers, researchers, and affiliated institutions assume **no liability** for any misuse, misinterpretation, or consequences arising from use of this software.

---

## License

This repository is licensed under the **MIT License**. Only **attribution** is required — please retain the copyright
notice crediting **LalithK90** in copies or substantial portions of the software.

See the full license text in [LICENSE](LICENSE).

---

## Contact Information

**Research Inquiries**: asakahatapitiya@gmail.com  
**Institution**: Postgraduate Institute of Science, University of Peradeniya, Sri Lanka  
 

---

**Last Updated**: December 2025  
**Research Project**: SC 699 - Level 10 Research  
**Degree Program**: Master of Science in Computer Science (SLQF Level 10)  
**Institution**: Postgraduate Institute of Science (PGIS), University of Peradeniya