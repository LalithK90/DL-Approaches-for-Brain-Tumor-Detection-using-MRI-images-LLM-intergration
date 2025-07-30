# Explainable AI for Brain Tumor Detection

This project is a comprehensive web application designed for academic research, demonstrating the use of deep learning models for brain tumor classification from MRI scans. The platform's core focus is on Explainable AI (XAI), providing rich visualizations, quantitative metrics, and AI-powered reporting to make the model's decisions transparent and understandable.

This work was developed as part of the final research for the Master of Science in Computer Science (MSc in CS - SLQF Level 10) degree program at the Postgraduate Institute of Science (PGIS), University of Peradeniya.

---

## ✨ Features

*   **Multi-Model Analysis**: Select from various pre-trained deep learning models (VGG16, VGG19, ResNet50, etc.) for classification.
*   **Brain Tumor Classification**: Classifies MRI scans into four categories: Glioma, Meningioma, Pituitary tumor, or No Tumor.
*   **Rich XAI Visualizations**:
    *   **Grad-CAM**: Highlights class-discriminative regions in the image.
    *   **Saliency Maps**: Shows pixel-level importance for the prediction.
    *   **LIME (Local Interpretable Model-agnostic Explanations)**: Identifies key super-pixels contributing to the decision.
*   **Quantitative Trust & Uncertainty Metrics**:
    *   **Confidence & Margin Scores**: Measure the model's decisiveness.
    *   **Softmax Entropy & Brier Score**: Quantify prediction uncertainty and accuracy.
    *   **MC Dropout**: Estimates model uncertainty through variance and confidence intervals.
    *   **XAI Agreement**: Calculates Dice and IoU scores to measure the consistency between different explanation methods.
*   **AI-Powered Reporting**:
    *   Utilizes a multi-LLM pipeline (Llama 3.2, MedGemma, DeepSeek) to synthesize a detailed, structured medical report from all available data (image analysis, metrics, patient info).
*   **Context-Aware Chatbot**:
    *   An interactive chat interface to ask follow-up questions about the analysis.
    *   The chatbot uses Retrieval-Augmented Generation (RAG) with a ChromaDB vector store to maintain the context of the current report.
*   **Secure User Authentication**: A login system to manage access.

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

## ⚠️ Disclaimer

This tool is for **academic research and educational purposes only**. It is **not intended for medical diagnostic use**. The analysis is based on publicly available data and has not been clinically validated.

## 🎓 Acknowledgements

This project is a final research work for the Master of Science in Computer Science (MSc in CS - SLQF Level 10) degree program conducted by the Postgraduate Institute of Science (PGIS) and the Department of Statistics & Computer Science, University of Peradeniya.
