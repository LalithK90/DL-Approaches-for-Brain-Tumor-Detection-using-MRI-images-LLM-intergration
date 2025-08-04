# Brain Tumor Identification and Explainable AI Platform

This repository contains a comprehensive web-based platform for brain tumor identification from MRI scans, with a strong focus on Explainable AI (XAI) and clinical transparency. The project integrates deep learning models for tumor classification, advanced XAI visualizations, quantitative metrics, and AI-powered reporting, making it a valuable tool for academic research and education in medical imaging.

The solution is composed of two main parts:
- **Backend API** (`brain_tumor_identification_api/`): A Python Flask application that handles image analysis, model inference, XAI visualizations (Grad-CAM, LIME, Saliency Maps), quantitative metrics (confidence, Dice, IoU, Brier score, etc.), and generates detailed medical reports using large language models (LLMs) via Ollama and ChromaDB for RAG-based chat.
- **Frontend App** (`braintumoridentificationapp/`): A modern, cross-platform Ionic React application that provides an intuitive interface for uploading MRI images, selecting models, viewing predictions and explanations, and interacting with an AI assistant.

**Key Features:**
- Upload and analyze brain MRI images using multiple deep learning models.
- Visualize model explanations with Grad-CAM, LIME, and Saliency Maps.
- Review quantitative metrics for prediction confidence and explanation agreement.
- Generate comprehensive, AI-synthesized medical reports combining image analysis, patient data, and XAI insights.
- Chat with an AI assistant for follow-up questions using Retrieval-Augmented Generation (RAG).
- Secure user authentication and session management.
- Designed for research and educational use, not for clinical diagnosis.

**Project Structure:**
- `brain_tumor_identification_api/` – Flask backend, ML/XAI logic, API, templates, static assets.
- `braintumoridentificationapp/` – Ionic React frontend app.
- `brain tumor dataset/`, `model_training_notebook/`, `data collection sheet/` – Supporting data and research materials.

> **Disclaimer:**  
> This tool is for academic research and educational purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.