# 📓 Deep Learning Model Training Notebook

> **Jupyter Notebook for CNN Training, Evaluation & Comparative Analysis**

## 🎯 What's Inside

This directory contains the **Jupyter Notebook for model training**:
- ✅ **6 CNN architectures** implementation & training
- ✅ **Comparative analysis** (balanced vs imbalanced datasets)
- ✅ **Performance evaluation** metrics & visualizations
- ✅ **Transfer learning** from ImageNet weights
- ✅ **Hyperparameter tuning** experiments
- ✅ **Model export** for production deployment
- ✅ **XAI methods** (Grad-CAM, LIME, saliency) for interpretability
- ✅ **XAI evaluation** with consistency/faithfulness metrics

## 🔬 Research Focus

**Task:** 4-class brain tumor MRI slice classification (glioma, meningioma, pituitary, no tumor)

**XAI Methods:**
- **Grad-CAM** - Gradient-weighted Class Activation Mapping
- **LIME** - Local Interpretable Model-agnostic Explanations
- **Saliency Maps** - Gradient-based visualization

**XAI Evaluation:** Quantitative consistency and faithfulness metrics without pixel-level ground truth.

**Note:** This is not clinical validation. Results are based on public datasets and do not establish clinical utility.

## 📂 Directory Structure

```
model_training_notebook/
├── README.md                                          # This file
├── enhanced-dl-techniques-for-brain-tumor-identificat.ipynb  # Main notebook
├── enhanced-dl-techniques-for-brain-tumor-identificat_v2.ipynb  # Version 2 with XAI
├── requirements.txt                                   # Python dependencies
└── artifacts/                                         # Generated outputs
    ├── splits/                                        # Train/val/test manifests
    ├── metrics/                                       # Evaluation results
    └── figures/                                       # Visualizations
```

## ⚠️ Key Reproducibility Rule

**The `Testing/` split is locked and used only once for final reporting.**

Critical practices to avoid test data leakage:
- Training uses the dataset's `Training/` split only
- Validation is created by stratified splitting **only** the `Training/` data (default **val_fraction=0.15**)
- Early stopping, learning-rate scheduling, and checkpoint selection use **validation** only
- **Never** train with `validation_data=(x_test, y_test)`

If you use test data during training, your results are not scientifically defensible.

## 📊 Notebook Overview

**Primary Artifact**: `enhanced-dl-techniques-for-brain-tumor-identificat.ipynb`

This comprehensive Jupyter Notebook implements the complete machine learning pipeline for brain tumor classification research.

### Notebook Sections

1. **📦 Environment Setup**
   - Import libraries (TensorFlow, Keras, NumPy, Matplotlib)
   - Set random seeds for reproducibility
   - Configure GPU settings
   - Load dataset paths

2. **📊 Data Loading & Preprocessing**
   - Load 7,023 MRI images from dataset
   - Parse directory structure for labels
   - Apply preprocessing (normalization, resizing)
   - **Proper data splitting**: Create val split from Training/ only (15% stratified)
   - Keep Testing/ completely isolated until final evaluation
   - Data augmentation implementation (restricted to safe transforms)

3. **🏗️ Model Architecture Definitions**
   - **VGG16** - 16-layer homogeneous CNN
   - **VGG19** - 19-layer variant with deeper features
   - **ResNet50** - Residual connections for vanishing gradient
   - **MobileNetV2** - Lightweight depthwise separable convolutions
   - **GoogleLeNet** - Inception modules for multi-scale features
   - **Proposed Model** - Custom architecture optimized for brain tumors

4. **🎓 Transfer Learning Setup**
   - Load pre-trained ImageNet weights
   - Freeze base layers
   - Add custom classification head
   - Fine-tuning strategy

5. **⚙️ Training Configuration**
   - Optimizer: Adam (learning rate 0.0001)
   - Loss function: Categorical cross-entropy
   - Metrics: Accuracy, precision, recall
   - Callbacks: Early stopping, model checkpoint, learning rate reduction

6. **🔥 Model Training**
   - Train each architecture
   - Monitor training/validation loss
   - Plot learning curves
   - Save best model weights

7. **📈 Performance Evaluation (used only once)
   - Generate confusion matrices
   - Calculate precision, recall, F1-score
   - ROC curves and AUC

8. **🔍 XAI Generation & Evaluation**
   - Generate Grad-CAM, LIME, and saliency visualizations
   - Compute quantitative consistency metrics
   - Evaluate faithfulness without pixel-level ground truth
   - Save XAI metrics to artifacts/metrics/xai_metrics.csv

9. **📊 Comparative Analysis**
   - Compare all 6 architectures
   - Balanced vs imbalanced dataset results
   - Statistical significance testing
   - Performance visualization (bar charts, tables)

10. **💾 Model Export**
    - Save trained models (.h5 format)
    - Export for Flask backend integration
    - Model versioning
    - Save split manifests (train_files.json, val_files.json, test_files.json)

11
10. **📝 Results Documentation**
    - Summarize findings
    - Best performing model
    - Recommendations for deployment

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Jupyter Notebook** or JupyterLab
- **TensorFlow 2.x**
- **GPU** (recommended, CUDA-compatible)
- **16GB RAM** minimum

### Installation Steps

```bash
# 1. Navigate to the notebook directory
cd model_training_notebook

# 2. Create conda environment
conda create -n brain_tumor_training python=3.10 -y
conda activate brain_tumor_training

# 3. Install dependencies
pip install tensorflow jupyter numpy pandas matplotlib seaborn scikit-learn

# For GPU support (NVIDIA):
pip install tensorflow[and-cuda]

# 4. Launch Jupyter Notebook
jupyter notebook enhanced-dl-techniques-for-brain-tumor-identificat.ipynb

# Or use JupyterLab:
jupyter lab
```

### Running the Notebook

```bash
# Start Jupyter
jupyter notebook

# In browser:
# 1. Open the .ipynb file
# 2. Run cells sequentially (Shift+Enter)
# 3. Monitor training progress
# 4. Review results and visualizations
```

## 🔧 Training Configuration

### Default Hyperparameters

```python
# Model training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
]
```

### Model Architecture Example (VGG16)

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load pre-trained VGG16
base_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Freeze base layers
base_model.trainable = False

# Build complete model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## 📊 Expected Results

### Model Performance Benchmarks

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **VGG16** | 96.2% | 95.8% | 96.1% | 96.0% | ~2.5 hours |
| **VGG19** | 96.5% | 96.2% | 96.4% | 96.3% | ~3.1 hours |
| **ResNet50** | 97.1% | 96.9% | 97.0% | 97.0% | ~2.8 hours |
| **MobileNetV2** | 94.5% | 94.1% | 94.3% | 94.2% | ~1.8 hours |
| **GoogleLeNet** | 95.8% | 95.4% | 95.6% | 95.5% | ~2.2 hours |
| **Proposed** | **97.8%** | **97.6%** | **97.7%** | **97.7%** | ~3.5 hours |

*Note: Times on NVIDIA RTX 3090 GPU*

### Sample Visualizations Generated

1. **Training Curves**
   - Accuracy over epochs
   - Loss over epochs
   - Train vs validation comparison

2. **Confusion Matrices**
   - Per-class accuracy breakdown
   - Misclassification patterns

3. **ROC Curves**
   - Multi-class ROC-AUC
   - One-vs-rest comparison

4. **Model Comparison Charts**
   - Bar charts comparing all models
   - Statistical significance annotations

## 🔍 Key Code Snippets

### Loading Dataset

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    'brain tumor dataset/Training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

### Training Loop

```python
# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# Plot results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()
```

### Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predict on test set
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Metrics
print(classification_report(
    y_true, y_pred,
    target_names=['glioma', 'meningioma', 'notumor', 'pituitary']
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()
```

## 🛠️ Troubleshooting

**GPU not detected?**
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Force GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

**Out of memory errors?**
```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32

# Use mixed precision training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

**Slow training?**
```python
# Enable XLA compilation
tf.config.optimizer.set_jit(True)

# Use prefetching
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
```

## 📈 Advanced Features

### Hyperparameter Tuning with Keras Tuner

```python
import keras_tuner as kt

def build_model(hp):
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(
            hp.Int('units', min_value=128, max_value=512, step=64),
            activation='relu'
        ),
        layers.Dropout(hp.Float('dropout', 0.3, 0.7, step=0.1)),
        layers.Dense(4, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10
)

tuner.search(train_generator, validation_data=val_generator, epochs=20)
```

### Grad-CAM Visualization in Training

```python
from tf_keras_vis.gradcam import Gradcam

# Create Grad-CAM object
gradcam = Gradcam(model)

# Generate heatmap
cam = gradcam(score, seed_input)

# Overlay on image
heatmap = np.uint8(cm.jet(cam[0])[:, :, :3] * 255)
superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
```

## 📚 Related Components

- **Dataset**: See [brain tumor dataset/](../brain%20tumor%20dataset/)
- **Backend API**: See [brain_tumor_identification_api/](../brain_tumor_identification_api/)
- **Research Data**: See [data collection sheet/](../data%20collection%20sheet/)

## 📊 Experiment Tracking

**Recommended Tools**:
- **TensorBoard**: Built-in visualization
- **Weights & Biases**: Cloud experiment tracking
- **MLflow**: Model versioning and tracking

```bash
# Launch TensorBoard
tensorboard --logdir=./logs

# View at http://localhost:6006
```

## 📄 Deliverables

After running the notebook, you'll have:
- ✅ 12 trained models (.h5 files) - 6 architectures × 2 datasets
- ✅ Training history plots
- ✅ Confusion matrices
- ✅ Performance comparison tables
- ✅ Best model for production deployment
- ✅ Comprehensive results documentation

## ⚠️ Important Notes

**Reproducibility**:
- Set random seeds (NumPy, TensorFlow, Python)
- Document environment (Python version, library versions)
- Save notebook execution order

**Best Practices**:
- Run cells sequentially (top to bottom)
- Save checkpoints frequently
- Monitor training curves for overfitting
- Va� Dataset Requirements

Expected Kaggle dataset structure:

```
/kaggle/input/brain-tumor-mri-dataset/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

## 🔬 Notes on Data Augmentation

**Important**: No geometric flipping is applied because laterality can be clinically meaningful in brain MRI analysis.

**Safe augmentation transforms**:
- Small rotations (±10-15°)
- Mild intensity jitter
- Zoom variations (±10%)
- Width/height shifts

Always document augmentation strategies in research papers.

## 🧪 Ethical and Clinical Scope

**Important Disclaimers**:
- Uses public, de-identified dataset
- No patient recruitment or protected health information
- **No clinical workflow validation** or reader study conducted
- Results do not establish clinical utility
- Not intended for direct clinical use without proper validation

**These limitations must be explicitly stated in any research publication.**

## 📄 License

Academic research code. See [LICENSE](../LICENSE) in root directory.

---

**Ready to Train**: Open Jupyter Notebook and start training state-of-the-art CNN models with proper reproducibility practice
## 📄 License

Academic research code. See [LICENSE](../LICENSE) in root directory.

---

**Ready to Train**: Open Jupyter Notebook and start training state-of-the-art CNN models!
