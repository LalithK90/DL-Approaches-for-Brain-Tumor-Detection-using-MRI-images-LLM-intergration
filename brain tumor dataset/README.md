# 🧠 Brain Tumor MRI Dataset

> **Curated Multi-Class Neuroimaging Dataset for Deep Learning Research**

## 🎯 What's Inside

This directory contains the **MRI dataset** used for training and evaluating CNN models:
- ✅ **7,023 MRI images** (T1/T2/FLAIR sequences)
- ✅ **4 tumor classes** (Glioma, Meningioma, Pituitary, No Tumor)
- ✅ **Pre-processed** and ready for model training
- ✅ **Balanced & imbalanced** versions for comparative research

## 📊 Dataset Statistics

| Class | Images | Percentage |
|-------|--------|------------|
| **Glioma** | ~1,621 | 23% |
| **Meningioma** | ~1,645 | 23% |
| **Pituitary** | ~1,757 | 25% |
| **No Tumor** | ~2,000 | 29% |
| **Total** | **7,023** | 100% |

## 📂 Directory Structure

```
brain tumor dataset/
├── README.md                    # This file
├── dataset_details.txt         # Detailed class distribution
├── Training/                   # Training split (70%)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Validation/                # Validation split (15%)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/                   # Test split (15%)
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 🔬 Tumor Classes Explained

### 1. Glioma
**Type**: Malignant primary brain tumor  
**Origin**: Glial cells (supportive brain tissue)  
**Characteristics**:
- Most common malignant brain tumor
- Infiltrative growth pattern
- Grades I-IV (WHO classification)
- Poor prognosis for high-grade tumors

**MRI Features**:
- Irregular borders
- Heterogeneous enhancement
- Mass effect with edema
- Commonly in cerebral hemispheres

### 2. Meningioma
**Type**: Typically benign tumor  
**Origin**: Meningeal tissue (brain/spinal cord coverings)  
**Characteristics**:
- Second most common brain tumor
- Slow-growing
- Usually well-circumscribed
- Good prognosis after surgical resection

**MRI Features**:
- Well-defined margins
- Homogeneous enhancement
- Dural tail sign
- Extra-axial location

### 3. Pituitary Adenoma
**Type**: Benign tumor  
**Origin**: Pituitary gland cells  
**Characteristics**:
- Can cause hormonal imbalances
- Visual field defects (if large)
- Classified as micro (<1cm) or macro (>1cm)
- Treated with surgery or medication

**MRI Features**:
- Sellar/suprasellar location
- Variable signal intensity
- May show hemorrhage or cyst formation
- Optic chiasm compression (if large)

### 4. No Tumor
**Type**: Normal brain MRI  
**Characteristics**:
- Control cases for classification
- No space-occupying lesions
- Normal brain anatomy
- Essential for reducing false positives

## 📥 Data Source

**Primary Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**Aggregated from**:
- **figshare** - Research data repository
- **SARTAJ dataset** - Specialized brain tumor collection
- **Br35H dataset** - Binary tumor classification dataset

## 🛠️ Dataset Preparation

### Image Specifications

- **Format**: JPEG/PNG
- **Color**: Grayscale (converted from RGB if needed)
- **Resolution**: Standardized to 224×224 or 256×256 pixels
- **Bit Depth**: 8-bit intensity values
- **Normalization**: Pixel values scaled to [0, 1]

### Data Split Strategy

| Split | Percentage | Purpose |
|-------|------------|---------|
| **Training** | 70% | Model parameter optimization |
| **Validation** | 15% | Hyperparameter tuning, early stopping |
| **Testing** | 15% | Final performance evaluation |

**Random Seed**: Fixed for reproducibility  
**Stratification**: Maintained class distribution across splits

### Data Augmentation (Applied During Training)

**Geometric Transformations**:
- Random rotation (±15°)
- Horizontal/vertical flipping
- Random shifting (10% width/height)
- Zoom range (0.9 - 1.1)

**Photometric Transformations**:
- Brightness adjustment (±20%)
- Contrast adjustment (±20%)

**Benefits**:
- Prevents overfitting
- Increases dataset diversity
- Improves model generalization

## 🔍 Data Quality Control

### Inclusion Criteria
✅ Clear brain anatomy visible  
✅ Minimal motion artifacts  
✅ Adequate contrast resolution  
✅ Properly labeled by medical experts  

### Exclusion Criteria
❌ Severe motion artifacts  
❌ Poor image quality  
❌ Incorrect orientation  
❌ Ambiguous labels  

## 📊 Dataset Usage

### Loading Dataset (Python Example)

```python
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
DATASET_DIR = "brain tumor dataset/"
TRAIN_DIR = os.path.join(DATASET_DIR, "Training")
VAL_DIR = os.path.join(DATASET_DIR, "Validation")
TEST_DIR = os.path.join(DATASET_DIR, "Testing")

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Class mapping
print(train_generator.class_indices)
# Output: {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
```

## 🧬 Research Significance

### Clinical Importance

**Early Detection**:
- Improves patient survival rates
- Enables timely intervention
- Reduces treatment complexity

**Accurate Classification**:
- Guides treatment planning
- Determines surgical vs. non-surgical management
- Informs prognosis discussions

**Decision Support**:
- Assists radiologists in diagnosis
- Reduces diagnostic variability
- Provides second opinion

### AI/ML Research Value

**Multi-Class Classification**:
- Challenging problem (4 classes)
- Real-world medical scenario
- Imbalanced data considerations

**Transfer Learning Benchmark**:
- Test pre-trained models (ImageNet)
- Compare custom architectures
- Evaluate domain adaptation

**Explainability Research**:
- Generate interpretable predictions
- Validate XAI techniques
- Build clinician trust

## 📈 Baseline Performance

Reported accuracies from literature:

| Model | Accuracy | Reference |
|-------|----------|-----------|
| VGG16 | 95-96% | Multiple studies |
| ResNet50 | 96-97% | State-of-the-art |
| Custom CNN | 97-98% | Our research |
| EfficientNet | 96-97% | Recent work |

## ⚠️ Ethical Considerations

### Data Privacy
- All images anonymized
- No patient identifiable information
- Publicly available dataset
- Appropriate for research use

### Usage Restrictions
✅ **Allowed**: Academic research, education, non-commercial projects  
❌ **Not allowed**: Clinical diagnosis without validation, commercial use without permission

### Acknowledgment
When using this dataset, please cite:
```
Nickparvar, M. (2021). Brain Tumor MRI Dataset. 
Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
```

## 🚀 Quick Start

```bash
# 1. Download dataset from Kaggle (if not already present)
# Requires Kaggle account and API key

# 2. Verify directory structure
ls "brain tumor dataset/"
# Should show: Training/ Validation/ Testing/

# 3. Check class distribution
python -c "
import os
for split in ['Training', 'Validation', 'Testing']:
    path = f'brain tumor dataset/{split}'
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f'{split}/{class_name}: {count} images')
"
```

## 📚 Related Resources

- **Model Training**: See [model_training_notebook/](../model_training_notebook/)
- **Backend API**: See [brain_tumor_identification_api/](../brain_tumor_identification_api/)
- **Frontend App**: See [braintumoridentificationapp/](../braintumoridentificationapp/)

## 🔗 External References

- [Neurosurgical Atlas - Brain Tumors](https://www.neurosurgicalatlas.com/)
- [Radiopaedia - Brain Tumors](https://radiopaedia.org/articles/brain-tumours)
- [WHO Classification of CNS Tumors](https://www.who.int/)
- [NCCN Guidelines - CNS Cancers](https://www.nccn.org/)

## 📄 License

Dataset license follows original Kaggle dataset terms. See individual source repositories for specific licensing.

---

**Dataset Ready**: 7,023 MRI images across 4 classes for brain tumor classification research!
