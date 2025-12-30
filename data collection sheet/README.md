# 📊 Research Data Collection & Methodology

> **Experimental Design, Data Tracking, and Research Documentation**

## 🎯 What's Inside

This directory contains **research planning and data collection materials**:
- ✅ **Experimental design** documentation
- ✅ **Model performance tracking** sheets
- ✅ **Comparative analysis** templates
- ✅ **Research milestones** and progress logs
- ✅ **Statistical analysis** results

## 📂 Directory Contents

```
data collection sheet/
├── README.md                   # This file
└── Level 10 Research.xlsx     # Comprehensive research workbook
```

## 📋 Research Workbook Overview

**File**: `Level 10 Research.xlsx`

This Excel workbook serves as the central repository for experimental data and research planning.

### Potential Worksheets

1. **Literature Review**
   - Cataloging of relevant papers
   - Research gap analysis
   - Methodological approaches from literature
   - Citation tracking

2. **Research Questions & Hypotheses**
   - Primary research questions
   - Testable hypotheses
   - Expected outcomes
   - Success criteria

3. **Model Performance Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - ROC curves and AUC values
   - Training/validation loss curves

4. **Comparative Analysis**
   - VGG16 vs VGG19 vs ResNet50 vs MobileNetV2 vs GoogleLeNet vs Proposed
   - Balanced vs Imbalanced dataset results
   - XAI method comparisons
   - Statistical significance testing

5. **Hyperparameter Configuration Log**
   - Learning rate schedules
   - Batch sizes
   - Optimizer settings (Adam, SGD, RMSprop)
   - Regularization parameters (dropout, L2)
   - Data augmentation parameters

6. **XAI Validation Results**
   - Comprehensiveness scores
   - Sufficiency scores
   - Dice coefficients
   - Inter-method agreement metrics

7. **Experimental Timeline**
   - Research milestones
   - Gantt chart or project schedule
   - Deliverable deadlines
   - Progress tracking

8. **Dataset Provenance**
   - Data source details
   - Preprocessing steps applied
   - Train/val/test split records
   - Data quality checks

9. **Computational Environment**
   - Hardware specifications (CPU, GPU, RAM)
   - Software versions (Python, TensorFlow, libraries)
   - Random seeds for reproducibility
   - Training duration logs

10. **Statistical Analysis**
    - Hypothesis testing results
    - p-values and confidence intervals
    - Effect sizes
    - ANOVA / t-tests

## 🔬 Research Methodology

### Experimental Design

**Research Type**: Quantitative experimental research  
**Paradigm**: Applied research in medical AI  
**Methodology**: Comparative analysis with controlled variables

**Independent Variables**:
- CNN architecture type
- Dataset balance strategy
- XAI method selection
- Hyperparameter configurations

**Dependent Variables**:
- Classification accuracy
- Inference time
- XAI explanation quality
- Clinical interpretability

**Control Variables**:
- Dataset (same 7,023 MRI images)
- Image preprocessing
- Training/validation/test splits
- Hardware environment

### Data Collection Protocol

**Phase 1: Model Training**
1. Train each architecture (6 models × 2 datasets = 12 configurations)
2. Record training metrics (loss, accuracy per epoch)
3. Log computational resources used
4. Save model checkpoints

**Phase 2: Performance Evaluation**
1. Test on held-out test set
2. Generate confusion matrices
3. Calculate precision, recall, F1-score
4. Measure inference times

**Phase 3: XAI Generation**
1. Apply Grad-CAM, LIME, Saliency Maps
2. Generate visual explanations
3. Compute XAI validation metrics
4. Expert radiologist review (if applicable)

**Phase 4: LLM Integration**
1. Test multi-LLM pipeline
2. Evaluate report quality
3. Measure generation time
4. Clinical relevance assessment

**Phase 5: Statistical Analysis**
1. Compare model performances
2. Test for statistical significance
3. Conduct ablation studies
4. Document findings

## 📊 Data Recording Template

### Model Performance Template

| Model | Dataset | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|---------|----------|-----------|--------|----------|---------------|----------------|
| VGG16 | Balanced | 96.2% | 95.8% | 96.1% | 96.0% | 2.5 hrs | 35 ms |
| VGG19 | Balanced | 96.5% | 96.2% | 96.4% | 96.3% | 3.1 hrs | 38 ms |
| ResNet50 | Balanced | 97.1% | 96.9% | 97.0% | 97.0% | 2.8 hrs | 40 ms |
| ... | ... | ... | ... | ... | ... | ... | ... |

### XAI Metrics Template

| Model | Grad-CAM Comp | LIME Comp | Saliency Comp | Avg Sufficiency | Dice Coeff |
|-------|---------------|-----------|---------------|-----------------|------------|
| VGG16 | 0.82 | 0.79 | 0.85 | 0.68 | 0.74 |
| VGG19 | 0.84 | 0.81 | 0.87 | 0.71 | 0.76 |
| ... | ... | ... | ... | ... | ... |

## 🎓 Research Compliance

### Quality Assurance Checklist

- [ ] All experiments documented
- [ ] Random seeds recorded for reproducibility
- [ ] Software versions logged
- [ ] Hardware specifications noted
- [ ] Data splits preserved
- [ ] Model checkpoints saved
- [ ] Results peer-reviewed
- [ ] Statistical tests conducted
- [ ] Figures and tables prepared
- [ ] Thesis/paper draft updated

### Reproducibility Standards

**Version Control**:
- Git commits for code changes
- Model version tracking
- Dataset version control

**Documentation**:
- Clear experimental procedures
- Step-by-step protocols
- Troubleshooting notes

**Data Management**:
- Organized file structure
- Backup procedures
- Data integrity checks

## 🔍 Analysis Techniques

### Performance Metrics

**Classification Metrics**:
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print detailed report
print(classification_report(y_true, y_pred, 
    target_names=['glioma', 'meningioma', 'notumor', 'pituitary']))
```

**Statistical Significance**:
```python
from scipy.stats import ttest_rel, friedmanchisquare

# Paired t-test (comparing two models)
statistic, pvalue = ttest_rel(model1_accuracies, model2_accuracies)

# Friedman test (comparing multiple models)
statistic, pvalue = friedmanchisquare(
    vgg16_acc, vgg19_acc, resnet_acc, mobile_acc, google_acc, proposed_acc
)
```

### Visualization Templates

**Training Curves**:
```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_curve.png')
```

**Confusion Matrix Heatmap**:
```python
import seaborn as sns

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
```

## 📚 Related Components

- **Dataset**: See [brain tumor dataset/](../brain%20tumor%20dataset/)
- **Model Training**: See [model_training_notebook/](../model_training_notebook/)
- **Backend API**: See [brain_tumor_identification_api/](../brain_tumor_identification_api/)

## 🔗 Research Resources

### Thesis/Dissertation Guidelines
- MSc CS - SLQF Level 10 Research
- Postgraduate Institute of Science (PGIS)
- University of Peradeniya

### Statistical Software
- SPSS / R / Python (scipy, statsmodels)
- Excel for data organization
- GraphPad Prism for visualization

### Citation Management
- Mendeley / Zotero / EndNote
- IEEE / APA citation formats

## 📄 Deliverables Checklist

- [ ] Comprehensive data collection spreadsheet
- [ ] Model performance comparison tables
- [ ] Statistical analysis results
- [ ] Training curves and visualizations
- [ ] XAI validation metrics
- [ ] Thesis methodology chapter
- [ ] Research findings summary
- [ ] Future work recommendations

## 🎯 Usage Instructions

```bash
# Open the research workbook
open "data collection sheet/Level 10 Research.xlsx"

# Or use LibreOffice / Excel / Google Sheets

# Update sheets as experiments progress
# - Log each experiment run
# - Record all hyperparameters
# - Note any anomalies or observations
# - Calculate aggregate statistics
```

## ⚠️ Data Integrity

**Best Practices**:
- Regular backups (cloud + local)
- Version control for data files
- Audit trail for modifications
- Data validation checks
- Peer review of results

**Never**:
- Modify raw data without documentation
- Delete original recordings
- Cherry-pick results
- Skip failed experiments

## 📄 License

Research data follows university research ethics guidelines and is intended for academic use.

---

**Research Excellence**: Systematic documentation ensures reproducible, credible, and impactful research!
