# Brain Tumor MRI Dataset

## Overview

This directory contains the curated magnetic resonance imaging (MRI) dataset employed for training, validation, and testing of deep learning models in the context of automated brain tumor classification. The dataset constitutes a fundamental component of this MSc Computer Science (SLQF Level 10) research investigation, providing the empirical foundation for model development and performance evaluation.

## Dataset Characteristics

### Source and Composition

The dataset represents a consolidated collection aggregated from three established neuroimaging repositories:
- **figshare** - Research data repository
- **SARTAJ dataset** - Specialized brain tumor MRI collection
- **Br35H dataset** - Binary classification dataset (tumor/no tumor)

**Primary Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Dataset Statistics

- **Total Instances**: 7,023 MRI images
- **Image Modality**: T1-weighted, T2-weighted, and FLAIR MRI sequences
- **Classification Schema**: Multi-class categorization (4 classes)
- **Image Format**: Standardized resolution suitable for CNN input

### Taxonomic Classification

The dataset encompasses four distinct pathological and anatomical classifications:

1. **Glioma** - Primary malignant brain tumors originating from glial cells
2. **Meningioma** - Typically benign tumors arising from meningeal tissue
3. **Pituitary Adenoma** - Tumors of the pituitary gland
4. **No Tumor** - Control cases with no pathological findings

## Clinical Context

### Brain Tumor Pathophysiology

A brain tumor constitutes an abnormal proliferation of cells within the intracranial space. The rigid structure of the cranial cavity constrains any space-occupying lesions, rendering even benign tumors potentially life-threatening due to:
- Elevated intracranial pressure
- Mass effect on adjacent neural structures
- Disruption of cerebrospinal fluid dynamics
- Compression of vital brain regions

### Research Significance

Early detection and accurate classification of brain tumors are paramount for:
- **Treatment Planning**: Determining appropriate therapeutic interventions (surgical resection, radiotherapy, chemotherapy)
- **Prognosis Assessment**: Stratifying patients based on tumor type and malignancy grade
- **Clinical Decision Support**: Facilitating rapid diagnostic workflows
- **Survival Outcomes**: Improving patient survival rates through timely intervention

According to the World Health Organization (WHO), comprehensive brain tumor diagnosis necessitates:
1. Tumor detection and localization
2. Classification by histological type
3. Grading of malignancy
4. Molecular characterization (where applicable)

## Data Preprocessing and Quality Assurance

### Curation Methodology

The dataset underwent rigorous quality control procedures:

- **Class Imbalance Correction**: Images from the 'No Tumor' class were sourced exclusively from the Br35H dataset to ensure distribution quality
- **Mislabeled Data Removal**: Glioma class images from the SARTAJ dataset were excluded due to identified classification inconsistencies, and replaced with verified images from the figshare repository
- **Validation Protocol**: Cross-referencing with medical literature and expert annotations

### Data Organization

```
brain tumor dataset/
├── README.md                    # This documentation
├── dataset_details.txt          # Comprehensive metadata and provenance
└── [MRI images organized by class]
```

## Ethical Considerations

### Data Privacy and Compliance

All MRI images in this dataset are:
- De-identified and anonymized according to HIPAA guidelines
- Publicly available through established research repositories
- Used strictly for academic research purposes
- Subject to ethical review board approval for research utilization

### Attribution and Citation

Researchers utilizing this dataset should appropriately cite the original sources and acknowledge the contributions of the medical imaging community.

## Technical Specifications

### Image Properties

- **Color Space**: Grayscale (single channel)
- **Bit Depth**: Variable (typically 8-bit or 16-bit)
- **Preprocessing Requirements**: 
  - Intensity normalization
  - Skull stripping (optional)
  - Spatial standardization (resizing/resampling)
  - Data augmentation for model robustness

### Recommended Preprocessing Pipeline

1. **Intensity Normalization**: Z-score normalization or min-max scaling
2. **Spatial Standardization**: Resizing to uniform dimensions (e.g., 224×224, 256×256)
3. **Data Augmentation**: Rotation, flipping, zooming, brightness adjustment
4. **Train-Validation-Test Split**: Stratified partitioning (e.g., 70-15-15 or 80-10-10)

## Applications in This Research

This dataset serves as the foundation for:

1. **Model Training**: Development of deep convolutional neural network architectures
2. **Comparative Analysis**: Benchmarking multiple state-of-the-art architectures (VGG, ResNet, MobileNet, etc.)
3. **Class Balance Studies**: Investigating performance under balanced and imbalanced data distributions
4. **Explainable AI Research**: Generating interpretation visualizations (Grad-CAM, LIME, saliency maps)
5. **Generalization Assessment**: Evaluating model robustness and clinical applicability

## Related Components

- **Model Training**: See [Model Training Notebook](../model_training_notebook/README.md)
- **Backend Implementation**: See [Brain Tumor Identification API](../brain_tumor_identification_api/README.md)
- **Frontend Application**: See [Brain Tumor Identification App](../braintumoridentificationapp/README.md)

## References

1. Masoud Nickparvar (2021). Brain Tumor MRI Dataset. Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. World Health Organization. (2021). WHO Classification of Tumours of the Central Nervous System.
3. Louis, D. N., et al. (2016). The 2016 World Health Organization Classification of Tumors of the Central Nervous System: a summary. Acta Neuropathologica.

---

**Last Updated**: December 2025  
**MSc CS Research Project** | Postgraduate Institute of Science, University of Peradeniya
