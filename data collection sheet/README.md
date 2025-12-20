# Research Data Collection and Methodology Documentation

## Overview

This directory contains the structured data collection instruments, experimental protocols, and methodological documentation developed for this MSc Computer Science (SLQF Level 10) research investigation. The materials housed within this component ensure methodological rigor, reproducibility, and adherence to established scientific research standards.

## Purpose and Scope

The data collection framework serves multiple critical functions within the research lifecycle:

1. **Experimental Design Documentation**: Systematic recording of research hypotheses, variables, and experimental conditions
2. **Data Acquisition Protocols**: Standardized procedures for gathering empirical evidence
3. **Metadata Management**: Comprehensive annotation of experimental parameters and configurations
4. **Reproducibility Assurance**: Detailed documentation enabling independent verification of research findings
5. **Progress Tracking**: Monitoring of research milestones and deliverables

## Contents

### Primary Documents

- **Level 10 Research.xlsx**: Comprehensive research planning and data collection instrument

This spreadsheet encompasses multiple dimensions of the research process, potentially including:

#### Research Planning Components
- **Literature Review Matrix**: Systematic cataloging of relevant scholarly works
- **Research Questions and Hypotheses**: Formalized research objectives and testable propositions
- **Experimental Design Schema**: Independent and dependent variables, control conditions
- **Timeline and Milestones**: Gantt charts or project scheduling artifacts

#### Data Collection Instruments
- **Model Performance Metrics**: Tabulated results across different architectures and configurations
- **Comparative Analysis Templates**: Structured comparison of model performance
- **Hyperparameter Configuration Log**: Systematic recording of training parameters
- **Validation Results**: Cross-validation scores, test set performance metrics
- **Statistical Analysis**: Significance testing, confidence intervals, effect sizes

#### Quality Assurance Documentation
- **Data Provenance**: Source, version, and preprocessing history of datasets
- **Experimental Conditions**: Hardware specifications, software versions, random seeds
- **Anomaly Logs**: Documentation of unexpected behaviors or outliers
- **Validation Checklists**: Quality control procedures and verification steps

## Methodological Framework

### Research Design Paradigm

This investigation employs a **quantitative experimental research design** with the following characteristics:

- **Research Type**: Applied research in computational medical imaging
- **Methodology**: Comparative experimental analysis
- **Data Nature**: Secondary data (publicly available MRI datasets)
- **Analysis Approach**: Statistical evaluation of deep learning model performance

### Key Research Variables

#### Independent Variables
- Model architecture (VGG16, VGG19, ResNet50, MobileNet, GoogleLeNet, Proposed architecture)
- Training data distribution (balanced vs. imbalanced)
- Hyperparameters (learning rate, batch size, epochs, optimizers)
- Data augmentation strategies

#### Dependent Variables
- Classification accuracy
- Precision, recall, and F1-score
- Area Under the Curve (AUC-ROC)
- Confusion matrix metrics
- Computational efficiency (training time, inference latency)
- Explainability metrics (faithfulness, stability, complexity)

#### Control Variables
- Dataset source and version
- Image preprocessing pipeline
- Hardware configuration
- Random seed initialization
- Cross-validation methodology

### Experimental Protocol

#### Phase 1: Data Preparation
1. Dataset acquisition and verification
2. Exploratory data analysis (EDA)
3. Preprocessing pipeline development
4. Train-validation-test partitioning

#### Phase 2: Model Development
1. Baseline model establishment
2. Architecture selection and implementation
3. Hyperparameter tuning via systematic search
4. Training protocol execution

#### Phase 3: Evaluation and Analysis
1. Performance metric calculation
2. Statistical significance testing
3. Comparative analysis across architectures
4. Explainability visualization generation

#### Phase 4: Documentation and Dissemination
1. Results compilation and interpretation
2. Manuscript preparation
3. Research artifact archival

## Data Collection Standards

### Quantitative Metrics

All experimental results are documented with:

- **Precision**: Minimum 4 decimal places for accuracy metrics
- **Replicability**: Multiple runs with different random seeds
- **Statistical Rigor**: Mean, standard deviation, confidence intervals
- **Comparative Context**: Baseline benchmarks and state-of-the-art comparisons

### Qualitative Observations

Supplementary qualitative data includes:

- Training convergence behavior notes
- Visual inspection of model predictions
- Failure case analysis
- Computational resource utilization observations

## Reproducibility Considerations

### Documentation Standards

To ensure independent replication of research findings:

1. **Complete Parameter Specification**: All hyperparameters, random seeds, software versions
2. **Environmental Description**: Hardware specifications, operating system, library versions
3. **Preprocessing Details**: Exact transformation pipeline, normalization parameters
4. **Data Splits**: Explicit indices or random seed for train-validation-test partitioning

### Version Control

- **Dataset Version**: Specific Kaggle dataset version or download date
- **Code Repository**: Git commit hash for exact code state
- **Model Checkpoints**: Saved weights and architecture definitions
- **Dependency Manifest**: requirements.txt or environment.yml files

## Integration with Research Components

This methodological documentation supports:

- **Dataset**: Provides context for [Brain Tumor Dataset](../brain%20tumor%20dataset/README.md) utilization
- **Training**: Guides experimental protocols in [Model Training Notebook](../model_training_notebook/README.md)
- **Implementation**: Informs deployment decisions in [API Backend](../brain_tumor_identification_api/README.md)
- **Validation**: Structures performance evaluation in [Frontend Application](../braintumoridentificationapp/README.md)

## Ethical and Regulatory Compliance

### Research Ethics

This investigation adheres to:

- **Institutional Review Board (IRB)**: Research conducted under appropriate ethical oversight
- **Data Privacy**: Use of de-identified, publicly available datasets only
- **Academic Integrity**: Proper attribution, citation, and acknowledgment of prior work
- **Responsible AI**: Commitment to transparency, fairness, and beneficence

### Limitations and Constraints

Explicitly documented limitations include:

- Non-clinical validation (not evaluated by medical professionals)
- Dataset bias and representativeness constraints
- Computational resource limitations
- Temporal scope of investigation

## Academic Contribution

This research contributes to the scholarly discourse on:

1. **Explainable AI in Medical Imaging**: Novel integration of multiple XAI techniques
2. **Comparative Deep Learning Analysis**: Systematic benchmarking of architectures
3. **Clinical Decision Support Systems**: Practical implementation frameworks
4. **Model Transparency**: Quantitative evaluation of explainability methods

## Future Research Directions

Potential extensions documented for future investigation:

- Multi-modal imaging integration (MRI + CT + PET)
- Longitudinal tumor progression tracking
- 3D volumetric analysis
- Federated learning for privacy-preserving model training
- Clinical validation studies with radiologist evaluation

## References and Standards

1. **Research Methodology**: Creswell, J. W., & Creswell, J. D. (2017). Research design: Qualitative, quantitative, and mixed methods approaches.
2. **Medical Imaging Standards**: DICOM (Digital Imaging and Communications in Medicine)
3. **AI Ethics**: IEEE Ethically Aligned Design principles
4. **Statistical Analysis**: American Psychological Association (APA) guidelines for statistical reporting

---

**Research Level**: MSc Computer Science (SLQF Level 10)  
**Institution**: Postgraduate Institute of Science, University of Peradeniya  
**Research Code**: SC 699 - Level 10 Research  
**Last Updated**: December 2025
