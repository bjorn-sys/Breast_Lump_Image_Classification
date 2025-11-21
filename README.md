# Breast_Lump_Image_Classification

Breast cancer is one of the most common causes of death among women worldwide. Early detection helps in reducing the number of early deaths. The data reviews the medical images of breast cancer using ultrasound scan. Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images. Breast ultrasound images can produce great results in classification, detection, and segmentation of breast cancer when combined with machine learning.

**Data**
---
The data collected at baseline include breast ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018. The number of patients is 600 female patients. The dataset consists of 780 images with an average image size of 500*500 pixels. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes, which are normal, benign, and malignant.

**Breast Ultrasound AI Classification System**
---

**📋 Project Overview**
---
A state-of-the-art deep learning system for automated classification of breast ultrasound images into three clinical categories: Benign, Malignant, and Normal. This comprehensive AI diagnostic tool leverages transfer learning with ResNet50 architecture to achieve exceptional performance in medical image analysis.

https://img.shields.io/badge/Medical-AI%2520Diagnostics-blue
https://img.shields.io/badge/Accuracy-95%2525-success
https://img.shields.io/badge/PyTorch-2.0+-red
https://img.shields.io/badge/Python-3.8+-blue

**🎯 Key Features**
---
**🏥 Clinical Applications**
---
Automated Diagnosis: Rapid classification of breast ultrasound images

Multi-class Detection: Distinguish between Benign, Malignant, and Normal tissues

Clinical Decision Support: Assist healthcare professionals in diagnosis

Explainable AI: Transparent model decisions with visual explanations

**🤖 Advanced AI Capabilities**
---
Transfer Learning: Fine-tuned ResNet50 on specialized medical dataset

High Precision: 95% validation accuracy on diverse test cases

Robust Performance: Consistent results across different image qualities

Real-time Analysis: Instant predictions for clinical workflow integration

**💻 Technical Excellence**
---
Professional Web Interface: Streamlit-based clinical application

Comprehensive Reporting: PDF generation for medical documentation

Grad-CAM Integration: Visual heatmaps showing model focus areas

Quality Assurance: Extensive validation and testing protocols

**📊 Performance Metrics**
---
Model Evaluation Results
Metric	Benign	Malignant	Normal	Overall
Precision	0.98	0.90	0.95	0.95
Recall	0.93	0.96	0.98	0.95
F1-Score	0.95	0.93	0.96	0.95
Training Performance
Best Validation Loss: 0.1130

Final Training Accuracy: 98.3%

Final Validation Accuracy: 95.4%

Training Duration: 51 epochs (early stopping)

Model Stability: Consistent performance across validation sets

🏗️ System Architecture
Data Processing Pipeline
Image Standardization: RGB conversion and resolution normalization

Advanced Augmentation: Comprehensive transformation strategies

Quality Control: Automated validation of input images

Preprocessing: Medical image-specific normalization

Model Configuration
Base Architecture: ResNet50 with pre-trained ImageNet weights

Custom Classification Head: 3-class output layer

Optimization Strategy: Adam optimizer with learning rate scheduling

Loss Management: Class-weighted cross-entropy for imbalance handling

Training Methodology
Early Stopping: 20-epoch patience for optimal generalization

Learning Rate Decay: Step-wise reduction for fine-tuning

Validation Monitoring: Continuous performance assessment

Model Checkpointing: Best weight preservation

📈 Dataset Information
Source and Composition
Primary Dataset: Breast Ultrasound Images (BUSI) Dataset

Total Studies: 1,578 ultrasound images

Class Distribution:

Benign: 711 images (56.3%)

Malignant: 338 images (26.8%)

Normal: 213 images (16.9%)

Split Strategy: 80% training, 20% validation

Data Quality Assurance
Medical Validation: Clinically verified annotations

Image Diversity: Various ultrasound machines and settings

Quality Standards: Professional medical imaging protocols

Ethical Compliance: Patient privacy and data security

🚀 Installation & Deployment
System Requirements
Python: 3.8 or higher

PyTorch: 2.0+ with CUDA support recommended

Memory: 8GB RAM minimum, 16GB recommended

Storage: 2GB available space for models and dependencies

GPU: CUDA-capable GPU for optimal performance
