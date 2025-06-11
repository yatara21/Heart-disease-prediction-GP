# Heart Disease Risk Assessment Application

<div align="center">

![Python](https://img.shields.io/badge/python-v3.9-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0.0-lightgrey.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A machine learning web application that predicts heart disease risk using Random Forest classification.  
Built as a graduation project for Egyptian E-Learning University (EELU).

[Key Features](#-key-features) •
[Getting Started](#-getting-started) •
[Usage Guide](#-usage-guide) •
[Deployment](#-deployment-options) •
[Model Information](#-model-information)

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technical Stack](#-technical-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Model Information](#-model-information)
  - [Dataset Features](#dataset-features)
  - [Model Performance](#model-performance)
- [Deployment Options](#-deployment-options)
  - [Docker Deployment](#docker-deployment)
  - [Azure Cloud Deployment](#azure-cloud-deployment)
- [Usage Guide](#-usage-guide)
- [Contributors](#-contributors)
- [License](#-license)


## 🔍 Overview
This application leverages machine learning to predict heart disease risk based on medical parameters. It provides healthcare professionals with a user-friendly interface for instant risk assessments.

## ✨ Key Features
- **Real-time Predictions**: Instant heart disease risk assessment
- **User-friendly Interface**: Clean, responsive design with input validation
- **High Accuracy**: 91.1% prediction accuracy
- **Confidence Levels**: Detailed prediction confidence metrics
- **Deployment Ready**: Docker support for easy deployment
- **Cloud Compatible**: Azure deployment configuration included

## 🛠 Technical Stack
- **Backend Framework**: Python 3.9, Flask 3.0.0
- **Frontend**: HTML5, TailwindCSS
- **Machine Learning**: scikit-learn 1.3.0, numpy, pandas
- **Containerization**: Docker
- **Cloud Platform**: Microsoft Azure
- **Version Control**: Git

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker (for containerized deployment)
- Azure CLI (for cloud deployment)

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yatara21/Heart-disease-prediction-GP.git
cd Heart-disease-prediction-GP
```

2. **Set Up Virtual Environment**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Application**
```bash
python app.py
```

## 📊 Model Information

### Dataset Features
| Feature | Type | Range | Description |
|---------|------|--------|-------------|
| Age | Numeric | 28-77 | Patient's age in years |
| Sex | Categorical | M/F | Gender |
| ChestPainType | Categorical | TA/ATA/NAP/ASY | Type of chest pain |
| RestingBP | Numeric | 80-200 | Resting blood pressure (mm Hg) |
| Cholesterol | Numeric | 85-603 | Serum cholesterol (mg/dl) |
| FastingBS | Binary | 0/1 | Fasting blood sugar > 120 mg/dl |
| RestingECG | Categorical | Normal/ST/LVH | Resting electrocardiogram results |
| MaxHR | Numeric | 60-202 | Maximum heart rate achieved |
| ExerciseAngina | Binary | Y/N | Exercise-induced angina |
| Oldpeak | Numeric | -2.6-6.2 | ST depression induced by exercise |
| ST_Slope | Categorical | Up/Flat/Down | Slope of peak exercise ST segment |

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| Random Forest | 88.04% | 89% | 88% | 88% |
| Logistic Regression | 85.87% | 86% | 86% | 86% |
| SVM (RBF) | 74.46% | 75% | 74% | 74% |
| MLP Classifier | 85.33% | 86% | 85% | 85% |

***NOTE***: The model performance numbers may differ every time you train your models

## 🐳 Deployment Options

### Docker Deployment
```bash
# Build Image
docker build -t heart-disease-prediction .

# Run Container
docker run -p 80:80 heart-disease-prediction
```

### Azure Cloud Deployment
```bash
# Login to Azure
az login

# Create and Configure ACR
az acr create --resource-group YOUR_RESOURCE_GROUP --name YOUR_REGISTRY_NAME --sku Basic
az acr login --name YOUR_REGISTRY_NAME

# Deploy Container
docker tag heart-disease-prediction YOUR_REGISTRY_NAME.azurecr.io/heart-disease-prediction:latest
docker push YOUR_REGISTRY_NAME.azurecr.io/heart-disease-prediction:latest
```

## 💻 Usage Guide
1. Access the application (local: `http://localhost:80`)
2. Input patient medical parameters
3. Click "Assess Risk"
4. View prediction results and confidence level

🌐 Web Application

    🔗 Live Demo: http://eelu-demo-app.westeurope.azurecontainer.io

Users can enter patient data through an intuitive form and receive an instant prediction of heart disease presence or absence.

🧩 Challenges

    Data Quality: Cleaned invalid values (e.g., zero cholesterol)

    Model Interpretability: Balanced accuracy vs. explainability

    Deployment Compatibility: Resolved environment issues with Docker

    Web Input Matching: Handled Flask form reshaping for model prediction

🚧 Limitations & Future Work

    Use larger, more diverse datasets

    Incorporate lifestyle and genetic data

    Integrate model into hospital EHR systems

    Implement SHAP/LIME for interpretability

    Develop a mobile-friendly version

    Validate in clinical trials
    
👨‍💻 Contributors

    Khalid Elshawadfy Ahmed

    Ammar Mohamed Hassan

    Abdullah Ahmed Abdellazim

    Hossam Hassan Mohamed

    Mohamed Elsayed Abdelsamad

    Omar Abdelrhaman Yousef

Supervised by:

    Prof. Ahmed Ezz (RIP)

    Dr. Yasmin Mahmoud

    Eng. Nourhan Salah

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
© 2025 EELU-Graduation-Project - Egyptian E-Learning University.  
All rights reserved.
</div>
