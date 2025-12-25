# Emotion Detector – MLOps Sentiment Analysis Project

## 📌 Project Overview
This project focuses on the development of a **Machine Learning application for customer sentiment analysis** based on comments collected from social media platforms such as YouTube and Facebook.

The main objective is to design and implement an **end-to-end MLOps pipeline** that automates the entire machine learning lifecycle, from data preprocessing to model deployment, while ensuring reproducibility, scalability, and reliability.

---

## 🎯 Problem Statement
Traditional Machine Learning workflows rely heavily on manual processes, including data preparation, model training, testing, and deployment. These approaches often suffer from:
- Poor reproducibility
- High risk of human errors
- Difficult model lifecycle management
- Slow and unreliable deployment

This project addresses these challenges by introducing **MLOps practices** to automate and standardize the ML workflow.

---

## 🚀 Project Objectives
- Automate the Machine Learning lifecycle
- Ensure data and model versioning
- Track experiments and model performance
- Enable continuous integration and deployment (CI/CD)
- Deploy only models that meet performance criteria
- Provide a user-friendly interface for inference

---

## 🔄 Global Workflow
The implemented workflow follows these steps:

1. Source code versioning with GitHub  
2. Data versioning using DVC and MinIO  
3. Automated data preprocessing  
4. Automated model training  
5. Experiment tracking with MLflow  
6. Performance validation using a Model Gate  
7. Docker image build  
8. Image push to Nexus registry  
9. Deployment of the Streamlit application  

---

## 🧠 Model Gate (Key Feature)
A **performance-based Model Gate** is integrated into the CI/CD pipeline.

- The pipeline compares the new model’s F1-score with the previous best score.
- The Docker image is built and pushed **only if the new model outperforms the old one**.
- This prevents performance regression and ensures reliable deployment.

---

## 🛠️ Tools and Technologies

| Category | Tools |
|--------|-------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Object Storage | MinIO |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Image Registry | Sonatype Nexus |
| User Interface | Streamlit |

---

## 🧪 Machine Learning Model
- **Algorithm:** Logistic Regression  
- **Pipeline:** CountVectorizer + Classifier  
- **Why Logistic Regression?**
  - Simple and interpretable
  - Fast training
  - Suitable for text classification tasks

---

## 📊 Experiment Tracking
MLflow is used to:
- Log model parameters
- Track evaluation metrics
- Store training artifacts
- Compare experiments across runs

---

## 📦 Data Management and Versioning
- DVC tracks datasets and trained models  
- MinIO acts as an S3-compatible remote storage  
- Ensures reproducibility and traceability of experiments  

---

## 🐳 Containerization and Deployment
- Docker is used to package the application and its dependencies  
- Images are pushed to a private Nexus Docker registry  
- Each image is tagged using the Git commit SHA  

---

## 🖥️ User Interface
A **Streamlit web application** allows users to:
- Enter text comments  
- Obtain real-time sentiment predictions  
- Interact easily with the deployed model  

---

## ⚙️ CI/CD Pipeline
The pipeline is triggered by:
- Pushes to the `master` branch  
- A scheduled cron job  
- Manual execution (`workflow_dispatch`)  

It automates:
- Dependency installation  
- Data retrieval  
- Training and evaluation  
- Model validation (Model Gate)  
- Docker build and push  

---

## ⚠️ Limitations
- Simple ML model (no deep learning)  
- No real-time monitoring or alerting  
- Deployment is on-premise  
- No automatic retraining based on data drift  

---

## 🔮 Future Improvements
- Use advanced NLP models (BERT, Transformers)  
- Add monitoring and alerting (Prometheus, Grafana)  
- Implement automatic retraining strategies  
- Deploy on Kubernetes  
- Introduce feature stores  
- Multi-environment CI/CD (dev, staging, production)  

---

## 📌 Conclusion
This project demonstrates how **MLOps practices** significantly improve the reliability, automation, and scalability of Machine Learning systems.  
By combining data versioning, experiment tracking, CI/CD automation, and performance-based deployment, the solution provides a strong foundation for production-ready ML applications.
