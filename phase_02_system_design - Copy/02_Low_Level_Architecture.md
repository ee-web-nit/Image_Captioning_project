# Low-Level Architecture

## Project Title

**AI-Powered Image Captioning System Using CNN-Transformer Architecture**

---

# 1. Purpose

The Low-Level Architecture (LLA) provides a detailed representation of the internal components and workflows of the Image Captioning System. Unlike the High-Level Architecture, which presents the overall system modules, the LLA focuses on the internal implementation details, data flow, model lifecycle, and deployment process.

This document serves as a blueprint for developers, AI engineers, and system architects during implementation and maintenance.

---

# 2. Objectives

The objectives of the Low-Level Architecture are to:

* Describe the complete AI workflow from data ingestion to inference.
* Illustrate the interaction between machine learning components.
* Define the model training and deployment lifecycle.
* Explain the role of MLOps in experiment tracking and model management.
* Provide a reference for software development and deployment.

---

# 3. Architecture Overview

The system is divided into four major layers:

1. Training Pipeline
2. MLOps Layer
3. Application & Deployment Layer
4. Inference Pipeline

Each layer is responsible for a specific stage of the machine learning lifecycle and collectively enables an end-to-end AI application.

---

# 4. Training Pipeline

The training pipeline is responsible for developing the image captioning model using the Flickr8k dataset.

### Workflow

1. Flickr8k Dataset
2. Image Validation
3. Image Preprocessing
4. Caption Tokenization
5. CNN Feature Extraction (EfficientNetB0)
6. Transformer Decoder
7. Model Training
8. BLEU/CIDEr Evaluation

### Description

The Flickr8k dataset is first validated to ensure data integrity. Images are then preprocessed by resizing and normalizing pixel values. Captions are tokenized to convert text into numerical sequences suitable for deep learning.

A pretrained EfficientNetB0 model extracts visual features from each image, while the Transformer Decoder learns to generate natural language descriptions. After training, the model is evaluated using BLEU and CIDEr metrics to assess caption quality.

---

# 5. Saved Model

After successful training, the best-performing model is exported and stored as a serialized model file (e.g., `.keras`).

The saved model serves as the production model used during inference without requiring retraining.

---

# 6. MLOps Layer

The MLOps layer manages the machine learning lifecycle by tracking experiments, storing metrics, and versioning trained models.

### Components

* MLflow Experiment Tracking
* Metrics Logging
* Model Registry
* Artifact Storage
* Best Model Selection

### Responsibilities

* Track training experiments.
* Compare model performance.
* Store evaluation metrics.
* Maintain version history.
* Register the best-performing model for deployment.

---

# 7. Application & Deployment Layer

This layer hosts the deployed AI application and manages user requests.

### Components

* Docker Container
* Cloud Hosting (AWS EC2 / Render)
* Flask REST API
* Loaded Model
* PostgreSQL Database
* Monitoring & Logging

### Responsibilities

* Host the application.
* Load the trained model.
* Serve REST API requests.
* Store prediction history.
* Monitor application performance.

---

# 8. Inference Pipeline

The inference pipeline executes whenever a user uploads an image.

### Workflow

1. End User
2. Web Browser
3. Frontend User Interface
4. Flask REST API
5. Image Validation
6. Image Preprocessing
7. Load Saved Model
8. CNN Feature Extractor
9. Transformer Decoder
10. Caption Generation
11. Return Caption to User

### Description

When an image is uploaded, the Flask API validates the input and preprocesses the image. The deployed model extracts visual features using EfficientNetB0 and generates a caption through the Transformer Decoder.

The generated caption is returned to the user, while prediction details are stored in PostgreSQL for future reference and analytics.

---

# 9. Data Flow Summary

The overall data flow of the system is illustrated below:

```text
Training Pipeline
        │
        ▼
 Saved Model (.keras)
        │
        ▼
     MLflow Registry
        │
        ▼
Application & Deployment
        │
        ▼
Inference Pipeline
        │
        ▼
Generated Caption
```

---

# 10. Design Principles

The system architecture follows the following software engineering principles:

* Modular Design
* Separation of Concerns
* Scalability
* Maintainability
* Reusability
* MLOps Integration
* Cloud Deployment Readiness

These principles ensure that each component can be developed, tested, deployed, and maintained independently.

---

# 11. Benefits

The proposed architecture offers several advantages:

* Clear separation between training and inference workflows.
* Improved maintainability through modular design.
* Efficient experiment tracking using MLflow.
* Production-ready deployment using Docker and cloud infrastructure.
* Scalable REST API for real-time caption generation.
* Persistent storage of prediction history using PostgreSQL.

---

# 12. Conclusion

The Low-Level Architecture provides a comprehensive technical blueprint for implementing the AI-Powered Image Captioning System. By separating training, model management, deployment, and inference into independent layers, the architecture improves scalability, maintainability, and reproducibility. This design follows modern machine learning engineering practices and supports the development of a robust end-to-end AI application suitable for real-world deployment.
