# PROJECT CHARTER

## Project Title

**AI-Powered Image Captioning System Using CNN-Transformer Architecture**

---

## 1. Project Overview

The AI-Powered Image Captioning System is a deep learning application designed to automatically generate natural language descriptions for images. The system combines Computer Vision and Natural Language Processing techniques to understand image content and produce meaningful captions.

The project aims to assist organizations in automating image annotation tasks, improving accessibility, and reducing manual effort associated with content description generation.

---

## 2. Business Need

The rapid growth of digital image data across industries such as e-commerce, healthcare, social media, surveillance, and digital asset management has created a need for automated image understanding systems.

Manual image captioning is:

* Time-consuming
* Expensive
* Difficult to scale
* Prone to inconsistency

An automated image captioning system can significantly reduce operational costs while improving efficiency and accessibility.

---

## 3. Problem Statement

Organizations handling large image datasets require accurate textual descriptions for indexing, accessibility, search, and content management purposes. Generating these descriptions manually is inefficient and resource-intensive.

The proposed solution is an AI-powered image captioning system capable of automatically generating descriptive captions from images using deep learning models.

---

## 4. Project Objectives

### Primary Objectives

* Develop an automated image caption generation system.
* Generate meaningful captions from image inputs.
* Build a scalable AI pipeline.
* Deploy the solution through a web application and REST API.

### Secondary Objectives

* Improve accessibility for visually impaired users.
* Reduce manual image annotation effort.
* Demonstrate an end-to-end MLOps workflow.

---

## 5. Scope

### In Scope

* Image upload functionality
* Image preprocessing pipeline
* CNN-based feature extraction
* Transformer-based caption generation
* Model evaluation
* Flask API development
* Database integration
* Web application deployment
* Docker containerization

### Out of Scope

* Video caption generation
* Real-time surveillance systems
* Multilingual caption generation
* Speech synthesis
* Mobile application development

---

## 6. Expected Deliverables

### Documentation Deliverables

* Business Requirement Document (BRD)
* Software Requirement Specification (SRS)
* Architecture Design Documents
* System Diagrams
* Testing Documentation
* Internship Report

### Technical Deliverables

* Trained Deep Learning Model
* Image Captioning Pipeline
* REST API
* Database Integration
* Web Application
* Dockerized Deployment
* Cloud Deployment

---

## 7. Success Criteria

### Model Performance Metrics

| Metric              | Target                 |
| ------------------- | ---------------------- |
| BLEU Score          | > 0.50                 |
| CIDEr Score         | > 0.70                 |
| Validation Accuracy | Continuous Improvement |
| Training Stability  | Consistent Convergence |

### System Performance Metrics

| Metric                 | Target      |
| ---------------------- | ----------- |
| API Response Time      | < 3 Seconds |
| Upload Success Rate    | > 95%       |
| Availability           | > 99%       |
| Prediction Reliability | High        |

---

## 8. Assumptions

* Flickr8k dataset is available for training.
* GPU resources are available for model training.
* Internet connectivity is available for deployment activities.
* Required software libraries are accessible.

---

## 9. Constraints

* Limited internship duration.
* Dataset size limitations.
* Hardware resource limitations.
* Cloud deployment budget constraints.

---

## 10. Risk Analysis

| Risk                   | Impact | Mitigation Strategy                  |
| ---------------------- | ------ | ------------------------------------ |
| Overfitting            | High   | Data Augmentation and Regularization |
| Poor Caption Quality   | High   | Hyperparameter Tuning                |
| Long Training Time     | Medium | Transfer Learning                    |
| Deployment Failure     | Medium | Dockerization and Testing            |
| Dataset Quality Issues | Medium | Data Validation Pipeline             |

---

## 11. Stakeholders

| Stakeholder           | Responsibility                           |
| --------------------- | ---------------------------------------- |
| Project Developer     | Design, Development, Testing, Deployment |
| Internship Supervisor | Project Review and Guidance              |
| End Users             | Use and Evaluate System                  |
| Organization          | Assess Business Value                    |

---

## 12. Project Timeline

| Phase                          | Duration |
| ------------------------------ | -------- |
| Business Understanding         | Week 1   |
| System Design                  | Week 2   |
| Data Engineering               | Week 3   |
| Model Development              | Week 4   |
| MLOps Integration              | Week 5   |
| Backend & Frontend Development | Week 6   |
| Deployment                     | Week 7   |
| Documentation & Presentation   | Week 8   |

---

## 13. Technology Stack

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### Computer Vision

* EfficientNetB0 / ResNet50

### Natural Language Processing

* Transformer Decoder
* Text Vectorization

### Backend

* Flask

### Database

* PostgreSQL

### MLOps

* MLflow

### Deployment

* Docker
* Render / AWS

### Version Control

* Git
* GitHub

---

## 14. Expected Outcome

The final system will allow users to upload an image and automatically receive a meaningful natural language description generated by a CNN-Transformer-based deep learning model.

The project will demonstrate a complete industrial AI workflow, including data engineering, model development, MLOps, backend development, deployment, and monitoring.

---

## Approval

| Role                  | Name                       | Signature  |
| --------------------- | -------------------------- | ---------- |
| Project Developer     | Mohammed Bilal Shihabudeen | Mohammed Bilal Shihabudeen |
| Internship Supervisor | Shihabudeen Mohammad Haneefa |Shihabudeen Mohammad Haneefa |
| Organization Mentor   | __________                 | __________ |

---

**Document Version:** 1.0

**Prepared By:** Mohammed Bilal Shihabudeen

**Project:** AI-Powered Image Captioning System Using CNN-Transformer Architecture

**Date:** June 2026
