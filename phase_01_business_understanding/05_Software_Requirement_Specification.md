# SOFTWARE REQUIREMENT SPECIFICATION (SRS)

## Project Title

AI-Powered Image Captioning System Using CNN-Transformer Architecture

---

# 1. Introduction

## 1.1 Purpose

This document specifies the functional and non-functional requirements for the AI-Powered Image Captioning System.

The system is designed to automatically generate descriptive captions for uploaded images using deep learning techniques.

---

## 1.2 Scope

The system will:

* Accept image uploads
* Process image data
* Generate captions
* Store prediction history
* Provide API access
* Support cloud deployment

---

# 2. System Overview

The Image Captioning System combines Computer Vision and Natural Language Processing techniques.

The workflow includes:

1. Image Upload
2. Image Preprocessing
3. Feature Extraction
4. Caption Generation
5. Result Presentation
6. Prediction Storage

---

# 3. Functional Requirements

## FR-1: Image Upload

### Description

The system shall allow users to upload image files.

### Input

* JPG
* JPEG
* PNG

### Output

Uploaded image stored temporarily for processing.

---

## FR-2: Image Validation

### Description

The system shall validate uploaded files.

### Validation Rules

* File must be an image
* File size must be within limits
* Corrupted files must be rejected

---

## FR-3: Image Preprocessing

### Description

The system shall preprocess images before inference.

### Activities

* Resize image
* Normalize pixel values
* Convert image to model input format

---

## FR-4: Feature Extraction

### Description

The system shall extract visual features using a pretrained CNN encoder.

### Expected Encoder

* EfficientNetB0
  or
* ResNet50

---

## FR-5: Caption Generation

### Description

The system shall generate captions using a Transformer Decoder.

### Output Example

"A dog running through a grassy field."

---

## FR-6: Prediction Storage

### Description

The system shall store prediction results.

### Stored Data

* Image Name
* Generated Caption
* Timestamp

---

## FR-7: REST API

### Description

The system shall provide REST API endpoints.

### Endpoint

POST /predict

### Response

Generated caption in JSON format.

---

## FR-8: User Interface

### Description

The system shall provide a web interface for image upload and caption display.

---

# 4. Non-Functional Requirements

## Performance

* Response time < 3 seconds
* Efficient model inference

---

## Scalability

* Support future cloud deployment
* Support increasing user requests

---

## Reliability

* Stable operation
* Error handling
* Logging support

---

## Maintainability

* Modular architecture
* Version control
* Reusable components

---

## Security

* File validation
* Input sanitization
* Restricted uploads

---

# 5. System Constraints

* Internship duration limitations
* Hardware resource limitations
* Dataset size limitations
* Deployment budget limitations

---

# 6. Assumptions

* Flickr8k dataset is available
* GPU resources are available
* Internet access is available
* Required software dependencies are available

---

# 7. Technology Requirements

## Programming Language

* Python

## Deep Learning

* TensorFlow
* Keras

## Backend

* Flask

## Database

* PostgreSQL

## MLOps

* MLflow

## Deployment

* Docker
* Render / AWS

---

# 8. Acceptance Criteria

The system will be considered successful if:

* Users can upload images successfully
* Captions are generated correctly
* API responses are returned successfully
* Prediction history is stored
* Deployment is functional
* Evaluation metrics meet project targets

---

# 9. Conclusion

The Software Requirement Specification provides the technical foundation for developing the AI-Powered Image Captioning System and serves as a reference for design, implementation, testing, and deployment activities.
