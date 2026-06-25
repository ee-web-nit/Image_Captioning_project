# LOW LEVEL ARCHITECTURE

## Purpose

The Low-Level Architecture describes the detailed internal workflow of the AI-Powered Image Captioning System.

It illustrates how images move through the preprocessing, machine learning, storage, and response generation components.

---

# Detailed Workflow

```text
User Uploads Image
         │
         ▼
 Image Validation
         │
         ▼
 Image Preprocessing
 (Resize + Normalize)
         │
         ▼
 CNN Encoder
 (EfficientNetB0)
         │
         ▼
 Feature Extraction
         │
         ▼
 Transformer Decoder
         │
         ▼
 Caption Generation
         │
         ▼
 Result Validation
         │
         ▼
 PostgreSQL Storage
         │
         ▼
 Flask API Response
         │
         ▼
 User Interface
```

---

# Components

## Image Upload Module

Accepts image files from users.

Supported formats:

* JPG
* JPEG
* PNG

---

## Validation Module

Validates:

* File type
* File size
* Corrupted images

---

## Preprocessing Module

Operations:

* Image resizing
* Pixel normalization
* Tensor conversion

---

## CNN Encoder Module

Responsibilities:

* Feature extraction
* Transfer learning
* Visual representation generation

Recommended Model:

* EfficientNetB0

Alternative:

* ResNet50

---

## Feature Vector Module

Converts image into numerical representation.

Output:

High-dimensional feature vector.

---

## Transformer Decoder Module

Responsibilities:

* Attention mechanism
* Sequence generation
* Caption prediction

---

## Caption Generation Module

Generates final natural language description.

Example:

"A man riding a bicycle on a road."

---

## Database Module

Stores:

* Image name
* Generated caption
* Timestamp

Database:

PostgreSQL

---

## Flask API Module

Responsibilities:

* Receive requests
* Trigger model inference
* Return captions

Endpoint:

POST /predict

---

## User Interface Module

Displays:

* Uploaded image
* Generated caption
* Prediction history
