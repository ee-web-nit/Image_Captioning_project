# HIGH LEVEL ARCHITECTURE

## Purpose

The High-Level Architecture provides a simplified overview of the Image Captioning System and illustrates how major system components interact.

---

## System Flow

```text
                User
                  │
                  ▼
        Web Application (UI)
                  │
                  ▼
             Flask API
                  │
                  ▼
       Image Preprocessing
                  │
                  ▼
      CNN Feature Extractor
      (EfficientNetB0)
                  │
                  ▼
      Transformer Decoder
                  │
                  ▼
        Caption Generator
                  │
                  ▼
           PostgreSQL
                  │
                  ▼
           Response UI
```

---

## Components

### User

Uploads images and receives generated captions.

### Web Application

Provides image upload functionality and displays results.

### Flask API

Acts as the communication layer between frontend and AI model.

### Image Preprocessing

Resizes and normalizes images.

### CNN Feature Extractor

Extracts visual features from images.

### Transformer Decoder

Generates natural language captions.

### PostgreSQL Database

Stores prediction history and metadata.
