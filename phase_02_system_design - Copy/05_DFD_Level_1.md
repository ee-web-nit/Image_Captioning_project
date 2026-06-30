# Data Flow Diagram (DFD) Level 1

## Project Title

**AI-Powered Image Captioning System Using CNN-Transformer Architecture**

---

# 1. Purpose

The DFD Level 1 expands the single process shown in the DFD Level 0 into a detailed view of the internal processing steps performed by the Image Captioning System.

It illustrates how image data flows through preprocessing, deep learning inference, caption generation, and storage before being returned to the user.

---

# 2. Objective

The objectives of the DFD Level 1 are to:

* Describe the internal processing workflow.
* Illustrate the movement of image data through each processing stage.
* Show interactions with the database.
* Define system outputs generated after processing.

---

# 3. Processes

### Process 1: Image Upload

The End User uploads an image through the web interface.

**Input:** Image File

**Output:** Uploaded Image

---

### Process 2: Image Validation

The uploaded image is validated to ensure it is in a supported format and free from corruption.

**Input:** Uploaded Image

**Output:** Validated Image

---

### Process 3: Image Preprocessing

The validated image is resized, normalized, and converted into the format required by the CNN encoder.

**Input:** Validated Image

**Output:** Processed Image Tensor

---

### Process 4: Caption Generation

The processed image is passed through the CNN Feature Extractor and Transformer Decoder to generate a natural language description.

**Input:** Processed Image Tensor

**Output:** Generated Caption

---

### Process 5: Store Prediction

The generated caption and associated metadata are stored in the PostgreSQL database.

**Stored Information**

* Image Name
* Generated Caption
* Timestamp

---

### Process 6: Display Caption

The generated caption is returned to the End User through the web application.

---

# 4. Data Stores

## PostgreSQL Database

Stores:

* Prediction History
* Generated Captions
* Metadata
* Timestamps

---

# 5. External Entity

**End User**

Provides image input and receives generated captions.

---

# 6. Overall Workflow

```text
End User
      ↓
Upload Image
      ↓
Image Validation
      ↓
Image Preprocessing
      ↓
CNN Feature Extractor
      ↓
Transformer Decoder
      ↓
Generate Caption
      ↓
Store Prediction
      ↓
Return Caption
```

---

# 7. Benefits

* Clearly represents the internal data processing pipeline.
* Bridges system architecture and implementation.
* Improves understanding of application workflow.
* Serves as a reference during development and testing.

---

# 8. Conclusion

The DFD Level 1 provides a detailed representation of the internal processing performed by the AI-Powered Image Captioning System, enabling developers and stakeholders to understand how data moves through the application from image upload to caption generation.
