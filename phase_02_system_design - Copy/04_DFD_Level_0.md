# Data Flow Diagram (DFD) Level 0

## Project Title

**AI-Powered Image Captioning System Using CNN-Transformer Architecture**

---

# 1. Purpose

The DFD Level 0, also known as the Context Diagram, provides a high-level representation of the data exchanged between external entities and the Image Captioning System.

It focuses on the overall system without exposing internal implementation details.

---

# 2. Objective

The objectives of the DFD Level 0 are:

* Identify external entities interacting with the system.
* Illustrate major data inputs and outputs.
* Define the system boundary.
* Provide an overview of information flow.

---

# 3. External Entities

## End User

The primary actor who uploads images and receives generated captions.

---

## PostgreSQL Database

Stores prediction history, generated captions, timestamps, and related metadata for future retrieval and analysis.

---

# 4. Main Process

The central process is:

**AI-Powered Image Captioning System**

The system accepts image inputs, processes them using the trained CNN-Transformer model, generates descriptive captions, and stores prediction details.

---

# 5. Data Flow

### Input

* Image Upload

### Outputs

* Generated Caption
* Prediction History

### Storage

* Caption Records
* Image Metadata
* Prediction Timestamp

---

# 6. Description

The End User uploads an image to the Image Captioning System. The system processes the image and generates a descriptive caption. The generated caption is returned to the user while prediction details are stored in the PostgreSQL database.

---

# 7. Benefits

The DFD Level 0 provides a simplified view of system communication, making it easier for stakeholders to understand how information enters and leaves the application.

---

# 8. Conclusion

The DFD Level 0 establishes the overall data flow of the AI-Powered Image Captioning System and serves as the foundation for the more detailed DFD Level 1.
