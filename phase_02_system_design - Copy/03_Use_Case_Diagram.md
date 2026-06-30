# Use Case Diagram

## Project Title

**AI-Powered Image Captioning System Using CNN-Transformer Architecture**

---

# 1. Purpose

The Use Case Diagram identifies the interactions between users and the Image Captioning System. It defines the functional capabilities offered by the application from the perspective of different actors.

---

# 2. Objectives

The objectives of the Use Case Diagram are to:

* Identify system actors.
* Define user interactions.
* Capture functional requirements.
* Provide a high-level view of system functionality.

---

# 3. Actors

## 3.1 End User

The End User interacts with the application to upload images and receive automatically generated captions.

Responsibilities:

* Upload images.
* Generate captions.
* View generated captions.
* View prediction history.

---

## 3.2 System Administrator

The System Administrator manages the deployed application and machine learning models.

Responsibilities:

* Manage trained models.
* Monitor system health.
* View application logs.
* Deploy updated models.

---

# 4. Use Cases

## End User

* Upload Image
* Generate Caption
* View Generated Caption
* View Prediction History

---

## System Administrator

* Manage Models
* Monitor System
* View Logs
* Deploy New Model

---

# 5. Description

The End User uploads an image through the web interface. The system processes the image using the deployed CNN-Transformer model and generates a descriptive caption. The caption is returned to the user and stored in the prediction history.

The System Administrator is responsible for maintaining the deployed application, monitoring performance, managing trained models, and deploying updated model versions.

---

# 6. Benefits

The Use Case Diagram provides a clear understanding of how different users interact with the system. It serves as a bridge between business requirements and software implementation by defining the expected functionality before development begins.

---

# 7. Conclusion

The Use Case Diagram captures the primary interactions between users and the AI-Powered Image Captioning System. It provides a foundation for designing workflows, APIs, and application interfaces during the development process.
