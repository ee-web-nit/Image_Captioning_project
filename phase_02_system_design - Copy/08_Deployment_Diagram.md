# Deployment Diagram

## Project Title

**AI-Powered Image Captioning System Using CNN-Transformer Architecture**

---

# 1. Purpose

The Deployment Diagram illustrates how the software components of the Image Captioning System are deployed across the execution environment.

It describes the physical architecture required to host the application.

---

# 2. Deployment Components

## Client Machine

* Web Browser
* User Interface

---

## Application Server

* Flask REST API
* Image Captioning Model
* Python Runtime

---

## Database Server

* PostgreSQL Database

---

## Cloud Infrastructure

* Docker Container
* AWS EC2 / Render
* Monitoring & Logging

---

# 3. Deployment Workflow

1. User accesses the application through a web browser.
2. Requests are sent to the Flask REST API.
3. The deployed CNN-Transformer model performs inference.
4. Prediction history is stored in PostgreSQL.
5. Generated captions are returned to the user.

---

# 4. Deployment Architecture

```text
End User
      │
      ▼
Web Browser
      │
      ▼
Docker Container
      │
      ▼
Flask REST API
      │
      ▼
Image Captioning Model
      │
      ▼
PostgreSQL Database
      │
      ▼
AWS EC2 / Render
```

---

# 5. Benefits

* Defines physical deployment.
* Supports scalable cloud hosting.
* Enables containerized deployment.
* Simplifies infrastructure management.

---

# 6. Conclusion

The Deployment Diagram provides a physical view of the AI-Powered Image Captioning System and demonstrates how application components are hosted, connected, and managed within the deployment environment.
