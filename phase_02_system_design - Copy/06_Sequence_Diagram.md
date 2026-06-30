# Sequence Diagram

## Project Title

**AI-Powered Image Captioning System Using CNN-Transformer Architecture**

---

# 1. Purpose

The Sequence Diagram illustrates the chronological interaction between system components during image caption generation.

It focuses on the order in which messages are exchanged between the user, frontend, backend, AI model, and database.

---

# 2. Participants

* End User
* Web Browser
* Frontend UI
* Flask REST API
* Image Captioning Model
* PostgreSQL Database

---

# 3. Interaction Flow

1. End User uploads an image.
2. Frontend sends the image to the Flask REST API.
3. Flask validates the uploaded image.
4. Image is preprocessed.
5. The trained CNN-Transformer model performs inference.
6. A caption is generated.
7. Prediction details are stored in PostgreSQL.
8. Caption is returned to the frontend.
9. Frontend displays the generated caption to the End User.

---

# 4. Sequence Summary

```text
End User
    │
    ▼
Frontend
    │
    ▼
Flask REST API
    │
    ▼
Image Captioning Model
    │
    ▼
PostgreSQL
    │
    ▼
Frontend
    │
    ▼
End User
```

---

# 5. Benefits

* Defines execution order.
* Helps understand runtime interactions.
* Assists API design.
* Useful during debugging and integration testing.

---

# 6. Conclusion

The Sequence Diagram demonstrates the complete request-response cycle followed by the Image Captioning System during inference.
