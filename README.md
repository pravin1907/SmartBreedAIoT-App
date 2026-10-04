# SmartBreedAIoT-App 🐄

AI-powered Indian cattle breed classification application using **EfficientNet-B5** and **FastAPI**.

## 📌 Overview

SmartBreedAIoT-App is an AI-based cattle breed classification system designed to identify Indian cattle breeds from images.

The application uses a trained **EfficientNet-B5** deep learning model to classify cattle images into **50 different breeds**.

The current version provides a FastAPI backend that accepts a cattle image and returns the predicted breed, confidence score, and top-5 predictions.

## ✨ Features

- 🐄 Indian cattle breed classification
- 🤖 EfficientNet-B5 deep learning model
- 📷 Image-based prediction
- 🎯 50 cattle breed classes
- 📊 Confidence score for predictions
- 🔝 Top-5 breed predictions
- ⚡ FastAPI REST API
- 💻 Local API testing through Swagger UI
- 📦 Model stored using Git LFS

## 🧠 Model

**Model:** EfficientNet-B5  
**Input Size:** 300 × 300 pixels  
**Number of Classes:** 50  
**Validation Accuracy:** 67.90%

The model was trained on **8,531 cattle images**.

### Model Performance

| Metric | Result |
|---|---:|
| Top-1 Accuracy | 67.90% |
| Top-5 Accuracy | 92.15% |
| Macro Precision | 65.51% |
| Macro Recall | 65.26% |
| Macro F1 | 64.80% |
| Weighted F1 | 67.21% |

## 🛠️ Tech Stack

### Machine Learning
- Python
- PyTorch
- EfficientNet-B5
- Torchvision
- PIL
- NumPy

### Backend
- FastAPI
- Uvicorn
- Python

### Development
- VS Code
- Git
- GitHub
- Git LFS

### Future
- Flutter mobile application

## 📂 Project Structure

```text
SmartBreedAIoT-App/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   │
│   └── model/
│       └── smartbreed_cattle_efficientnet_b5_300x300.pth
│
├── .gitattributes
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/SmartBreedAIoT-App.git
```

```bash
cd SmartBreedAIoT-App
```

### 2. Open the backend

```bash
cd backend
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Run the API

From the `backend` directory:

```bash
uvicorn app:app --reload --port 8000
```

The API will run at:

```text
http://127.0.0.1:8000
```

## 📖 API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

### Prediction Endpoint

```text
POST /predict
```

Upload a cattle image and the API returns:

- Predicted breed
- Prediction confidence
- Top-5 predictions

### Example Response

```json
{
  "success": true,
  "predicted_breed": "Umblachery",
  "confidence": 91.95,
  "top5_predictions": [
    {
      "breed": "Umblachery",
      "confidence": 91.95
    },
    {
      "breed": "Pulikulam",
      "confidence": 2.57
    },
    {
      "breed": "Kenkatha",
      "confidence": 1.73
    },
    {
      "breed": "Malnad_gidda",
      "confidence": 1.45
    },
    {
      "breed": "Nimari",
      "confidence": 0.58
    }
  ]
}
```

## 🔄 Prediction Pipeline

```text
Cattle Image
     ↓
Image Upload
     ↓
FastAPI
     ↓
Image Preprocessing
     ↓
EfficientNet-B5
     ↓
50-Class Classification
     ↓
Predicted Breed
     ↓
Confidence + Top-5 Results
```

## 🧪 Current Status

### Completed

- [x] EfficientNet-B5 model trained
- [x] 300 × 300 model configuration
- [x] 50-breed classification
- [x] FastAPI backend
- [x] Image upload API
- [x] Prediction endpoint
- [x] Confidence score
- [x] Top-5 predictions
- [x] Local API testing
- [x] GitHub repository
- [x] Git LFS model storage

### Planned

- [ ] Flutter mobile application
- [ ] Camera-based cattle image capture
- [ ] Mobile image upload
- [ ] Connect Flutter frontend with FastAPI
- [ ] Deploy backend
- [ ] Deploy the complete application

## 🎯 Future Scope

The application can be extended with additional cattle-related AI and IoT capabilities, including integration with sensor-based data and other SmartBreedAIoT components.

## 👨‍💻 Project

**SmartBreedAIoT-App**

AI-based Indian cattle breed classification using EfficientNet-B5.
