import os
import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "smartbreed_cattle_efficientnet_b5_300x300.pth"
)

IMG_SIZE = 300

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SmartBreedAIoT API",
    description="Indian Cattle Breed Classification API",
    version="1.0"
)

# Allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")
print("Model path:", MODEL_PATH)
print("Device:", DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Get class names saved during training
class_names = checkpoint["class_names"]

num_classes = len(class_names)

print("Number of classes:", num_classes)

# Create EfficientNet-B5
model = EfficientNet.from_name(
    "efficientnet-b5"
)

# Replace final layer
model._fc = nn.Linear(
    model._fc.in_features,
    num_classes
)

# Load trained weights
model.load_state_dict(
    checkpoint["model_state"]
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully!")

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "SmartBreedAIoT API is running",
        "model": "EfficientNet-B5",
        "image_size": "300x300",
        "classes": num_classes,
        "device": str(DEVICE)
    }

# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read uploaded image
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Preprocess
    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = class_names[
        predicted_idx.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )

    # Top 5 predictions
    top5_prob, top5_idx = torch.topk(
        probabilities,
        min(5, num_classes),
        dim=1
    )

    top5_predictions = []

    for prob, idx in zip(
        top5_prob[0],
        top5_idx[0]
    ):

        top5_predictions.append({
            "breed": class_names[idx.item()],
            "confidence": round(
                prob.item() * 100,
                2
            )
        })

    return {
        "success": True,
        "predicted_breed": predicted_class,
        "confidence": round(
            confidence_value,
            2
        ),
        "top5_predictions": top5_predictions
    }