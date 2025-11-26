from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI(title="MNIST Prediction API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Définition du modèle EXACTEMENT comme dans le notebook
# C'est un Sequential simple, pas une classe personnalisée !
model_architecture = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)


# Chargement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

try:
    model = model_architecture.to(device)
    model.load_state_dict(torch.load('model_mnist.pth', map_location=device, weights_only=True))
    model.eval()
    print("✅ Modèle PyTorch chargé avec succès")
    print(f"📱 Device: {device}")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    print("⚠️  L'API fonctionnera mais les prédictions échoueront")


class ImageData(BaseModel):
    image: str


def preprocess_image(image_data: str) -> torch.Tensor:
    """Prétraite l'image pour PyTorch"""
    if 'base64,' in image_data:
        image_data = image_data.split('base64,')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(image)
    
    # Inverser si nécessaire (MNIST = blanc sur noir)
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Normaliser
    img_array = img_array / 255.0
    
    # PyTorch: FLATTEN to (784,) pour le modèle Dense
    # PAS (1, 28, 28) car c'est un modèle Dense, pas CNN !
    img_tensor = torch.FloatTensor(img_array.flatten())
    
    return img_tensor.unsqueeze(0).to(device)  # Ajouter dimension batch


@app.get("/")
def root():
    return {
        "message": "API MNIST PyTorch fonctionnelle",
        "model_loaded": model is not None,
        "device": str(device),
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict-file": "/predict-file (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "status": "healthy",
        "model": "loaded",
        "device": str(device)
    }


@app.post("/predict")
async def predict(data: ImageData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        img_tensor = preprocess_image(data.image)
        
        with torch.no_grad():
            output = model(img_tensor)
            # Softmax pour obtenir les probabilités
            probabilities = torch.softmax(output, dim=1)
            digit = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][digit].item()
        
        # Toutes les probabilités pour debug
        probs_dict = {str(i): float(probabilities[0][i]) for i in range(10)}
        
        return {
            "digit": digit,
            "confidence": round(confidence, 4),
            "probabilities": probs_dict,
            "success": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de traitement : {str(e)}")


@app.post("/predict-file")
async def predict_from_file(file: UploadFile = File(...)):
    """Upload direct d'une image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
        
        img_array = np.array(image)
        if img_array.mean() > 127:
            img_array = 255 - img_array
        
        img_array = img_array / 255.0
        # FLATTEN pour le modèle Dense
        img_tensor = torch.FloatTensor(img_array.flatten()).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            digit = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][digit].item()
        
        return {
            "digit": digit,
            "confidence": round(confidence, 4),
            "success": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur : {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Démarrage de l'API MNIST...")
    print("📖 Documentation : http://localhost:8000/docs")
    print("🏥 Health check : http://localhost:8000/health\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)