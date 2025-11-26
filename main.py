from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import os
import uvicorn 

app = FastAPI(title="MNIST Prediction API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DÉFINITION DU MODÈLE CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28) 
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Chargement du modèle ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel()
model_path = 'new_cnn_model_mnist.pth' 

if not os.path.exists(model_path):
    print(f"Erreur: Le fichier du modèle '{model_path}' est introuvable.")
    model = None 
else:
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Modèle chargé avec succès depuis '{model_path}' sur le périphérique : {device}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        model = None

# --- Schéma Pydantic pour l'entrée base64 ---
class ImageInput(BaseModel):
    image: str

# --- Endpoints de l'API ---

@app.post("/predict")
async def predict(data: ImageInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        header, encoded = data.image.split(',', 1)
        image_data = base64.b64decode(encoded)
        
        image_buffer = io.BytesIO(image_data)
        image = Image.open(image_buffer)
        
        image = image.convert('L').resize((28, 28), Image.Resampling.NEAREST)
        image_buffer.close()
        
        img_array = np.array(image)
        
        # ⭐ CORRECTION: Suppression de la logique d'inversion des couleurs. 
        # Le canvas du frontend envoie désormais les chiffres en blanc sur fond noir.
        
        # Normalisation (0 à 1)
        img_array = img_array / 255.0

        # FIX: Création explicite du tenseur 4D (1, 1, 28, 28)
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
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
        raise HTTPException(status_code=400, detail=f"Erreur de traitement : {str(e)}")


@app.post("/predict-file")
async def predict_from_file(file: UploadFile = File(...)):
    # L'endpoint /predict-file a conservé sa logique d'inversion pour gérer
    # les images téléchargées qui pourraient être en noir sur blanc.
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        contents = await file.read()
        image_buffer = io.BytesIO(contents)
        image = Image.open(image_buffer)
        
        image = image.convert('L').resize((28, 28), Image.Resampling.NEAREST)
        image_buffer.close()
        
        img_array = np.array(image)

        # Inversion conservée pour les fichiers uploadés par l'utilisateur
        if img_array.mean() > 127:
            img_array = 255 - img_array
        
        img_array = img_array / 255.0
        
        # FIX: Création explicite du tenseur 4D (1, 1, 28, 28)
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
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
        raise HTTPException(status_code=400, detail=f"Erreur de traitement de l'image uploadée : {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)