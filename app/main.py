from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
import torch
import torch.nn as nn
import numpy as np
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. MODEL ARCHITECTURE ---
class SignLanguageANN(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageANN, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)
            
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 2. LOAD ASSETS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes = np.load(os.path.join(BASE_DIR, 'classes.npy'), allow_pickle=True)
num_classes = len(classes)

model = SignLanguageANN(num_classes)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'best_asl_model.pth'), map_location='cpu'))
model.eval()

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "ok", "classes": [str(c) for c in classes], "num_classes": num_classes}


# --- 3. DATA SCHEMA ---
class LandmarkInput(BaseModel):
    landmarks: list[float] 

# --- 4. PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict_asl(data: Union[LandmarkInput, list[float]]):
    landmarks = data.landmarks if isinstance(data, LandmarkInput) else data

    if len(landmarks) != 63:
        raise HTTPException(status_code=400, detail="Expected 63 landmarks (21 points * 3 coords)")

    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
    normalized_landmarks = []
    
    for i in range(0, 63, 3):
        normalized_landmarks.extend([
            landmarks[i] - wrist_x,
            landmarks[i+1] - wrist_y,
            landmarks[i+2] - wrist_z
        ])

    input_tensor = torch.FloatTensor(normalized_landmarks).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        label = classes[predicted_idx.item()]
        score = float(confidence.item())

    return {
        "letter": label,
        "confidence": round(score, 4),
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)