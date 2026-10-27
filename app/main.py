"""
main.py — FastAPI backend for real-time ASL sign language recognition.

Pipeline per frame:
  1. Receive 63 raw landmark floats from the frontend
  2. Kalman-filter each coordinate (constant-velocity model) to reduce jitter
  3. Wrist-normalize (subtract wrist coords so wrist = origin)
  4. PCA-project from 63-dim → N_PCA_COMPONENTS-dim
  5. ANN inference → 29-class softmax probabilities
  6. HMM Viterbi decoding on sliding window → temporally smoothed prediction

Covers: Kalman filtering, PCA, HMM + Viterbi, FastAPI serving.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

from hmm import ASLHiddenMarkovModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# 1. MODEL ARCHITECTURE (must match train_model.py)
# =====================================================================
class SignLanguageANN(nn.Module):
    """ANN classifier operating on PCA-reduced input."""
    def __init__(self, input_dim, num_classes):
        super(SignLanguageANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =====================================================================
# 2. KALMAN FILTER — constant-velocity model per coordinate
# =====================================================================
class LandmarkKalmanFilter:
    """
    Constant-velocity Kalman filter for 63 landmark coordinates.
    
    State per coordinate: [position, velocity] (2-dim).
    Observation: position only (1-dim).
    
    This smooths MediaPipe jitter before classification, reducing
    misclassifications near letter boundaries.
    
    Topics covered: linear dynamical systems, Kalman filtering.
    """

    def __init__(self, n_coords: int = 63, process_noise: float = 1e-4, measurement_noise: float = 5e-3):
        self.n = n_coords
        dt = 1.0  # one frame

        # State transition: x_{t+1} = F * x_t
        self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        # Observation matrix: z_t = H * x_t
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        # Process noise covariance
        self.Q = np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ], dtype=np.float64) * process_noise
        # Measurement noise covariance
        self.R = np.array([[measurement_noise]], dtype=np.float64)

        # Per-coordinate state: [position, velocity]
        self.x = np.zeros((n_coords, 2), dtype=np.float64)
        # Per-coordinate covariance
        self.P = np.tile(np.eye(2, dtype=np.float64) * 1.0, (n_coords, 1, 1))

        self._initialized = False

    def reset(self):
        """Reset filter state."""
        self.x[:] = 0.0
        self.P[:] = np.eye(2, dtype=np.float64) * 1.0
        self._initialized = False

    def filter(self, z: np.ndarray) -> np.ndarray:
        """
        Process one frame of 63 observed coordinates.
        Returns the filtered (smoothed) 63-dim vector.
        """
        z = z.astype(np.float64)

        if not self._initialized:
            # Initialize state to first observation
            self.x[:, 0] = z
            self.x[:, 1] = 0.0
            self._initialized = True
            return z.copy()

        filtered = np.empty(self.n, dtype=np.float64)

        for i in range(self.n):
            # --- Predict ---
            x_pred = self.F @ self.x[i]
            P_pred = self.F @ self.P[i] @ self.F.T + self.Q

            # --- Update ---
            innovation = z[i] - (self.H @ x_pred)[0]
            S = (self.H @ P_pred @ self.H.T + self.R)[0, 0]
            K = (P_pred @ self.H.T) / S  # Kalman gain (2×1)

            self.x[i] = x_pred + K.flatten() * innovation
            self.P[i] = (np.eye(2) - K @ self.H) @ P_pred

            filtered[i] = self.x[i, 0]

        return filtered.astype(np.float32)


# =====================================================================
# 3. LOAD ASSETS (model, classes, PCA)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes = np.load(os.path.join(BASE_DIR, 'classes.npy'), allow_pickle=True)
num_classes = len(classes)

# Load PCA model
pca_path = os.path.join(BASE_DIR, 'pca_model.pkl')
with open(pca_path, 'rb') as f:
    pca_model = pickle.load(f)
n_pca_components = pca_model.n_components_

# Load ANN model (input dim = PCA components)
model = SignLanguageANN(n_pca_components, num_classes)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'best_asl_model.pth'), map_location='cpu'))
model.eval()

# Initialize Kalman filter and HMM decoder
kalman = LandmarkKalmanFilter(n_coords=63)
hmm_decoder = ASLHiddenMarkovModel(
    num_classes=num_classes,
    self_transition_prob=0.85,
    window_size=10,
)

print(f"[main] Loaded model: input_dim={n_pca_components}, classes={num_classes}")
print(f"[main] Kalman filter: 63 coords, constant-velocity model")
print(f"[main] HMM Viterbi: window=10, self_transition=0.85")


# =====================================================================
# 4. ENDPOINTS
# =====================================================================
@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "classes": [str(c) for c in classes],
        "num_classes": num_classes,
        "pca_components": n_pca_components,
        "hmm_window": hmm_decoder.window_size,
        "features": ["pca", "kalman_filter", "hmm_viterbi"],
    }


class LandmarkInput(BaseModel):
    landmarks: list[float]


@app.post("/predict")
async def predict_asl(data: Union[LandmarkInput, list[float]]):
    """
    Full inference pipeline:
      raw landmarks → Kalman → normalize → PCA → ANN → HMM Viterbi → result
    """
    landmarks = data.landmarks if isinstance(data, LandmarkInput) else data

    if len(landmarks) != 63:
        raise HTTPException(status_code=400, detail="Expected 63 landmarks (21 points * 3 coords)")

    raw = np.array(landmarks, dtype=np.float32)

    # --- Step 1: Kalman filter (smooth jitter) ---
    smoothed = kalman.filter(raw)

    # --- Step 2: Wrist-normalize ---
    wrist = smoothed[:3].copy()
    normalized = np.empty(63, dtype=np.float32)
    for i in range(0, 63, 3):
        normalized[i]     = smoothed[i]     - wrist[0]
        normalized[i + 1] = smoothed[i + 1] - wrist[1]
        normalized[i + 2] = smoothed[i + 2] - wrist[2]

    # --- Step 3: PCA projection (63 → n_pca_components) ---
    pca_features = pca_model.transform(normalized.reshape(1, -1)).astype(np.float32)

    # --- Step 4: ANN inference → softmax probabilities ---
    input_tensor = torch.FloatTensor(pca_features)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]

    # Raw ANN prediction (before HMM)
    raw_idx = int(np.argmax(probabilities))
    raw_label = str(classes[raw_idx])
    raw_conf = float(probabilities[raw_idx])

    # --- Step 5: HMM Viterbi decoding (temporal smoothing) ---
    hmm_state = hmm_decoder.observe(probabilities)
    hmm_label = str(classes[hmm_state])
    hmm_conf = float(probabilities[hmm_state])

    return {
        "letter": hmm_label,
        "confidence": round(hmm_conf, 4),
        "raw_letter": raw_label,
        "raw_confidence": round(raw_conf, 4),
        "hmm_buffer": hmm_decoder.buffer_length,
        "status": "success",
    }


@app.post("/reset")
async def reset_state():
    """Reset Kalman filter and HMM decoder state (e.g., when user clears sentence)."""
    kalman.reset()
    hmm_decoder.reset()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)