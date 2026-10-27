import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from sklearn.decomposition import PCA, FactorAnalysis

# Set seed for reproducible excellence
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes_path = os.path.join(BASE_DIR, 'classes.npy')
model_path = os.path.join(BASE_DIR, 'best_asl_model.pth')
pca_path = os.path.join(BASE_DIR, 'pca_model.pkl')

# --- PCA CONFIG ---
N_PCA_COMPONENTS = 15  # Keep top 15 principal components out of 63

classes = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
])
np.save(classes_path, classes)
num_classes = len(classes)

class SignLanguageANN(nn.Module):
    """ANN classifier that takes PCA-reduced input (N_PCA_COMPONENTS dims) instead of raw 63-dim landmarks."""
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

def build_base_skeleton():
    """
    Build anatomically calibrated open hand (21 landmarks) normalized to wrist at (0,0,0).
    MediaPipe coordinate convention:
    - X: Horizontal (- is thumb side for right hand, + is pinky side)
    - Y: Vertical (- is upward toward fingertips, + is downward toward forearm)
    - Z: Depth (- is closer to camera / palm front, + is away / dorsal back)
    """
    h = np.zeros((21, 3), dtype=np.float32)
    h[0] = [0.0, 0.0, 0.0] # Wrist

    # Thumb (1: CMC, 2: MCP, 3: IP, 4: TIP)
    h[1] = [-0.045, -0.045, -0.010]
    h[2] = [-0.080, -0.085, -0.018]
    h[3] = [-0.115, -0.125, -0.022]
    h[4] = [-0.145, -0.165, -0.025]

    # Index (5: MCP, 6: PIP, 7: DIP, 8: TIP)
    h[5] = [-0.042, -0.145, 0.000]
    h[6] = [-0.050, -0.215, 0.000]
    h[7] = [-0.055, -0.270, 0.000]
    h[8] = [-0.060, -0.325, 0.000]

    # Middle (9: MCP, 10: PIP, 11: DIP, 12: TIP)
    h[9]  = [0.000, -0.155, 0.000]
    h[10] = [0.000, -0.235, 0.000]
    h[11] = [0.000, -0.295, 0.000]
    h[12] = [0.000, -0.355, 0.000]

    # Ring (13: MCP, 14: PIP, 15: DIP, 16: TIP)
    h[13] = [0.042, -0.145, 0.000]
    h[14] = [0.050, -0.215, 0.000]
    h[15] = [0.055, -0.268, 0.000]
    h[16] = [0.060, -0.315, 0.000]

    # Pinky (17: MCP, 18: PIP, 19: DIP, 20: TIP)
    h[17] = [0.075, -0.125, 0.000]
    h[18] = [0.090, -0.178, 0.000]
    h[19] = [0.100, -0.220, 0.000]
    h[20] = [0.110, -0.258, 0.000]
    return h

def curl_finger(h, start_idx, fold_factor=1.0, lateral_shift=0.0):
    """Curl finger down tightly toward palm."""
    mcp = h[start_idx]
    h[start_idx + 1] = mcp + np.array([lateral_shift * 0.25, 0.038 * fold_factor, -0.032 * fold_factor], dtype=np.float32)
    h[start_idx + 2] = mcp + np.array([lateral_shift * 0.50, 0.075 * fold_factor, -0.055 * fold_factor], dtype=np.float32)
    h[start_idx + 3] = mcp + np.array([lateral_shift * 0.70, 0.105 * fold_factor, -0.045 * fold_factor], dtype=np.float32)

def set_finger_hook(h, start_idx):
    """Bend finger into curved hook/crook (used in 'X')."""
    mcp = h[start_idx]
    h[start_idx + 1] = mcp + np.array([0.0, -0.065, -0.015], dtype=np.float32)
    h[start_idx + 2] = mcp + np.array([0.0, -0.065, -0.055], dtype=np.float32)
    h[start_idx + 3] = mcp + np.array([0.0, -0.030, -0.065], dtype=np.float32)

def generate_canonical_asl(label):
    """
    Generate accurate 3D landmark coordinates matching the standard American Sign Language
    alphabet chart (letters a-z, del, nothing, space).
    """
    h = build_base_skeleton()
    l = label.upper() if label not in ['del', 'nothing', 'space'] else label

    if l == 'A':
        # [Chart 'a']: Tight fist facing front; thumb straight up resting vertically against the index finger side
        curl_finger(h, 5, 1.25)
        curl_finger(h, 9, 1.25)
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        # Thumb points straight up along index side:
        h[1] = [-0.045, -0.050, 0.005]
        h[2] = [-0.065, -0.100, 0.012]
        h[3] = [-0.070, -0.150, 0.018]
        h[4] = [-0.072, -0.200, 0.020]

    elif l == 'B':
        # [Chart 'b']: Four fingers held together pointing straight up; thumb folded horizontally across lower palm
        h[1] = [-0.035, -0.040, 0.015]
        h[2] = [-0.015, -0.070, 0.025]
        h[3] = [0.010, -0.090, 0.028]
        h[4] = [0.028, -0.100, 0.030]

    elif l == 'C':
        # [Chart 'c']: Hand forms a curved 'C' shape; fingers arched above, thumb arched below
        for f in [5, 9, 13, 17]:
            mcp = h[f]
            h[f+1] = mcp + np.array([0.0, -0.055, -0.045])
            h[f+2] = mcp + np.array([0.0, -0.085, -0.085])
            h[f+3] = mcp + np.array([0.0, -0.105, -0.065])
        # Thumb curved below:
        h[1] = [-0.045, -0.045, -0.020]
        h[2] = [-0.075, -0.075, -0.050]
        h[3] = [-0.085, -0.110, -0.075]
        h[4] = [-0.080, -0.135, -0.065]

    elif l == 'D':
        # [Chart 'd']: Index finger pointing straight up; thumb, middle, ring, pinky tips touching in a round circle
        curl_finger(h, 9, 1.05)
        curl_finger(h, 13, 1.05)
        curl_finger(h, 17, 1.05)
        # Thumb touches middle/ring fingertips:
        h[1] = [-0.040, -0.045, -0.015]
        h[2] = [-0.050, -0.080, -0.035]
        h[3] = [-0.035, -0.110, -0.045]
        h[4] = [-0.015, -0.125, -0.040]
        h[12] = [-0.015, -0.125, -0.040]

    elif l == 'E':
        # [Chart 'e']: All 4 fingers curled down tightly at knuckles; thumb tucked tight underneath the fingertips
        for f in [5, 9, 13, 17]:
            curl_finger(h, f, 1.35)
        # Thumb tucked under knuckles:
        h[1] = [-0.035, -0.040, 0.010]
        h[2] = [-0.015, -0.065, 0.020]
        h[3] = [0.005, -0.075, 0.022]
        h[4] = [0.025, -0.075, 0.022]

    elif l == 'F':
        # [Chart 'f']: Index tip and thumb tip touch in 'O' circle; middle, ring, pinky straight up and slightly spread
        h[4] = [-0.035, -0.145, -0.035]
        h[8] = [-0.035, -0.145, -0.035]
        # Middle, ring, pinky spread upwards:
        h[12] = [-0.010, -0.355, 0.0]
        h[16] = [0.065, -0.320, 0.0]
        h[20] = [0.125, -0.265, 0.0]

    elif l == 'G':
        # [Chart 'g']: Hand turned sideways; index finger pointing straight out horizontally, thumb pointing parallel
        curl_finger(h, 9, 1.25)
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        # Index pointing horizontally:
        h[5] = [-0.030, -0.120, 0.0]
        h[6] = [0.040, -0.120, 0.0]
        h[7] = [0.110, -0.120, 0.0]
        h[8] = [0.180, -0.120, 0.0]
        # Thumb parallel horizontally:
        h[1] = [-0.030, -0.070, 0.0]
        h[2] = [0.020, -0.075, 0.0]
        h[3] = [0.070, -0.075, 0.0]
        h[4] = [0.120, -0.075, 0.0]

    elif l == 'H':
        # [Chart 'h']: Index AND middle fingers extended together horizontally; ring and pinky curled down
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        # Index horizontal:
        h[5] = [-0.030, -0.130, 0.0]
        h[6] = [0.040, -0.130, 0.0]
        h[7] = [0.115, -0.130, 0.0]
        h[8] = [0.190, -0.130, 0.0]
        # Middle horizontal right beside index:
        h[9]  = [-0.030, -0.095, 0.0]
        h[10] = [0.040, -0.095, 0.0]
        h[11] = [0.115, -0.095, 0.0]
        h[12] = [0.190, -0.095, 0.0]
        # Thumb folded across ring/pinky:
        h[4] = [-0.010, -0.065, 0.025]

    elif l == 'I':
        # [Chart 'i']: Pinky finger pointing straight up; other three fingers folded in a fist with thumb over them
        curl_finger(h, 5, 1.25)
        curl_finger(h, 9, 1.25)
        curl_finger(h, 13, 1.25)
        h[4] = [-0.015, -0.085, 0.025] # Thumb across knuckles
        # Pinky straight up:
        h[17] = [0.075, -0.125, 0.0]
        h[18] = [0.085, -0.180, 0.0]
        h[19] = [0.095, -0.225, 0.0]
        h[20] = [0.100, -0.265, 0.0]

    elif l == 'J':
        # [Chart 'j']: Pinky extended, sweeping in a 'J' curve down and inwards
        curl_finger(h, 5, 1.25)
        curl_finger(h, 9, 1.25)
        curl_finger(h, 13, 1.25)
        h[4] = [-0.015, -0.085, 0.025]
        # Curved sweeping pinky:
        h[17] = [0.075, -0.125, 0.0]
        h[18] = [0.090, -0.170, 0.03]
        h[19] = [0.070, -0.190, 0.07]
        h[20] = [0.035, -0.180, 0.09]

    elif l == 'K':
        # [Chart 'k']: Index pointing straight up; middle finger angled forward; thumb pressed between them pointing up
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        # Index straight up:
        h[8] = [-0.055, -0.325, 0.0]
        # Middle angled forward (Z offset):
        h[10] = [0.010, -0.220, -0.035]
        h[11] = [0.020, -0.270, -0.070]
        h[12] = [0.025, -0.310, -0.095]
        # Thumb between index and middle:
        h[4] = [-0.025, -0.190, -0.040]

    elif l == 'L':
        # [Chart 'l']: Thumb pointing horizontally and index finger pointing straight up, forming 90-degree 'L'
        curl_finger(h, 9, 1.25)
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        # Index straight up:
        h[8] = [-0.055, -0.340, 0.0]
        # Thumb fully extended horizontally at 90 deg:
        h[1] = [-0.050, -0.030, 0.0]
        h[2] = [-0.105, -0.035, 0.0]
        h[3] = [-0.160, -0.038, 0.0]
        h[4] = [-0.215, -0.040, 0.0]

    elif l == 'M':
        # [Chart 'm']: Three fingers (index, middle, ring) folded over thumb; thumb tip peeks between ring and pinky
        curl_finger(h, 5, 1.10)
        curl_finger(h, 9, 1.10)
        curl_finger(h, 13, 1.10)
        curl_finger(h, 17, 1.30)
        # Thumb reaches under index, middle, ring:
        h[4] = [0.045, -0.095, -0.015]

    elif l == 'N':
        # [Chart 'n']: Two fingers (index, middle) folded over thumb; thumb tip peeks between middle and ring
        curl_finger(h, 5, 1.10)
        curl_finger(h, 9, 1.10)
        curl_finger(h, 13, 1.30)
        curl_finger(h, 17, 1.30)
        # Thumb reaches under index and middle:
        h[4] = [0.015, -0.095, -0.015]

    elif l == 'O':
        # [Chart 'o']: All four fingertips and thumb tip touching to form a circular 'O'
        for f in [8, 12, 16, 20]:
            h[f] = [-0.025, -0.155, -0.055]
        h[4] = [-0.025, -0.155, -0.055]

    elif l == 'P':
        # [Chart 'p']: Downward angled 'K'; index pointing forward-down, middle straight down, thumb between them
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        # Index forward-down:
        h[6] = [-0.030, -0.080, 0.100]
        h[7] = [-0.040, -0.025, 0.165]
        h[8] = [-0.045, 0.030, 0.220]
        # Middle pointing straight down:
        h[10] = [0.010, -0.080, 0.060]
        h[11] = [0.015, -0.020, 0.110]
        h[12] = [0.020, 0.040, 0.150]
        # Thumb between them:
        h[4] = [-0.015, -0.040, 0.090]

    elif l == 'Q':
        # [Chart 'q']: Downward angled 'G'; index and thumb pointing down like downward pinch
        curl_finger(h, 9, 1.25)
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        # Index pointing down:
        h[6] = [-0.030, -0.060, 0.080]
        h[7] = [-0.035, 0.010, 0.140]
        h[8] = [-0.040, 0.080, 0.190]
        # Thumb pointing down parallel:
        h[2] = [-0.075, -0.040, 0.060]
        h[3] = [-0.085, 0.020, 0.110]
        h[4] = [-0.090, 0.075, 0.150]

    elif l == 'R':
        # [Chart 'r']: Index and middle fingers extended straight up and crossed over one another
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        # Index crosses in front toward right (+X):
        h[6] = [-0.020, -0.215, -0.015]
        h[7] = [0.005, -0.270, -0.020]
        h[8] = [0.025, -0.325, -0.020]
        # Middle crosses behind toward left (-X):
        h[10] = [-0.010, -0.235, 0.015]
        h[11] = [-0.025, -0.295, 0.020]
        h[12] = [-0.035, -0.355, 0.020]
        h[4] = [-0.025, -0.085, 0.025]

    elif l == 'S':
        # [Chart 's']: Fist with thumb wrapped firmly across the front of all folded fingers
        curl_finger(h, 5, 1.30)
        curl_finger(h, 9, 1.30)
        curl_finger(h, 13, 1.30)
        curl_finger(h, 17, 1.30)
        # Thumb crosses the front of knuckles horizontally:
        h[1] = [-0.035, -0.045, -0.020]
        h[2] = [-0.010, -0.080, -0.045]
        h[3] = [0.020, -0.100, -0.055]
        h[4] = [0.045, -0.105, -0.050]

    elif l == 'T':
        # [Chart 't']: Thumb tucked between index and middle; index finger curled over thumb tip
        curl_finger(h, 5, 1.15)
        curl_finger(h, 9, 1.30)
        curl_finger(h, 13, 1.30)
        curl_finger(h, 17, 1.30)
        # Thumb pokes out between index and middle:
        h[4] = [-0.015, -0.135, -0.020]

    elif l == 'U':
        # [Chart 'u']: Index and middle fingers extended straight up held together (touching, no gap)
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        h[8] = [-0.020, -0.335, 0.0]
        h[12] = [0.010, -0.335, 0.0]
        h[4] = [-0.025, -0.085, 0.025]

    elif l == 'V':
        # [Chart 'v']: Peace sign / V: index and middle extended straight up spread apart in a clear 'V'
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        h[8] = [-0.105, -0.325, 0.0]
        h[12] = [0.085, -0.325, 0.0]
        h[4] = [-0.025, -0.085, 0.025]

    elif l == 'W':
        # [Chart 'w']: Three fingers (index, middle, ring) extended straight up and spread apart in a 'W'
        curl_finger(h, 17, 1.25)
        h[8]  = [-0.105, -0.315, 0.0]
        h[12] = [0.000, -0.350, 0.0]
        h[16] = [0.105, -0.315, 0.0]
        h[4] = [-0.025, -0.085, 0.025]

    elif l == 'X':
        # [Chart 'x']: Index finger bent into a hook/crook; middle, ring, pinky curled in palm
        set_finger_hook(h, 5)
        curl_finger(h, 9, 1.25)
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        h[4] = [-0.025, -0.085, 0.025]

    elif l == 'Y':
        # [Chart 'y']: Shaka / hang loose: thumb and pinky extended fully outward in opposite directions; 3 middle curled
        curl_finger(h, 5, 1.25)
        curl_finger(h, 9, 1.25)
        curl_finger(h, 13, 1.25)
        # Thumb extended outward-up:
        h[1] = [-0.055, -0.040, 0.0]
        h[2] = [-0.115, -0.065, 0.0]
        h[3] = [-0.170, -0.090, 0.0]
        h[4] = [-0.225, -0.115, 0.0]
        # Pinky extended outward-up:
        h[17] = [0.075, -0.125, 0.0]
        h[18] = [0.120, -0.170, 0.0]
        h[19] = [0.160, -0.210, 0.0]
        h[20] = [0.200, -0.245, 0.0]

    elif l == 'Z':
        # [Chart 'z']: Index finger extended pointing forward ready to trace a 'Z'; others curled
        curl_finger(h, 9, 1.25)
        curl_finger(h, 13, 1.25)
        curl_finger(h, 17, 1.25)
        h[6] = [-0.035, -0.190, -0.040]
        h[7] = [-0.040, -0.235, -0.080]
        h[8] = [-0.045, -0.275, -0.120]
        h[4] = [-0.025, -0.085, 0.025]

    elif l == 'del':
        # Open hand tilted down
        h = h * 0.75
        h[:, 0] -= 0.12
        h[:, 1] += 0.05

    elif l == 'space':
        # Flat open hand horizontal
        h[:, 1] *= 0.4
        h[:, 0] += 0.08

    elif l == 'nothing':
        # Relaxed limp resting hand
        h = h * 0.45
        h[:, 1] += 0.18

    # Strict wrist anchor at (0, 0, 0)
    h[0] = [0.0, 0.0, 0.0]
    return h.flatten()

def rotate_landmarks(landmarks_63, angle_deg_x, angle_deg_y, angle_deg_z):
    """Apply realistic 3D Euler rotation jitter to simulate varying hand angles."""
    pts = landmarks_63.reshape((21, 3))
    rx = np.radians(angle_deg_x)
    ry = np.radians(angle_deg_y)
    rz = np.radians(angle_deg_z)

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    rot_pts = (pts @ R.T).astype(np.float32)
    # Re-anchor wrist strictly at (0,0,0)
    rot_pts = rot_pts - rot_pts[0]
    return rot_pts.flatten()

def train_and_save():
    print(f"Generating high-precision ASL dataset calibrated to visual chart for {num_classes} classes...")
    X_list = []
    y_list = []

    # 500 diverse augmented samples per class = 14,500 total samples
    SAMPLES_PER_CLASS = 500
    for idx, c in enumerate(classes):
        base_vec = generate_canonical_asl(c)
        for _ in range(SAMPLES_PER_CLASS):
            # 3D rotation jitter simulating user camera angles (+/- 9 degrees)
            ang_x = np.random.uniform(-9.0, 9.0)
            ang_y = np.random.uniform(-9.0, 9.0)
            ang_z = np.random.uniform(-9.0, 9.0)
            rot_vec = rotate_landmarks(base_vec, ang_x, ang_y, ang_z)

            # Physiological scaling (different hand dimensions)
            scale = np.random.uniform(0.90, 1.10)
            noise = np.random.normal(0, 0.007, size=63).astype(np.float32)
            aug_vec = (rot_vec * scale + noise).astype(np.float32)

            # Strict wrist anchor at (0, 0, 0)
            aug_vec[0] = 0.0
            aug_vec[1] = 0.0
            aug_vec[2] = 0.0

            X_list.append(aug_vec)
            y_list.append(idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle dataset
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # =====================================================
    # PCA: Fit on training data, reduce 63 → N_PCA_COMPONENTS
    # =====================================================
    print(f"\n--- PCA Dimensionality Reduction ---")
    print(f"Original feature dimensionality: {X.shape[1]}")

    pca = PCA(n_components=N_PCA_COMPONENTS)
    pca.fit(X)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    print(f"PCA components kept: {N_PCA_COMPONENTS}")
    print(f"Variance explained per component: {np.round(explained_var * 100, 2)}")
    print(f"Cumulative variance explained: {cumulative_var[-1]*100:.2f}%")

    # Save PCA model for use during inference
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA model saved to: {pca_path}")

    # --- Factor Analysis comparison (for viva discussion) ---
    print(f"\n--- Factor Analysis Comparison (for reference) ---")
    fa = FactorAnalysis(n_components=N_PCA_COMPONENTS, random_state=42)
    fa.fit(X)
    fa_score = fa.score(X)
    print(f"Factor Analysis log-likelihood on training data: {fa_score:.4f}")
    print(f"(FA models latent factors + noise; PCA finds max-variance directions)")
    print(f"For this low-dim (63) problem, PCA is sufficient. FA adds noise modeling.\n")

    # Transform data to PCA space
    X_pca = pca.transform(X).astype(np.float32)
    print(f"Reduced feature dimensionality: {X_pca.shape[1]}")

    split = int(0.85 * len(X_pca))
    X_train, X_val = torch.tensor(X_pca[:split]), torch.tensor(X_pca[split:])
    y_train, y_val = torch.tensor(y[:split]), torch.tensor(y[split:])

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = SignLanguageANN(N_PCA_COMPONENTS, num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    print("Training Enhanced SignLanguageANN on PCA-reduced ASL Chart Dataset...")
    best_acc = 0.0
    for epoch in range(40):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                pred = torch.argmax(out, dim=1)
                correct += (pred == by).sum().item()
                total += by.size(0)

        acc = (correct / total) * 100
        if (epoch + 1) % 5 == 0 or acc > best_acc:
            print(f"Epoch {epoch+1:02d}/40 | Val Accuracy: {acc:.2f}% | Best: {max(acc, best_acc):.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)

    print("\n==========================================")
    print(f"Training Complete! Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"PCA model saved to: {pca_path}")
    print(f"Input dim: {N_PCA_COMPONENTS} (was 63)")
    print("==========================================")

if __name__ == "__main__":
    train_and_save()
