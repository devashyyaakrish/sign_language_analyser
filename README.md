# Sign Language Detector

A high-performance, real-time sign language recognition system. This project leverages computer vision to transform 21-point hand landmarks into classified gestures through a robust full-stack architecture.

## 📁 Project Architecture

The repository is structured to separate the core ML translation interface from the backend processing logic:

* **`/sign-language-translator`**: The **Frontend** implementation. Built with **TypeScript** and **React**, this module handles real-time camera streaming, landmark visualization, and the user interface.
* **`/app`**: The **Backend** implementation. This handles the server-side logic, API endpoints, and model orchestration required to process and store gesture data.
* **`sign_language_interpreter.ipynb`**: The research and development environment used for training, testing, and evaluating the classification model.
* **`asl_landmarks_modern.csv`**: The processed dataset containing flattened $(x, y)$ coordinate mappings for various signs used during model training.

## 🛠️ Technical Implementation

### 1. Spatial Feature Extraction
Instead of processing raw RGB pixel data—which is computationally expensive and sensitive to environmental noise—the system utilizes **MediaPipe** to extract 21 hand landmarks. Each landmark consists of $(x, y, z)$ coordinates.

**Key Advantages:**
* **Dimensionality Reduction:** Transforms millions of pixels into a streamlined vector of 63 values.
* **Environmental Robustness:** Provides consistent detection regardless of lighting conditions or background clutter.
* **Performance:** Optimized for low-latency inference (~30+ FPS) on standard hardware.

### 2. The Classification Pipeline
The model is trained on the coordinate relationships stored in `asl_landmarks_modern.csv`. 

* **Normalization:** Coordinates are normalized relative to the wrist landmark. This ensures the model remains invariant to the hand's distance from the camera or its position within the frame.
* **Inference:** A Deep Neural Network (DNN) processes the flattened landmark array to output a probability distribution across the target sign classes.

### 3. Full-Stack Integration
The **React/TypeScript** frontend manages the lifecycle of the camera feed and landmarks. It communicates with the **Backend (/app)** to handle data persistence, user sessions, or complex inference tasks that require server-side resources.

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Node.js & npm

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/iPrq/sign-language-detector.git](https://github.com/iPrq/sign-language-detector.git)
    cd sign-language-detector
    ```

2.  **Frontend Setup:**
    ```bash
    cd sign-language-translator
    npm install
    npm start
    ```

3.  **Backend Setup:**
    ```bash
    cd ../app
    # Install backend dependencies (Node.js/Express or Python/FastAPI as applicable)
    npm install 
    npm run dev
    ```

## 📈 Future Development
* **Temporal Analysis:** Moving from static landmark detection to **LSTM** or **GRU** layers to enable fluid, continuous sentence recognition.
* **Dataset Expansion:** Scaling `asl_landmarks_modern.csv` to include a wider array of complex gestures and conversational signs.
* **Edge Deployment:** Optimizing the model for TensorFlow.js for dedicated client-side performance.
