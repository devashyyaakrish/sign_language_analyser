import os
import glob
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from main import model, classes

def run_evaluation():
    # Setup MediaPipe Tasks Vision HandLandmarker
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)

    test_dir = os.path.join(os.path.dirname(__file__), "..", "data", "asl_alphabet_test")
    print(f"Attempting to load images from: {test_dir}\n")
    
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg"))
    if not image_paths:
        print("No test images found! Check path.")
        return
        
    correct_count = 0
    total_count = len(image_paths)
    misclassifications = []
    
    formatted_classes = [str(c).upper().strip() for c in classes]

    for img_path in sorted(image_paths):
        filename = os.path.basename(img_path)
        truth_label = filename.split('_')[0].upper()
        
        # Load image via MediaPipe's recommended way
        try:
            mp_image = mp.Image.create_from_file(img_path)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            total_count -= 1
            continue
            
        result = detector.detect(mp_image)
        
        if not result.hand_landmarks:
            print(f"[{filename:<15}] ⚠️  No hand detected by MediaPipe")
            misclassifications.append((truth_label, "NO_HAND"))
            continue
            
        landmarks = []
        for lm in result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])
            
        if len(landmarks) != 63:
            print(f"[{filename:<15}] ⚠️  Unexpected landmark count: {len(landmarks)}")
            misclassifications.append((truth_label, "INVALID_LANDMARKS"))
            continue
            
        # Normalize coordinates relative to wrist index 0
        wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
        normalized_landmarks = []
        for i in range(0, 63, 3):
            normalized_landmarks.extend([
                landmarks[i] - wrist_x,
                landmarks[i+1] - wrist_y,
                landmarks[i+2] - wrist_z
            ])
            
        # Run inference
        input_tensor = torch.FloatTensor(normalized_landmarks).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_label = formatted_classes[predicted_idx.item()]
            
        # Match SPACE and NOTHING properly
        if truth_label == "NOTHING" and predicted_label == "NOTHING":
            pass # already matches

        if predicted_label == truth_label:
            correct_count += 1
            print(f"[{filename:<15}] ✔️  {truth_label}")
        else:
            print(f"[{filename:<15}] ❌  Expected: {truth_label}, Got: {predicted_label}")
            misclassifications.append((truth_label, predicted_label))
            
    print("\n" + "="*40)
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("EVALUATION RESULTS")
        print("="*40)
        print(f"Total Evaluated : {total_count}")
        print(f"Total Correct   : {correct_count}")
        print(f"Accuracy        : {accuracy:.2f}%")
        
        if misclassifications:
            print("\nMisclassified Items:")
            for truth, pred in misclassifications:
                print(f"  Truth: {truth} | Predicted: {pred}")

if __name__ == "__main__":
    run_evaluation()
