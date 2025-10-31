import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import os
import time

class FireDetectionModel(nn.Module):
    def __init__(self):
        super(FireDetectionModel, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
            
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FireDetectionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def check_frame_validity(frame):
    if frame is None or frame.size == 0:
        return False
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    brightness = np.mean(gray)
    
    std_dev = np.std(gray)
    
    is_valid = (
        brightness > 15 and 
        std_dev > 8 and    
        brightness < 240    
    )
    
    return is_valid

def enhance_image(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl,a,b))
    
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    alpha = 1.2  
    beta = 5     
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    return enhanced

def preprocess_frame(frame):
    if not check_frame_validity(frame):
        return None
        
    enhanced_frame = enhance_image(frame)
    
    frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(frame_rgb)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_image).unsqueeze(0)

def add_alert_overlay(frame, text):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 2, 3)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    
    cv2.putText(frame, text, (text_x+2, text_y+2), font, 2, (0, 0, 0), 3)
    cv2.putText(frame, text, (text_x, text_y), font, 2, (255, 255, 255), 3)

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fire_detection_model.pth')
    if not os.path.exists(model_path):
        print("Error: Model file not found. Please train the model first using test_fire_detection.py")
        return

    model, device = load_model(model_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Press 'q' to quit the live detection")
    
    confidence_history = []
    history_size = 20 
    alert_active = False
    alert_start_time = 0
    
    min_confidence = 0.75 
    consecutive_detections = 0
    required_consecutive = 5
    detection_cooldown = 0
    invalid_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        if not check_frame_validity(frame):
            invalid_frame_count += 1
            if invalid_frame_count > 5:
                status_text = "Camera Error: Please check camera connection"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Fire Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
        else:
            invalid_frame_count = 0

        input_tensor = preprocess_frame(frame)
        if input_tensor is None:
            continue
        
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            confidence = output.item()
            
            confidence_history.append(confidence)
            if len(confidence_history) > history_size:
                confidence_history.pop(0)
            
            weights = np.linspace(0.2, 1.0, len(confidence_history)) 
            smoothed_confidence = np.average(confidence_history, weights=weights)
            
            if detection_cooldown > 0:
                detection_cooldown -= 1
            else:
                if smoothed_confidence > min_confidence:
                    consecutive_detections += 1
                else:
                    consecutive_detections = max(0, consecutive_detections - 1)
                
                if consecutive_detections == 0 or consecutive_detections == required_consecutive:
                    detection_cooldown = 8  
        
            prediction = consecutive_detections >= required_consecutive


        status_text = f"Fire Detection: {smoothed_confidence*100:.1f}% (Confidence: {consecutive_detections}/{required_consecutive})"
        color = (0, 0, 255) if prediction else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if prediction:
            if not alert_active:
                alert_active = True
                alert_start_time = time.time()
            add_alert_overlay(frame, "FIRE ALERT! FIRE DETECTED!")
        else:
            alert_active = False


        cv2.imshow('Fire Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 