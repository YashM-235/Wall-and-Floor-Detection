import cv2
import torch
import numpy as np
from torchvision import transforms
from train_deeplab import model  # Import trained model

# Load model
model.load_state_dict(torch.load('src/models/deeplabv3_wall_floor.pth',map_location='cpu'))
model = model.eval()

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def process_frame(frame):
    # Preprocess
    input_tensor = transform(frame).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = torch.argmax(output, dim=0).cpu().numpy()
    print("Unique values in predicted mask:", np.unique(mask))
    # Create overlay
    overlay = frame.copy()
    overlay[mask == 0] = [0, 0, 255]  # Red for walls
    overlay[mask == 1] = [0, 255, 0]   # Green for floors
    print("Unique values in predicted mask:", np.unique(mask))
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

# Webcam loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = process_frame(frame)
    cv2.imshow('Wall/Floor Segmentation', processed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
