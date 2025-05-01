import streamlit as st
import cv2
import numpy as np
import torch
from brick_overlay import overlay_bricks
from train_deeplab import model
import torchvision.transforms as T

# Load trained model
model.load_state_dict(torch.load('src/models/deeplabv3_wall_floor_final_small.pth'))
model.eval()

# Preprocessing transforms
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
               std=[0.229, 0.224, 0.225])
])

def process_image(image):
    # Preprocess
    input_tensor = transform(image).unsqueeze(0)
    
    # Inference on CPU (no need for .cuda())
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Get the mask (argmax to get the class label)
    mask = torch.argmax(output, dim=0).cpu().numpy()

    # Create separate masks for walls and floors
    wall_mask = (mask == 0).astype(np.uint8)  # Wall (class 0)
    floor_mask = (mask == 1).astype(np.uint8)  # Floor (class 1)

    # Call overlay_bricks with both wall_mask and floor_mask
    return overlay_bricks(image, wall_mask, floor_mask)

def overlay_bricks(image, wall_mask, floor_mask):
    # Create overlay for walls
    overlay = image.copy()
    overlay[wall_mask == 1] = [0, 0, 255]  # Red for walls
    overlay[floor_mask == 1] = [0, 255, 0]  # Green for floors

    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

st.title('Wall/Floor Detection & Brick Overlay')
st.sidebar.title('Options')

input_option = st.sidebar.selectbox("Choose Input Source", ("Upload Image", "Live Camera"))

if input_option == "Upload Image":
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process and display
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        processed = process_image(image)
        with col2:
            st.image(processed, caption="Processed Image", use_container_width=True)

elif input_option == "Live Camera":
    st.write("Live camera feed - press 'q' to exit")
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        # Read and process frame
        bytes_data = img_file_buffer.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        processed = process_image(image)
        st.image(processed, caption="Processed Frame", use_container_width=True)
