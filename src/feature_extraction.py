import torch
import torchvision.models as models
import torchvision.transforms as T
import cv2
from torchvision.models import MobileNet_V2_Weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_deep_features(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = mobilenet(image)
    return features.view(-1).cpu().numpy()
