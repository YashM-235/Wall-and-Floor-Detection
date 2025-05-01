import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class WallFloorSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        
        # Verify dataset integrity
        assert len(os.listdir(mask_dir)) == len(self.images), \
            f"Mismatch: {len(self.images)} images vs {len(os.listdir(mask_dir))} masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
        
        # Load image and mask
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)  # Load as grayscale
        
        # Convert mask labels: 1→0 (wall), 4→1 (floor)
        mask = np.where(mask == 1, 0, np.where(mask == 4, 1, 255)).astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        return image, torch.tensor(mask, dtype=torch.long)
