import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enhanced augmentation pipeline with mask preservation
aug_pipeline = A.Compose([
    A.Resize(512, 512),  # Fixed size for consistency
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.Affine(
        translate_percent=(-0.1, 0.1),  # Increased translation range
        scale=(0.8, 1.2),  # More aggressive scaling
        rotate=(-45, 45),
        shear=(-15, 15),  # Added shear for better generalization
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        p=0.7
    ),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.2, 0.3), 
        contrast_limit=(-0.2, 0.3), 
        p=0.5
    ),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
], additional_targets={'mask': 'mask'})

def process_split(img_dir, mask_dir, output_dir, num_augments):
    """Process a single data split (training/validation)"""
    # Create output directories
    aug_img_dir = os.path.join(output_dir, 'images', os.path.basename(img_dir))
    aug_mask_dir = os.path.join(output_dir, 'masks', os.path.basename(mask_dir))
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)

    # Get sorted list of images and verify pairing
    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    mask_files = sorted([f.replace('.jpg', '.png') for f in image_files])
    
    # Verify mask existence
    for img, mask in zip(image_files, mask_files):
        if not os.path.exists(os.path.join(mask_dir, mask)):
            logging.error(f"Mask missing for {img}")
            raise FileNotFoundError(f"Mask missing: {mask}")

    # Process each image-mask pair
    for img_name, mask_name in tqdm(zip(image_files, mask_files), 
                                  total=len(image_files), 
                                  desc=f"Processing {os.path.basename(img_dir)}"):
        try:
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)

            # Load image and mask with validation
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                raise ValueError(f"Failed to load {img_path} or {mask_path}")

            # Save original copies first
            base_name = os.path.splitext(img_name)[0]
            cv2.imwrite(os.path.join(aug_img_dir, f"{base_name}_orig.jpg"),
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(aug_mask_dir, f"{base_name}_orig.png"), mask)

            # Generate augmented versions
            for i in range(num_augments):
                augmented = aug_pipeline(image=image, mask=mask)
                aug_img = augmented['image']
                aug_mask = augmented['mask']

                cv2.imwrite(os.path.join(aug_img_dir, f"{base_name}_aug{i}.jpg"),
                           cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(aug_mask_dir, f"{base_name}_aug{i}.png"), aug_mask)

        except Exception as e:
            logging.error(f"Error processing {img_name}: {str(e)}")
            continue

def augment_dataset(dataset_root, output_root, num_augments=5):
    """
    Main augmentation function that handles the entire dataset structure
    """
    # Validate input structure
    required_dirs = ['images/training', 'images/validation',
                    'masks/training', 'masks/validation']
    for d in required_dirs:
        if not os.path.exists(os.path.join(dataset_root, d)):
            raise FileNotFoundError(f"Missing directory: {d}")

    # Process training and validation splits
    for split in ['training', 'validation']:
        img_split_dir = os.path.join(dataset_root, 'images', split)
        mask_split_dir = os.path.join(dataset_root, 'masks', split)
        
        if not os.path.isdir(img_split_dir) or not os.path.isdir(mask_split_dir):
            continue
            
        process_split(
            img_dir=img_split_dir,
            mask_dir=mask_split_dir,
            output_dir=output_root,
            num_augments=num_augments
        )

    logging.info("Augmentation completed successfully!")

# Example usage
if __name__ == "__main__":
    augment_dataset(
        dataset_root="dataset",
        output_root="dataset/augmented",
        num_augments=5
    )
