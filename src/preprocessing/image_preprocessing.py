import cv2
import numpy as np
import torch
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    # 1. Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_path, Image.Image):
        img = np.array(image_path.convert('L'))
    else:
        img = image_path
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
        
    # 4. Resize image (bilinear for images)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 5. Normalize intensities
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # 6-7. Convert to tensor and add dimensions [B, C, H, W]
    tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
    
    return img_resized, tensor
