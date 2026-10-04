import torch
import numpy as np
from src.model.checkpoint import load_checkpoint
from src.preprocessing.image_preprocessing import load_and_preprocess_image

def run_inference(image_path, model_path, device):
    model = load_checkpoint(model_path, device)
    _, tensor = load_and_preprocess_image(image_path)
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
        prob_map = torch.sigmoid(output).cpu().numpy().squeeze()
        
    return prob_map
