import os
import torch
from src.model.model import DIF_SEG

def load_checkpoint(filepath, device):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}\nReason: Trained DIF-SEG checkpoint is unavailable.\nFix: Please train the model using 'python train.py' or download the pretrained weights to '{filepath}'.")
    
    checkpoint = torch.load(filepath, map_location=device)
    model = DIF_SEG(in_channels=3, out_channels=1)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    # Handle wrapping mismatch for the fallback model
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("model."):
            new_state_dict[f"model.{k}"] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model
