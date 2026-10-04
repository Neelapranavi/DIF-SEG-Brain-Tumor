import torch
import yaml
from src.model.model import DIF_SEG

def check():
    print("Checking checkpoint...")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    checkpoint_path = config["model"]["checkpoint"]
    
    import os
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint missing at {checkpoint_path}")
        print("Reason: Trained DIF-SEG checkpoint is unavailable.")
        print("Fix: Please train the model using 'python train.py' or download the pretrained weights.")
        return
        
    print("Checkpoint found.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Loading model...")
    model = DIF_SEG(in_channels=1, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully.")
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
    print("Running dummy inference...")
    dummy_input = torch.randn(1, 1, 256, 256)
    print(f"Input shape: {dummy_input.shape}")
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    check()
