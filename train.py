import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import glob
from src.model.model import DIF_SEG
from src.preprocessing.image_preprocessing import load_and_preprocess_image

# Dummy Dataset for skeleton
class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    def __len__(self):
        return max(10, len(self.images))  # Return dummy length if empty for testing

    def __getitem__(self, idx):
        if len(self.images) == 0:
            # Synthetic data if no real data
            img = torch.randn(1, self.img_size, self.img_size)
            mask = torch.randint(0, 2, (1, self.img_size, self.img_size), dtype=torch.float32)
            return img, mask
        
        _, img_tensor = load_and_preprocess_image(self.images[idx], (self.img_size, self.img_size))
        _, mask_tensor = load_and_preprocess_image(self.masks[idx], (self.img_size, self.img_size))
        # Binarize mask
        mask_tensor = (mask_tensor > 0.5).float()
        
        return img_tensor.squeeze(0), mask_tensor.squeeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = DIF_SEG(in_channels=1, out_channels=1).to(device)
    
    # Simple BCE loss for skeleton
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    train_dataset = MRIDataset(
        config["dataset"]["train_dir"] + "/images",
        config["dataset"]["train_dir"] + "/masks",
        config["dataset"]["image_size"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    print("Starting training...")
    epochs = args.epochs if args.epochs else config["training"]["epochs"]
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = config["model"]["checkpoint"]
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
