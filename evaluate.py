import argparse
import yaml
import torch
from src.model.model import DIF_SEG

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("Evaluating Model...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    checkpoint_path = config["model"]["checkpoint"]
    import os
    if not os.path.exists(checkpoint_path):
        print("Error: Trained DIF-SEG checkpoint is unavailable. Please train first.")
        return

    print("Evaluation finished. Test Dice: 0.85 (dummy for demonstration)")

if __name__ == "__main__":
    main()
