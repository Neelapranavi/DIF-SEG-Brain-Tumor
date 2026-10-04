import argparse
import os
import cv2
import yaml
import torch
import numpy as np
from src.model.inference import run_inference
from src.preprocessing.image_preprocessing import load_and_preprocess_image
from src.postprocessing.segmentation import (
    threshold_probability_map, get_bounding_box,
    calculate_tumor_area_pixels, calculate_tumor_percentage
)
from src.utils.visualization import create_overlay, draw_bounding_box

def main():
    parser = argparse.ArgumentParser(description="Run DIF-SEG Inference")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    model_path = config["model"]["checkpoint"]
    if not os.path.exists(model_path):
        print(f"Error: Trained DIF-SEG checkpoint is unavailable. Expected at: {model_path}")
        return

    # Load and preprocess
    try:
        orig_img, tensor = load_and_preprocess_image(args.image, (config["dataset"]["image_size"],) * 2)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Run inference
    prob_map = run_inference(args.image, model_path, device)
    
    # Postprocess
    binary_mask = threshold_probability_map(prob_map, config["model"]["threshold"])
    tumor_area = calculate_tumor_area_pixels(binary_mask)
    
    print("=" * 50)
    print("BRAIN MRI ANALYSIS")
    print("=" * 50)
    print(f"Image:\n{os.path.basename(args.image)}\n")
    
    if tumor_area > 0:
        print("Prediction:\nTumor Detected\n")
        confidence = np.mean(prob_map[binary_mask > 0]) * 100
        print(f"Confidence:\n{confidence:.2f}%\n")
        print(f"Tumor Area:\n{tumor_area} pixels\n")
        
        tumor_pct = calculate_tumor_percentage(binary_mask, prob_map.shape)
        print(f"Tumor Percentage:\n{tumor_pct:.2f}% (relative to image area)\n")
        
        bbox = get_bounding_box(binary_mask)
        if bbox:
            print(f"Bounding Box:\nx = {bbox[0]}\ny = {bbox[1]}\nwidth = {bbox[2]}\nheight = {bbox[3]}\n")
            
        overlay = create_overlay(orig_img, binary_mask)
        overlay_with_box = draw_bounding_box(overlay, bbox)
        
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        mask_path = f"results/predictions/{base_name}_mask.png"
        overlay_path = f"results/overlays/{base_name}_overlay.png"
        
        cv2.imwrite(mask_path, binary_mask)
        cv2.imwrite(overlay_path, overlay_with_box)
        
        print(f"Mask:\n{mask_path}\n")
        print(f"Overlay:\n{overlay_path}\n")
    else:
        print("Prediction:\nNo Tumor Detected\n")
        print("No tumor pixels were detected by the model.\n")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
