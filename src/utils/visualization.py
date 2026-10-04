import cv2
import numpy as np

def create_overlay(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    return image.copy()
    
def draw_regions(image, mask, regions, show_bbox=True, show_centroid=True):
    img_with_regions = image.copy()
    
    # Draw contour/boundary for the entire mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_with_regions, contours, -1, (0, 255, 255), 2) # Yellow contour
    
    for idx, r in enumerate(regions):
        x, y, w, h = r['bbox']
        cx, cy = int(r['centroid'][0]), int(r['centroid'][1])
        
        if show_bbox:
            cv2.rectangle(img_with_regions, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_with_regions, f"R{idx+1}", (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        if show_centroid:
            cv2.circle(img_with_regions, (cx, cy), 3, (255, 0, 0), -1)
            
    return img_with_regions
