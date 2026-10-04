import cv2
import numpy as np

def threshold_probability_map(prob_map, threshold=0.5):
    binary_mask = (prob_map >= threshold).astype(np.uint8) * 255
    return binary_mask

def analyze_connected_components(binary_mask):
    """
    Returns a list of dicts containing stats for each connected component.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    regions = []
    
    # Label 0 is background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Filter tiny noise regions
        if area < 10:
            continue
            
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        
        # Calculate perimeter
        region_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
        
        regions.append({
            'label': i,
            'area': area,
            'bbox': (x, y, w, h),
            'centroid': (cx, cy),
            'perimeter': perimeter
        })
        
    # Sort by area descending
    regions.sort(key=lambda x: x['area'], reverse=True)
    return regions

def get_bounding_box(binary_mask):
    regions = analyze_connected_components(binary_mask)
    if not regions:
        return None
    # Return largest bbox for legacy compatibility
    return regions[0]['bbox']

def calculate_tumor_area_pixels(binary_mask):
    return int(np.sum(binary_mask > 0))

def calculate_tumor_percentage(binary_mask, image_shape):
    tumor_pixels = calculate_tumor_area_pixels(binary_mask)
    total_pixels = image_shape[0] * image_shape[1]
    return (tumor_pixels / total_pixels) * 100.0
