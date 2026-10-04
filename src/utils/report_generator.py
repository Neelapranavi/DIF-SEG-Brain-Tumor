from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import datetime

def generate_pdf_report(report_path, data):
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "DIF-SEG: Data-Driven Model for Medical Image Detection")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, "Brain Tumor Detection and Segmentation Analysis")
    
    c.setFont("Helvetica", 10)
    y = height - 80
    c.drawString(50, y, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 15
    c.drawString(50, y, f"Image: {data.get('image_name', 'Unknown')}")
    y -= 15
    c.drawString(50, y, f"Model: {data.get('model', 'DIF-SEG')}")
    y -= 15
    c.drawString(50, y, f"Device: {data.get('device', 'CPU')}")
    y -= 15
    c.drawString(50, y, f"Inference Time: {data.get('inference_time', 0):.2f} seconds")
    y -= 25
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Analysis Results")
    y -= 20
    c.setFont("Helvetica", 10)
    
    c.drawString(50, y, f"Detection Result: {data.get('detection_result', 'N/A')}")
    y -= 15
    
    if data.get('tumor_area', 0) > 0:
        c.drawString(50, y, f"Confidence: {data.get('confidence', 0):.2f}%")
        y -= 15
        c.drawString(50, y, f"Total Tumor Area: {data.get('tumor_area', 0)} pixels")
        y -= 15
        c.drawString(50, y, f"Tumor Coverage: {data.get('tumor_pct', 0):.2f}%")
        y -= 25
        
        regions = data.get('regions', [])
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Connected Regions Detected: {len(regions)}")
        y -= 20
        c.setFont("Helvetica", 10)
        
        for idx, r in enumerate(regions):
            bbox = r['bbox']
            cx, cy = r['centroid']
            area = r['area']
            c.drawString(60, y, f"Region {idx+1}: Area={area} px, Centroid=({cx:.1f}, {cy:.1f}), BBox=(X:{bbox[0]} Y:{bbox[1]} W:{bbox[2]} H:{bbox[3]})")
            y -= 15
            if y < 100:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
    else:
        c.drawString(50, y, "No tumor pixels were detected by the model.")
        y -= 25
        
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Image Statistics")
    y -= 20
    c.setFont("Helvetica", 10)
    img_stats = data.get('image_stats', {})
    c.drawString(60, y, f"Min: {img_stats.get('min', 0):.2f}, Max: {img_stats.get('max', 0):.2f}, Mean: {img_stats.get('mean', 0):.2f}, Std: {img_stats.get('std', 0):.2f}")
    y -= 40
    
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer:")
    y -= 15
    c.drawString(50, y, "This system is intended for academic, research, and educational purposes only.")
    y -= 15
    c.drawString(50, y, "It is not a certified medical diagnostic system and should not be used as a")
    y -= 15
    c.drawString(50, y, "substitute for evaluation by a qualified medical professional.")
    
    c.save()
    return report_path
