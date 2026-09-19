import os
import cv2
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

def create_inspection_report(serial_number, status, date_str, time_str, 
                             zone_images, ocr_serial, confidence=None, 
                             defects=None, output_path=None):
    """
    Generate a PDF report for the tailgate inspection.
    zone_images: dict mapping zone name to image path (e.g. {'inner1': 'path/to/img.jpg', ...})
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 24)
    if status == "PASS":
        c.setFillColor(colors.darkgreen)
    else:
        c.setFillColor(colors.red)
    c.drawString(40, height - 50, f"Tailgate Inspection Report - {status}")
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 12)
    c.drawString(40, height - 80, f"Date: {date_str} {time_str}")
    c.drawString(40, height - 100, f"Serial Number: {serial_number}")
    
    if defects:
        c.setFillColor(colors.red)
        c.drawString(40, height - 120, f"Defects: {', '.join(defects)}")
    else:
        c.setFillColor(colors.darkgreen)
        c.drawString(40, height - 120, "Defects: None")
        
    c.setFillColor(colors.black)
    
    # ── 1. Serial Crop (Horizontal & Zoomed) ──────────────────────────────────
    crop_w = 280
    crop_h = 85
    crop_x = (width - crop_w) / 2
    crop_y = height - 230
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(crop_x, crop_y + crop_h + 8, "SERIAL CROP")
    
    crop_path = zone_images.get("ocr_crop")
    if crop_path and os.path.exists(crop_path):
        try:
            crop_bgr = cv2.imread(crop_path)
            if crop_bgr is not None:
                # Rotate counter-clockwise to horizontal if vertically oriented
                if crop_bgr.shape[0] > crop_bgr.shape[1]:
                    crop_bgr = cv2.rotate(crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Zoom in slightly (1.2x center crop) for improved readability
                zh, zw = crop_bgr.shape[:2]
                zoom = 1.2
                nh, nw = int(zh / zoom), int(zw / zoom)
                sy, sx = (zh - nh) // 2, (zw - nw) // 2
                crop_bgr = crop_bgr[sy:sy+nh, sx:sx+nw]
                
                crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                c.drawImage(ImageReader(crop_pil), crop_x, crop_y, width=crop_w, height=crop_h, preserveAspectRatio=True)
            else:
                raise ValueError("Could not read image data")
        except Exception as e:
            c.setStrokeColor(colors.red)
            c.rect(crop_x, crop_y, crop_w, crop_h)
            c.drawString(crop_x + 10, crop_y + crop_h / 2, "Error loading image")
    else:
        c.setStrokeColor(colors.gray)
        c.rect(crop_x, crop_y, crop_w, crop_h)
        c.drawString(crop_x + 10, crop_y + crop_h / 2, "No Image Available")

    # ── 2. Four Zone Images (Slightly Bigger, 2x2 Grid) ───────────────────────
    zone_w = 250
    zone_h = 145
    col1_x = 40
    col2_x = 305
    
    row1_y = crop_y - 35 - zone_h
    row2_y = row1_y - 30 - zone_h
    
    positions = {
        "inner1": (col1_x, row1_y),
        "inner2": (col2_x, row1_y),
        "outer1": (col1_x, row2_y),
        "outer2": (col2_x, row2_y)
    }
    
    for zone in ["inner1", "inner2", "outer1", "outer2"]:
        img_path = zone_images.get(zone)
        x, y = positions.get(zone, (0, 0))
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y + zone_h + 8, f"Zone: {zone.upper()}")
        
        if img_path and os.path.exists(img_path):
            try:
                c.drawImage(img_path, x, y, width=zone_w, height=zone_h, preserveAspectRatio=True)
            except Exception as e:
                c.setStrokeColor(colors.red)
                c.rect(x, y, zone_w, zone_h)
                c.drawString(x + 10, y + zone_h / 2, "Error loading image")
        else:
            c.setStrokeColor(colors.gray)
            c.rect(x, y, zone_w, zone_h)
            c.drawString(x + 10, y + zone_h / 2, "No Image Available")

    c.save()
    print(f"Report generated successfully: {output_path}")
