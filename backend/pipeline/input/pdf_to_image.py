import fitz  # PyMuPDF
from PIL import Image
import numpy as np

def convert_pdf_to_images(pdf_bytes):
    """
    Converts a PDF file (bytes) into a list of PIL Images compatible with YOLO.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # 1. Define Resolution (DPI)
        # Standard YOLO models work best with clear images. 
        # zoom=2 roughly equals 150-200 DPI (good balance of speed/quality).
        mat = fitz.Matrix(2, 2) 
        
        # 2. Render to Pixmap (Rasterization)
        pix = page.get_pixmap(matrix=mat)
        
        # 3. Convert to PIL Image (YOLO expects this)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        
    return images
