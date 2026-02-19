import fitz  # PyMuPDF
from PIL import Image
import io


def convert_pdf_to_images(pdf_bytes, dpi: int = 300):
    """
    Converts a PDF file (bytes) into high-resolution PIL Images
    suitable for YOLO detection and OCR.

    Args:
        pdf_bytes: Raw PDF file in memory.
        dpi: Rasterization DPI (300 recommended for OCR).

    Returns:
        List of PIL Images.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    zoom = dpi / 72  # PDF base DPI is 72
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Using PNG bytes avoids color/stride issues
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)

    return images
