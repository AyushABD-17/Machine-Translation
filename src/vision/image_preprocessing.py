import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Prepares the image to improve text clarity for the OCR engine.
    - Grayscale conversion
    - Denoising
    - Contrast Enhancement
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Contrast Stretching (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # Replace original with preprocessed for easyocr
    preprocessed_path = image_path.replace(".jpg", "_preprocessed.jpg")
    cv2.imwrite(preprocessed_path, contrast_enhanced)
    
    return preprocessed_path
