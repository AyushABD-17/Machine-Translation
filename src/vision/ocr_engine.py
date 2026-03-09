import easyocr

def extract_text(image_path):
    """
    Uses EasyOCR to detect English text from an image.
    Returns the combined string of extracted words.
    """
    # Initialize the reader for English
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Extract text components
    results = reader.readtext(image_path)
    
    # Recombine individual strings into a coherent sentence
    extracted = " ".join([text for (_, text, _) in results])
    
    return extracted.strip()
