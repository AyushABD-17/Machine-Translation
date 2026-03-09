from src.vision.ocr_engine import extract_text
from src.inference.translation_pipeline import TranslationPipeline
import os

# Uses the globally created translation pipeline directly
# assuming the `transformer_model.h5` exists or fall back dynamically.
def run_image_translation(image_path, pipeline: TranslationPipeline):
    """
    Reads text out of an image using OCR and translates it.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image path provided does not exist.")
        
    detected_english = extract_text(image_path)
    if not detected_english:
        return "", ""
        
    translated_french = pipeline.translate(detected_english)
    
    return detected_english, translated_french
