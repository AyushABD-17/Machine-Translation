from src.vision.camera_capture import capture_image
from src.vision.image_preprocessing import preprocess_image
from src.speech.tts_engine import speak_translation
from src.inference.image_translation_pipeline import run_image_translation
from src.inference.translation_pipeline import TranslationPipeline

class CameraTranslationPipeline:
    def __init__(self, translation_pipeline: TranslationPipeline):
        self.translation_pipeline = translation_pipeline

    def run(self):
        """
        Executes the entire end-to-end multimodal pipeline:
        1. Capture image from the camera
        2. Preprocess the image
        3. OCR text -> Translate Text
        4. Synthesize speech audio
        
        Returns a dictionary containing the outputs of each step.
        """
        image_path = capture_image()
        preprocessed_path = preprocess_image(image_path)
        
        english_text, french_text = run_image_translation(
            preprocessed_path, 
            self.translation_pipeline
        )
        
        audio_file = ""
        if french_text:
            audio_file = speak_translation(french_text)
            
        return {
            "detected_text": english_text,
            "translation": french_text,
            "audio_file": audio_file
        }
