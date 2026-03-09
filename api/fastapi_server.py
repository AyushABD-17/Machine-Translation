from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.translation_pipeline import TranslationPipeline
from src.inference.image_translation_pipeline import run_image_translation
from src.pipeline.camera_translation_pipeline import CameraTranslationPipeline
import uvicorn
import os

app = FastAPI()

# Load model path dynamically or default to transformer if exists
MODEL_PATH = "saved_models/transformer/transformer_model.h5"
CONFIG_PATH = "configs/transformer_config.yaml"

if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
    text_pipeline = TranslationPipeline(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH
    )
    camera_pipeline = CameraTranslationPipeline(text_pipeline)
else:
    text_pipeline = None
    camera_pipeline = None

class TextTranslationRequest(BaseModel):
    sentence: str

class ImageTranslationRequest(BaseModel):
    image_path: str

@app.post("/translate_text")
def translate_text(request: TextTranslationRequest):
    if not text_pipeline:
        return {"error": "Model not trained yet."}
    translated = text_pipeline.translate(request.sentence)
    return {"translation": translated}

@app.post("/translate_image")
def translate_image(request: ImageTranslationRequest):
    if not text_pipeline:
        return {"error": "Model not trained yet."}
        
    en_text, fr_text = run_image_translation(request.image_path, text_pipeline)
    return {
        "detected_text": en_text,
        "translation": fr_text
    }

@app.get("/camera_translate")
def camera_translate():
    if not camera_pipeline:
        return {"error": "Model not trained yet."}
        
    try:
        results = camera_pipeline.run()
        return results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
