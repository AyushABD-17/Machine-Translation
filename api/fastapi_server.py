from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.translation_pipeline import TranslationPipeline
import uvicorn
import os

app = FastAPI()

# Load model path dynamically or default to transformer if exists
MODEL_PATH = "saved_models/transformer/transformer_model.h5"
CONFIG_PATH = "configs/transformer_config.yaml"

if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
    pipeline = TranslationPipeline(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH
    )
else:
    pipeline = None

class TranslationRequest(BaseModel):
    sentence: str

@app.post("/translate")
def translate(request: TranslationRequest):
    if not pipeline:
        return {"error": "Model not trained yet. Run an experiment first."}
    
    translated = pipeline.translate(request.sentence)
    return {"translation": translated}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
