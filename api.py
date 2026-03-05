from fastapi import FastAPI
import numpy as np
from utils.model_io import load_saved_model
from utils.preprocessing import pad, tokenize, logits_to_text

app = FastAPI()

model = load_saved_model("saved_models/embed_model.h5")

@app.get("/translate")
def translate(text: str):

    seq, tokenizer = tokenize([text])
    padded = pad(seq)

    prediction = model.predict(padded)
    translated = logits_to_text(prediction[0], tokenizer)

    return {"input": text, "translation": translated}