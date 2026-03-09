import numpy as np
import yaml
from src.datasets.preprocessing import pad, tokenize, logits_to_text
from src.utils.model_io import load_saved_model
from src.datasets.data_loader import load_data

class TranslationPipeline:
    def __init__(self, model_path, config_path):
        from src.training.trainer import Trainer
        trainer = Trainer(config_path)
        trainer.prepare_data()
        trainer.initialize_model()
        
        # Load the weights manually instead of load_model to bypass deserialize bugs
        trainer.model.load_weights(model_path)
        self.model = trainer.model
        
        self.en_tokenizer = trainer.en_tokenizer
        self.fr_tokenizer = trainer.fr_tokenizer
        self.max_fr_len = trainer.max_fr_len

    def translate(self, text):
        seq = self.en_tokenizer.texts_to_sequences([text])
        padded = pad(seq, length=self.max_fr_len)
        
        prediction = self.model.predict(padded)
        translated = logits_to_text(prediction[0], self.fr_tokenizer)
        
        # Post-processing to clean up padding and unknown tokens
        translated_clean = translated.replace('<PAD>', '').strip()
        return translated_clean
