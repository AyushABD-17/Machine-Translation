import numpy as np
import yaml
from src.datasets.preprocessing import pad, tokenize, logits_to_text
from src.utils.model_io import load_saved_model
from src.datasets.data_loader import load_data

class TranslationPipeline:
    def __init__(self, model_path, config_path):
        self.model = load_saved_model(model_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract tokenizer state from small dataset subset as done previously
        en_path = config['data']['en_path']
        fr_path = config['data']['fr_path']
        
        english_sentences = load_data(en_path)
        french_sentences = load_data(fr_path)
        
        from src.datasets.preprocessing import preprocess
        _, preproc_fr, self.en_tokenizer, self.fr_tokenizer = preprocess(
            english_sentences, french_sentences
        )
            
        self.max_fr_len = preproc_fr.shape[1]

    def translate(self, text):
        seq = self.en_tokenizer.texts_to_sequences([text])
        padded = pad(seq, length=self.max_fr_len)
        
        prediction = self.model.predict(padded)
        translated = logits_to_text(prediction[0], self.fr_tokenizer)
        
        # Post-processing to clean up padding and unknown tokens
        translated_clean = translated.replace('<PAD>', '').strip()
        return translated_clean
