import os
import yaml
import numpy as np
import tensorflow as tf
from src.datasets.data_loader import load_data
from src.datasets.preprocessing import preprocess, pad
from src.utils.model_io import save_model

from src.models.rnn_baseline import build_simple_rnn
from src.models.embedding_rnn import build_embedding_rnn
from src.models.attention_seq2seq import build_attention_model
from src.models.transformer_from_scratch import build_transformer_model

class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.en_path = self.config['data']['en_path']
        self.fr_path = self.config['data']['fr_path']
        self.save_dir = self.config['data']['save_dir']

        os.makedirs(self.save_dir, exist_ok=True)

        self.model_type = self.config['model']['type']
        
        # Hyperparameters
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.learning_rate = self.config['training']['learning_rate']
        self.validation_split = self.config['training']['validation_split']

        self.model = None

    def prepare_data(self):
        print("Loading Data...")
        english_sentences = load_data(self.en_path)
        french_sentences = load_data(self.fr_path)

        print("Preprocessing Data...")
        self.preproc_en, self.preproc_fr, self.en_tokenizer, self.fr_tokenizer = \
            preprocess(english_sentences, french_sentences)

        self.max_fr_len = self.preproc_fr.shape[1]
        self.english_vocab_size = len(self.en_tokenizer.word_index)
        self.french_vocab_size = len(self.fr_tokenizer.word_index) + 1

        # Padding input sequences to max_fr_len as done in main.py
        self.x = pad(self.preproc_en, self.max_fr_len)

    def initialize_model(self):
        print(f"Initializing {self.model_type} model...")
        if self.model_type == "simple_rnn":
            self.x = self.x.reshape((-1, self.max_fr_len, 1))
            rnn_units = self.config['model'].get('rnn_units', 64)
            self.model = build_simple_rnn(self.x.shape, self.french_vocab_size, rnn_units, self.learning_rate)
            
        elif self.model_type == "embedding_rnn":
            embedding_dim = self.config['model'].get('embedding_size', 64)
            rnn_units = self.config['model'].get('rnn_units', 64)
            self.model = build_embedding_rnn(self.x.shape, self.english_vocab_size, self.french_vocab_size, embedding_dim, rnn_units, self.learning_rate)
            
        elif self.model_type == "attention":
            rnn_units = self.config['model'].get('rnn_units', 64)
            self.model = build_attention_model(self.x.shape, self.english_vocab_size, self.french_vocab_size, rnn_units, self.learning_rate)
            
        elif self.model_type == "transformer":
            embedding_dim = self.config['model'].get('embedding_size', 256)
            num_heads = self.config['model'].get('attention_heads', 4)
            ff_dim = self.config['model'].get('ff_dim', 128)
            self.model = build_transformer_model(self.max_fr_len, self.english_vocab_size, self.french_vocab_size, embedding_dim, num_heads, ff_dim, self.learning_rate)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.summary()

    def train(self):
        print("Starting Training...")
        # Add callbacks for checkpointing and logging
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.h5')
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]

        history = self.model.fit(
            self.x,
            self.preproc_fr,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks
        )

        model_path = os.path.join(self.save_dir, f"{self.model_type}_model.h5")
        save_model(self.model, model_path)
        print(f"Training Complete. Model saved at {model_path}.")
        return history
