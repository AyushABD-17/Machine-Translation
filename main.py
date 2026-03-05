import numpy as np
from config import *
from utils.data_loader import load_data
from utils.preprocessing import preprocess, pad, logits_to_text
from models.simple_rnn import build_simple_rnn
from models.embedding_rnn import build_embedding_rnn

from utils.model_io import save_model, load_saved_model

save_model(embed_model, "saved_models/embed_model.h5")

model = load_saved_model("saved_models/embed_model.h5")


def main():

    print("Loading Data...")
    english_sentences = load_data(DATA_PATH_EN)
    french_sentences = load_data(DATA_PATH_FR)

    print("Preprocessing...")
    preproc_en, preproc_fr, en_tokenizer, fr_tokenizer = \
        preprocess(english_sentences, french_sentences)

    max_fr_len = preproc_fr.shape[1]
    english_vocab_size = len(en_tokenizer.word_index)
    french_vocab_size = len(fr_tokenizer.word_index)

    # Prepare input for Simple RNN
    tmp_x = pad(preproc_en, max_fr_len)
    tmp_x = tmp_x.reshape((-1, max_fr_len, 1))

    print("\nTraining Simple RNN Model...")
    simple_model = build_simple_rnn(tmp_x.shape, french_vocab_size)

    simple_model.fit(
        tmp_x,
        preproc_fr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )

    print("\nSimple RNN Prediction:")
    print(logits_to_text(
        simple_model.predict(tmp_x[:1])[0],
        fr_tokenizer
    ))

    # Train Embedding Model
    print("\nTraining Embedding RNN Model...")

    tmp_x_embed = pad(preproc_en, max_fr_len)

    embed_model = build_embedding_rnn(
        tmp_x_embed.shape,
        english_vocab_size,
        french_vocab_size
    )

    embed_model.fit(
        tmp_x_embed,
        preproc_fr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )

    print("\nEmbedding Model Prediction:")
    print(logits_to_text(
        embed_model.predict(tmp_x_embed[:1])[0],
        fr_tokenizer
    ))


if __name__ == "__main__":
    main()