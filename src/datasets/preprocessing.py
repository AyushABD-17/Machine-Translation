import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize(x):
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer


def pad(x, length=None):
    if length is None:
        length = max(len(sentence) for sentence in x)
    return pad_sequences(x, maxlen=length, padding='post')


def preprocess(x, y):
    preprocess_x, x_tokenizer = tokenize(x)
    preprocess_y, y_tokenizer = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tokenizer, y_tokenizer


def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join(
        index_to_words[prediction]
        for prediction in np.argmax(logits, 1)
    )