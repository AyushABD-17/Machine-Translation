from keras.models import Sequential
from keras.layers import GRU, Embedding, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from config import LEARNING_RATE, RNN_UNITS, EMBEDDING_DIM


def build_embedding_rnn(input_shape, english_vocab_size, french_vocab_size):
    model = Sequential()

    model.add(
        Embedding(
            input_dim=english_vocab_size + 1,
            output_dim=EMBEDDING_DIM,
            input_length=input_shape[1]
        )
    )

    model.add(
        GRU(RNN_UNITS, return_sequences=True)
    )

    model.add(
        TimeDistributed(
            Dense(french_vocab_size, activation='softmax')
        )
    )

    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(LEARNING_RATE),
        metrics=['accuracy']
    )

    return model