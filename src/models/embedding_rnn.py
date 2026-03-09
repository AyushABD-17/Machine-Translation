from keras.models import Sequential
from keras.layers import GRU, Embedding, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy

def build_embedding_rnn(input_shape, english_vocab_size, french_vocab_size, embedding_dim=64, rnn_units=64, learning_rate=1e-3):
    model = Sequential()

    model.add(
        Embedding(
            input_dim=english_vocab_size + 1,
            output_dim=embedding_dim,
            input_length=input_shape[1]
        )
    )

    model.add(
        GRU(rnn_units, return_sequences=True)
    )

    model.add(
        TimeDistributed(
            Dense(french_vocab_size, activation='softmax')
        )
    )

    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate),
        metrics=['accuracy']
    )

    return model