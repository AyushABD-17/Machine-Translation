from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy

def build_simple_rnn(input_shape, french_vocab_size, rnn_units=64, learning_rate=1e-3):
    input_seq = Input(shape=(input_shape[1], 1))

    rnn = GRU(rnn_units, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)
    output = Activation('softmax')(logits)

    model = Model(input_seq, output)

    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate),
        metrics=['accuracy']
    )

    return model