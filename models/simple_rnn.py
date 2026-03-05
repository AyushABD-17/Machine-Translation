from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from config import LEARNING_RATE, RNN_UNITS


def build_simple_rnn(input_shape, french_vocab_size):
    input_seq = Input(shape=(input_shape[1], 1))

    rnn = GRU(RNN_UNITS, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)
    output = Activation('softmax')(logits)

    model = Model(input_seq, output)

    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(LEARNING_RATE),
        metrics=['accuracy']
    )

    return model