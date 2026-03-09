from keras.models import Model
from keras.layers import Input, GRU, Dense, TimeDistributed
from keras.layers import Dot, Activation, Concatenate
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy

def build_attention_model(input_shape, english_vocab_size, french_vocab_size, rnn_units=64, learning_rate=1e-3):

    # Encoder
    encoder_inputs = Input(shape=(input_shape[1],))
    encoder_gru = GRU(rnn_units, return_sequences=True, return_state=True)
    encoder_outputs, encoder_state = encoder_gru(encoder_inputs)

    # Decoder
    decoder_gru = GRU(rnn_units, return_sequences=True)
    decoder_outputs = decoder_gru(encoder_outputs, initial_state=encoder_state)

    # Attention
    attention_scores = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
    attention_weights = Activation("softmax")(attention_scores)

    context = Dot(axes=[2, 1])([attention_weights, encoder_outputs])

    concat = Concatenate()([context, decoder_outputs])

    output = TimeDistributed(Dense(french_vocab_size, activation="softmax"))(concat)

    model = Model(encoder_inputs, output)

    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate),
        metrics=["accuracy"]
    )

    return model