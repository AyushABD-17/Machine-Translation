import tensorflow as tf
from keras.layers import Input, Embedding, Dense, LayerNormalization
from keras.layers import MultiHeadAttention, Dropout
from keras.models import Model

def transformer_block(embed_dim, num_heads, ff_dim, rate=0.1):
    inputs = Input(shape=(None, embed_dim))

    attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    attn_output = attention(inputs, inputs)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(rate)(ffn)

    outputs = LayerNormalization(epsilon=1e-6)(out1 + ffn)

    return Model(inputs, outputs)


def build_transformer_model(max_len, english_vocab_size, french_vocab_size, embedding_dim=64, num_heads=4, ff_dim=128, learning_rate=1e-3):

    inputs = Input(shape=(max_len,))
    embedding = Embedding(english_vocab_size + 1, embedding_dim)(inputs)

    transformer = transformer_block(embedding_dim, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer(embedding)

    outputs = Dense(french_vocab_size, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=["accuracy"]
    )

    return model