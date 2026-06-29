import tensorflow as tf
from tensorflow.keras import layers

class TransformerDecoder(layers.Layer):

    def __init__(
        self,
        embed_dim,
        ff_dim,
        num_heads,
        vocab_size,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
        )
        self.position_embedding = layers.Embedding(
            input_dim=100,
            output_dim=embed_dim,
        )
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
        )
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.output_layer = layers.Dense(
            vocab_size,
            activation="softmax",
        )
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()