import tensorflow as tf
from tensorflow.keras import layers


class TransformerEncoder(layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=0.0,
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

        self.dense = layers.Dense(
            embed_dim,
            activation="relu",
        )

    def call(self, inputs, training=False, mask=None):

        inputs = self.layernorm_1(inputs)

        inputs = self.dense(inputs)

        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )

        return self.layernorm_2(
            inputs + attention_output
        )