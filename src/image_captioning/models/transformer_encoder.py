import tensorflow as tf

from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):

    def __init__(

        self,

        embed_dim,

        dense_dim,

        num_heads,

        **kwargs
    ):

        super().__init__(**kwargs)

        self.attention = layers.MultiHeadAttention(

            num_heads=num_heads,

            key_dim=embed_dim,
        )

        self.dense_proj = tf.keras.Sequential(

            [

                layers.Dense(

                    dense_dim,

                    activation="relu",
                ),

                layers.Dense(embed_dim),

            ]
        )

        self.layernorm_1 = layers.LayerNormalization()

        self.layernorm_2 = layers.LayerNormalization()

    def call(

        self,

        inputs,

        training=False
    ):
        attention_output = self.attention(

            query=inputs,

            value=inputs,

            key=inputs,

            training=training,
        )

        proj_input = self.layernorm_1(

            inputs + attention_output
        )

        proj_output = self.dense_proj(

            proj_input
        )

        return self.layernorm_2(

            proj_input + proj_output
        )