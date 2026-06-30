import tensorflow as tf

from image_captioning.models.positional_embedding import PositionalEmbedding

embedding = PositionalEmbedding(
    sequence_length=25,
    vocab_size=10000,
    embed_dim=512,
)

dummy = tf.constant(
    [
        [1, 20, 15, 2, 0, 0]
    ]
)

output = embedding(dummy)

print(output.shape)