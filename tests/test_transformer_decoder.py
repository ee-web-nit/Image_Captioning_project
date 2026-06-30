import tensorflow as tf

from image_captioning.models.transformer_decoder import TransformerDecoder

decoder = TransformerDecoder(
    embed_dim=512,
    ff_dim=512,
    num_heads=2,
    vocab_size=10000,
    sequence_length=25,
)

image_features = tf.random.normal((2, 64, 512))

captions = tf.random.uniform(
    (2, 25),
    maxval=10000,
    dtype=tf.int32,
)

output = decoder(
    captions,
    image_features,
    training=False,
)

print(output.shape)