import tensorflow as tf

from image_captioning.models.transformer_decoder import TransformerDecoder

decoder = TransformerDecoder(
    embed_dim=256,
    ff_dim=512,
    num_heads=2,
    vocab_size=10000,
)

image_features = tf.random.normal((4, 64, 256))
captions = tf.random.uniform(
    (4, 40),
    maxval=10000,
    dtype=tf.int32,
)

output = decoder(captions, image_features)

print(output.shape)