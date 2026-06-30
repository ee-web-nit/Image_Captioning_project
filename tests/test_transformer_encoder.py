import tensorflow as tf

from image_captioning.models.transformer_encoder import TransformerEncoder

encoder = TransformerEncoder(
    embed_dim=512,
    dense_dim=512,
    num_heads=1,
)

dummy = tf.random.normal((2, 64, 512))

output = encoder(dummy)

print(output.shape)