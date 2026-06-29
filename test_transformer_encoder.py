import tensorflow as tf

from image_captioning.models.transformer_encoder import TransformerEncoder

encoder = TransformerEncoder(

    embed_dim=256,

    dense_dim=512,

    num_heads=2,
)

dummy = tf.random.normal(

    shape=(4, 64, 256)
)

output = encoder(dummy)

print(output.shape)