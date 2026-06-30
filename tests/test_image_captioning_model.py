import tensorflow as tf

from image_captioning.config.configuration import ConfigurationManager
from image_captioning.models.image_captioning_model import ImageCaptioningModel

# Load configuration
config_manager = ConfigurationManager()
model_config = config_manager.get_model_config()

# Create model
model = ImageCaptioningModel(model_config)

# Compile model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction="none",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss_fn=loss_fn,
)

# Dummy inputs
batch_img = tf.random.normal((2, 299, 299, 3))

batch_seq = tf.random.uniform(
    shape=(2, model_config.sequence_length),
    maxval=model_config.vocab_size,
    dtype=tf.int32,
)

# Run one training step
history = model.train_step((batch_img, batch_seq))

print(history)