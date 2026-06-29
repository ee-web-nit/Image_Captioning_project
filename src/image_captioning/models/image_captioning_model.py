import tensorflow as tf

from tensorflow.keras import Model

from image_captioning.models.cnn_encoder import CNNEncoder
from image_captioning.models.transformer_encoder import TransformerEncoder
from image_captioning.models.transformer_decoder import TransformerDecoder
from image_captioning.entity.config_entity import ModelConfig

class ImageCaptioningModel(Model):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config
        self.cnn_model = CNNEncoder().model
        self.encoder = TransformerEncoder(

            embed_dim=config.embed_dim,

            dense_dim=config.ff_dim,

            num_heads=config.num_heads,
        )

        self.decoder = TransformerDecoder(

            embed_dim=config.embed_dim,

            ff_dim=config.ff_dim,

            num_heads=config.num_heads,

            vocab_size=config.vocab_size,

            sequence_length=config.sequence_length,
        )

        self.loss_tracker = tf.keras.metrics.Mean(
            name="loss"
        )

        self.acc_tracker = tf.keras.metrics.Mean(
            name="accuracy"
        )
    @property
    def metrics(self):

        return [

            self.loss_tracker,

            self.acc_tracker,

        ]