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
    
    def compile(self, optimizer, loss_fn):
        super().compile()

        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def calculate_loss(self, y_true, y_pred, mask):

        y_true = tf.cast(y_true, tf.int32)
        loss = self.loss_fn(
            y_true,
            y_pred,
        )

        mask = tf.cast(mask, dtype=loss.dtype)

        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    def calculate_accuracy(self, y_true, y_pred, mask):

        predictions = tf.argmax(
            y_pred,
            axis=2,
            output_type=tf.int32,
        )

        y_true = tf.cast(y_true, tf.int32)

        matches = tf.equal(
            y_true,
            predictions,
        )
        matches = tf.logical_and(
            mask,
            matches,
        )

        matches = tf.cast(
            matches,
            tf.float32,
        )

        mask = tf.cast(
            mask,
            tf.float32,
        )

        return tf.reduce_sum(matches) / tf.reduce_sum(mask)

    def train_step(self, batch_data):

        batch_img, batch_seq = batch_data

  

        batch_seq_inp = batch_seq[:, :-1]

        batch_seq_true = batch_seq[:, 1:]

        mask = tf.math.not_equal(
            batch_seq_true,
            0,
        )

        with tf.GradientTape() as tape:

            img_embed = self.cnn_model(
                batch_img,
                training=False,
            )

            encoder_out = self.encoder(
                img_embed,
                training=True,
            )

            predictions = self.decoder(
                batch_seq_inp,
                encoder_out,
                training=True,
                mask=mask,
            )

            loss = self.calculate_loss(
                batch_seq_true,
                predictions,
                mask,
            )

            acc = self.calculate_accuracy(
                batch_seq_true,
                predictions,
                mask,
            )
            train_vars = (

                self.encoder.trainable_variables +

                self.decoder.trainable_variables
            )
        grads = tape.gradient(

            loss,

            train_vars,
        )
        self.optimizer.apply_gradients(

            zip(

                grads,

                train_vars,
            )
        )
        self.loss_tracker.update_state(loss)

        self.acc_tracker.update_state(acc)

        return {
        "loss": self.loss_tracker.result(),

        "accuracy": self.acc_tracker.result(),
        }
    def test_step(self, batch_data):

        batch_img, batch_seq = batch_data

        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]

        mask = tf.math.not_equal(batch_seq_true, 0)

        img_embed = self.cnn_model(batch_img, training=False)

        encoder_out = self.encoder(
            img_embed,
            training=False,
        )

        predictions = self.decoder(
            batch_seq_inp,
            encoder_out,
            training=False,
            mask=mask,
        )

        loss = self.calculate_loss(
            batch_seq_true,
            predictions,
            mask,
        )

        acc = self.calculate_accuracy(
            batch_seq_true,
            predictions,
            mask,
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
        }