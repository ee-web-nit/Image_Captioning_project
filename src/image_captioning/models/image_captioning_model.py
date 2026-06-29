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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.config = config

        # ---------------- CNN ---------------- #

        self.cnn_model = CNNEncoder().model

        # ---------------- Encoder ---------------- #

        self.encoder = TransformerEncoder(

            embed_dim=config.embed_dim,

            dense_dim=config.ff_dim,

            num_heads=config.num_heads,
        )

        # ---------------- Decoder ---------------- #

        self.decoder = TransformerDecoder(

            embed_dim=config.embed_dim,

            ff_dim=config.ff_dim,

            num_heads=config.num_heads,

            vocab_size=config.vocab_size,

            sequence_length=config.sequence_length,
        )

        # ---------------- Metrics ---------------- #

        self.loss_tracker = tf.keras.metrics.Mean(
            name="loss"
        )

        self.acc_tracker = tf.keras.metrics.Mean(
            name="accuracy"
        )

    # ==========================================================
    # Metrics
    # ==========================================================

    @property
    def metrics(self):

        return [

            self.loss_tracker,

            self.acc_tracker,
        ]

    # ==========================================================
    # Build
    # ==========================================================

    def build(self, input_shape):

        super().build(input_shape)

    # ==========================================================
    # Forward Pass
    # ==========================================================

    def call(
        self,
        inputs,
        training=False,
    ):

        image_inputs, caption_inputs = inputs

        image_features = self.cnn_model(
            image_inputs,
            training=False,
        )

        encoder_outputs = self.encoder(

            image_features,

            training=training,
        )

        mask = tf.math.not_equal(
            caption_inputs,
            0,
        )

        predictions = self.decoder(

            caption_inputs,

            encoder_outputs,

            training=training,

            mask=mask,
        )

        return predictions
    
    # ==========================================================
    # Compile
    # ==========================================================

    def compile(
        self,
        optimizer,
        loss_fn,
        **kwargs,
    ):

        super().compile(**kwargs)

        self.optimizer = optimizer

        self.loss_fn = loss_fn

    # ==========================================================
    # Loss
    # ==========================================================

    def calculate_loss(
        self,
        y_true,
        y_pred,
        mask,
    ):

        y_true = tf.cast(
            y_true,
            tf.int32,
        )

        loss = self.loss_fn(

            y_true,

            y_pred,
        )

        mask = tf.cast(

            mask,

            dtype=loss.dtype,
        )

        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    # ==========================================================
    # Accuracy
    # ==========================================================

    def calculate_accuracy(
        self,
        y_true,
        y_pred,
        mask,
    ):

        predictions = tf.argmax(

            y_pred,

            axis=-1,

            output_type=tf.int32,
        )

        y_true = tf.cast(

            y_true,

            tf.int32,
        )

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

    # ==========================================================
    # Train Step
    # ==========================================================

    def train_step(
        self,
        batch_data,
    ):

        batch_img, batch_seq = batch_data

        batch_seq_inp = batch_seq[:, :-1]

        batch_seq_true = batch_seq[:, 1:]

        mask = tf.math.not_equal(
            batch_seq_true,
            0,
        )

        with tf.GradientTape() as tape:

            predictions = self(
                (
                    batch_img,
                    batch_seq_inp,
                ),
                training=True,
            )

            loss = self.calculate_loss(

                batch_seq_true,

                predictions,

                mask,
            )

            accuracy = self.calculate_accuracy(

                batch_seq_true,

                predictions,

                mask,
            )

        trainable_variables = (

            self.encoder.trainable_variables +

            self.decoder.trainable_variables
        )

        gradients = tape.gradient(

            loss,

            trainable_variables,
        )

        self.optimizer.apply_gradients(

            zip(

                gradients,

                trainable_variables,
            )
        )

        self.loss_tracker.update_state(loss)

        self.acc_tracker.update_state(accuracy)

        return {

            "loss": self.loss_tracker.result(),

            "accuracy": self.acc_tracker.result(),
        }

    # ==========================================================
    # Validation Step
    # ==========================================================

    def test_step(
        self,
        batch_data,
    ):

        batch_img, batch_seq = batch_data

        batch_seq_inp = batch_seq[:, :-1]

        batch_seq_true = batch_seq[:, 1:]

        mask = tf.math.not_equal(
            batch_seq_true,
            0,
        )

        predictions = self(
            (
                batch_img,
                batch_seq_inp,
            ),
            training=False,
        )

        loss = self.calculate_loss(

            batch_seq_true,

            predictions,

            mask,
        )

        accuracy = self.calculate_accuracy(

            batch_seq_true,

            predictions,

            mask,
        )

        self.loss_tracker.update_state(loss)

        self.acc_tracker.update_state(accuracy)

        return {

            "loss": self.loss_tracker.result(),

            "accuracy": self.acc_tracker.result(),
        }
    # ==========================================================
    # Serialization
    # ==========================================================

    def get_config(self):

        config = super().get_config()

        config.update(

            {

                "config": {

                    "image_size": self.config.image_size,

                    "embed_dim": self.config.embed_dim,

                    "ff_dim": self.config.ff_dim,

                    "num_heads": self.config.num_heads,

                    "vocab_size": self.config.vocab_size,

                    "sequence_length": self.config.sequence_length,

                    "batch_size": self.config.batch_size,

                    "epochs": self.config.epochs,
                }
            }
        )

        return config

    @classmethod
    def from_config(
        cls,
        config,
    ):

        model_config = ModelConfig(

            image_size=config["config"]["image_size"],

            embed_dim=config["config"]["embed_dim"],

            ff_dim=config["config"]["ff_dim"],

            num_heads=config["config"]["num_heads"],

            vocab_size=config["config"]["vocab_size"],

            sequence_length=config["config"]["sequence_length"],

            batch_size=config["config"]["batch_size"],

            epochs=config["config"]["epochs"],
        )

        return cls(model_config)