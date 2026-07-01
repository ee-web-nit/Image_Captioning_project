import tensorflow as tf
import time

from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.tokenizer_builder import TokenizerBuilder
from image_captioning.components.predictor import Predictor
from image_captioning.models.image_captioning_model import ImageCaptioningModel


class PredictionPipeline:

    def __init__(self):

        config = ConfigurationManager()

        self.model_config = config.get_model_config()
        self.tokenizer_config = config.get_tokenizer_builder_config()
        self.training_config = config.get_training_config()

        self.tokenizer = TokenizerBuilder(
            self.tokenizer_config
        ).build_tokenizer()

        self.model = ImageCaptioningModel(
            self.model_config
        )

        dummy_image = tf.zeros(
            (1, 299, 299, 3)
        )

        dummy_caption = tf.zeros(
            (
                1,
                self.model_config.sequence_length - 1,
            ),
            dtype=tf.int32,
        )

        _ = self.model(
            (
                dummy_image,
                dummy_caption,
            )
        )

        self.model.load_weights(
            self.training_config.model_dir /
            "final.weights.h5"
        )

        self.predictor = Predictor(
            model=self.model,
            tokenizer=self.tokenizer,
            sequence_length=self.model_config.sequence_length,
        )

    def predict(
        self,
        image_path,
    ):

        start_time = time.perf_counter()

        caption = self.predictor.predict(
            image_path
        )

        inference_time = round(
            time.perf_counter() - start_time,
            2,
        )

        return caption, inference_time