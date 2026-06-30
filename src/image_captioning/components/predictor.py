import tensorflow as tf

from image_captioning.utils.image_preprocessor import ImagePreprocessor
from image_captioning.components.caption_generator import CaptionGenerator
from image_captioning.logger import logger


class Predictor:

    def __init__(
        self,
        model,
        tokenizer,
        sequence_length,
    ):

        self.model = model

        self.generator = CaptionGenerator(
            tokenizer=tokenizer,
            model=model,
            max_length=sequence_length,
        )

    def extract_features(
        self,
        image_path,
    ):

        image = ImagePreprocessor.preprocess(
            image_path
        )

        image = tf.expand_dims(
            image,
            axis=0,
        )

        features = self.model.cnn_model(
            image,
            training=False,
        )

        features = self.model.encoder(
            features,
            training=False,
        )

        logger.info(
            "Image features extracted."
        )

        return features

    def predict(
        self,
        image_path,
    ):

        features = self.extract_features(
            image_path
        )

        caption = self.generator.generate(
            features
        )

        return caption