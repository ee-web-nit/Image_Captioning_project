import sys
import tensorflow as tf
import pandas as pd

from image_captioning.logger import logger
from image_captioning.exception import CustomException
from image_captioning.entity.config_entity import TokenizerBuilderConfig


class TokenizerBuilder:

    def __init__(self, config: TokenizerBuilderConfig):
        self.config = config

    def build_tokenizer(self):

        logger.info("Building tokenizer...")

        df = pd.read_csv(self.config.input_file)

        captions = df["caption"].tolist()

        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.config.max_tokens,
            output_mode="int",
            output_sequence_length=self.config.sequence_length,
        )

        vectorizer.adapt(captions)

        logger.info("Tokenizer built successfully.")

        return vectorizer