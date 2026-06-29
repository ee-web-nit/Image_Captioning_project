import sys
from collections import Counter

import pandas as pd

from image_captioning.logger import logger
from image_captioning.exception import CustomException
from image_captioning.entity.config_entity import VocabularyBuilderConfig


class VocabularyBuilder:

    def __init__(self, config: VocabularyBuilderConfig):
        self.config = config

    def build_vocabulary(self):

        logger.info("Building vocabulary...")

        try:

            df = pd.read_csv(self.config.input_file)

            counter = Counter()

            for caption in df["caption"]:

                words = caption.split()

                counter.update(words)

            vocabulary = counter.most_common(
                self.config.max_vocabulary_size
            )

            with open(
                self.config.vocabulary_file,
                "w",
                encoding="utf-8"
            ) as file:

                for word, count in vocabulary:

                    file.write(f"{word}\t{count}\n")

            logger.info(
                f"Vocabulary created with {len(vocabulary)} words."
            )

        except Exception as e:

            logger.error("Vocabulary creation failed.")

            raise CustomException(e, sys)