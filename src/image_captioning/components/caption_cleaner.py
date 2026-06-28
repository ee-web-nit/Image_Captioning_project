import sys
import pandas as pd
import re

from image_captioning.logger import logger
from image_captioning.exception import CustomException
from image_captioning.entity.config_entity import CaptionCleanerConfig


class CaptionCleaner:

    def __init__(self, config: CaptionCleanerConfig):
        self.config = config

    def clean_text(self, text: str) -> str:
        """
        Clean a single caption.
        """

        text = text.lower()

        text = re.sub(r"[^\w\s]", "", text)

        text = re.sub(r"\s+", " ", text)

        text = text.strip()

        text = "<start> " + text + " <end>"

        return text
    

    def clean_captions(self):

        logger.info("Starting caption cleaning...")

        try:

            df = pd.read_csv(self.config.input_file)

            df["caption"] = df["caption"].apply(self.clean_text)

            df.to_csv(
                self.config.output_file,
                index=False
            )

            logger.info(
                f"Caption cleaning completed successfully."
            )

        except Exception as e:

            logger.error("Caption cleaning failed.")

            raise CustomException(e, sys)