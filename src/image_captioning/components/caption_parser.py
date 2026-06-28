import sys
import pandas as pd

from image_captioning.logger import logger
from image_captioning.exception import CustomException
from image_captioning.entity.config_entity import CaptionParserConfig


class CaptionParser:

    def __init__(self, config: CaptionParserConfig):
        self.config = config

    def parse_captions(self):
        """
        Parse Flickr8k captions file and convert it into a CSV file.
        """

        logger.info("Starting caption parsing...")

        try:

            rows = []

            with open(self.config.captions_file, "r", encoding="utf-8") as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    image_caption, caption = line.split("\t")

                    image_name, caption_id = image_caption.split("#")

                    rows.append(
                        {
                            "image_name": image_name,
                            "caption_id": int(caption_id),
                            "caption": caption,
                        }
                    )

            df = pd.DataFrame(rows)

            df.to_csv(
                self.config.output_file,
                index=False
            )

            logger.info(
                f"Caption parsing completed successfully. "
                f"Saved {len(df)} captions."
            )

        except Exception as e:

            logger.error("Caption parsing failed.")

            raise CustomException(e, sys)