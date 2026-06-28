from pathlib import Path
import sys

from image_captioning.logger import logger
from image_captioning.exception import CustomException
from image_captioning.entity.config_entity import DataValidationConfig


class DataValidation:
    """
    Validates the extracted Flickr8k dataset before preprocessing
    and model training.
    """

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self):
        """
        Validate dataset structure and required files.
        """
    def validate_dataset(self):
        """
        Validate the extracted Flickr8k dataset.
        """

        logger.info("Starting dataset validation...")

        try:

            images_dir_exists = self.config.images_dir.exists()
            captions_exists = self.config.captions_file.exists()
            train_exists = self.config.train_split.exists()
            validation_exists = self.config.validation_split.exists()
            test_exists = self.config.test_split.exists()

            validation_status = (
                images_dir_exists
                and captions_exists
                and train_exists
                and validation_exists
                and test_exists
            )

            with open(self.config.status_file, "w") as f:

                f.write(f"Images Directory : {images_dir_exists}\n")
                f.write(f"Captions File    : {captions_exists}\n")
                f.write(f"Train Split      : {train_exists}\n")
                f.write(f"Validation Split : {validation_exists}\n")
                f.write(f"Test Split       : {test_exists}\n")
                f.write("\n")

                if validation_status:
                    f.write("Validation Status : SUCCESS\n")
                else:
                    f.write("Validation Status : FAILED\n")

        except Exception as e:
            logger.error("Dataset validation failed.")
            raise CustomException(e, sys)