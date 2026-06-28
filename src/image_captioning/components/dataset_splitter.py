import sys
import pandas as pd

from image_captioning.logger import logger
from image_captioning.exception import CustomException
from image_captioning.entity.config_entity import DatasetSplitterConfig


class DatasetSplitter:

    def __init__(self, config: DatasetSplitterConfig):
        self.config = config
        
    def load_image_list(self, file_path):

        with open(file_path, "r") as file:

            image_list = [
                line.strip()
                for line in file
                if line.strip()
            ]

        return image_list

    def split_dataset(self):

        logger.info("Starting dataset splitting...")

        try:

            df = pd.read_csv(self.config.captions_file)

            train_images = self.load_image_list(
                self.config.train_images
            )

            validation_images = self.load_image_list(
                self.config.validation_images
            )

            test_images = self.load_image_list(
                self.config.test_images
            )

            train_df = df[
                df["image_name"].isin(train_images)
            ]

            validation_df = df[
                df["image_name"].isin(validation_images)
            ]

            test_df = df[
                df["image_name"].isin(test_images)
            ]

            train_df.to_csv(
                self.config.train_output,
                index=False
            )

            validation_df.to_csv(
                self.config.validation_output,
                index=False
            )

            test_df.to_csv(
                self.config.test_output,
                index=False
            )

            logger.info("Dataset splitting completed successfully.")

        except Exception as e:

            logger.error("Dataset splitting failed.")

            raise CustomException(e, sys)