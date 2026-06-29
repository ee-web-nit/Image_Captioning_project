import sys
import zipfile

from image_captioning.logger import logger
from image_captioning.exception import CustomException
from image_captioning.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """
    Handles extraction of the Flickr8k dataset
    from ZIP files into the project's artifacts directory.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _extract_zip(self, zip_path, output_dir):
        """
        Extract a ZIP file into the specified directory.
        """

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

    def extract_dataset(self):
        """
        Extract Flickr8k dataset if it has not already been extracted.
        """

        logger.info("Starting dataset extraction...")

        try:

            # ---------- Check ZIP files ----------

            if not self.config.image_zip.exists():
                raise FileNotFoundError(
                    f"Image ZIP not found: {self.config.image_zip}"
                )

            if not self.config.text_zip.exists():
                raise FileNotFoundError(
                    f"Text ZIP not found: {self.config.text_zip}"
                )

            logger.info("ZIP files found successfully.")

            # ---------- Skip if already extracted ----------

            if (
                self.config.images_dir.exists()
                and any(self.config.images_dir.iterdir())
                and self.config.captions_file.exists()
            ):

                logger.info(
                    "Dataset already extracted. Skipping extraction."
                )

                return

            # ---------- Create extraction directory ----------

            self.config.extracted_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ---------- Extract Images ----------

            logger.info("Extracting image dataset...")

            self._extract_zip(
                self.config.image_zip,
                self.config.extracted_dir,
            )

            # ---------- Extract Text ----------

            logger.info("Extracting caption dataset...")

            self._extract_zip(
                self.config.text_zip,
                self.config.extracted_dir,
            )

            logger.info(
                "Dataset extraction completed successfully."
            )

        except Exception as e:

            logger.error(
                "Dataset extraction failed."
            )

            raise CustomException(e, sys)