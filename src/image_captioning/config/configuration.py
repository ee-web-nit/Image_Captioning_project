from pathlib import Path

from image_captioning.utils.common import read_yaml

from image_captioning.entity.config_entity import (
    DataIngestionConfig,
    ModelTrainerConfig,
)

from image_captioning.utils.common import (
    read_yaml,
    create_directories,
)


class ConfigurationManager:

    def __init__(
        self,
        config_filepath=Path("config/config.yaml"),
        params_filepath=Path("config/params.yaml"),
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns the configuration required for the Data Ingestion Pipeline.
        """

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            raw_dir=Path(config.raw_dir),
            extracted_dir=Path(config.extracted_dir),
            processed_dir=Path(config.processed_dir),
            image_zip=Path(config.image_zip),
            text_zip=Path(config.text_zip),
            images_dir=Path(config.images_dir),
            captions_file=Path(config.captions_file),
        )

        return data_ingestion_config