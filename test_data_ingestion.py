from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.data_ingestion import DataIngestion

config_manager = ConfigurationManager()

config = config_manager.get_data_ingestion_config()

data_ingestion = DataIngestion(config)

data_ingestion.extract_dataset()