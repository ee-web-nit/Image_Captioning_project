from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.data_validation import DataValidation

config_manager = ConfigurationManager()

config = config_manager.get_data_validation_config()

data_validation = DataValidation(config)

data_validation.validate_dataset()