from image_captioning.config.configuration import ConfigurationManager

config = ConfigurationManager()

model_config = config.get_model_config()

print(model_config)