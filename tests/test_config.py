from image_captioning.config.configuration import ConfigurationManager

config = ConfigurationManager()

print(config.config.artifacts_root)

print(config.params.BATCH_SIZE)