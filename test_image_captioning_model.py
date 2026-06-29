from image_captioning.config.configuration import ConfigurationManager

from image_captioning.models.image_captioning_model import ImageCaptioningModel


config = ConfigurationManager()

model_config = config.get_model_config()

model = ImageCaptioningModel(model_config)

print(model)