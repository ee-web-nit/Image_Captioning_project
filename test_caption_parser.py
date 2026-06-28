from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.caption_parser import CaptionParser

config_manager = ConfigurationManager()

config = config_manager.get_caption_parser_config()

caption_parser = CaptionParser(config)

caption_parser.parse_captions()