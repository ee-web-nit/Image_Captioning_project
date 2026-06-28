from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.caption_cleaner import CaptionCleaner

config = ConfigurationManager().get_caption_cleaner_config()

cleaner = CaptionCleaner(config)

cleaner.clean_captions()