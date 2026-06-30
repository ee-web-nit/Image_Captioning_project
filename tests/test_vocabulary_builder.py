from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.vocabulary_builder import VocabularyBuilder

config = ConfigurationManager().get_vocabulary_builder_config()

builder = VocabularyBuilder(config)

builder.build_vocabulary()