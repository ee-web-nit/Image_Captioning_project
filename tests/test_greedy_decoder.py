from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.tokenizer_builder import TokenizerBuilder
from image_captioning.models.image_captioning_model import ImageCaptioningModel
from image_captioning.components.caption_generator import CaptionGenerator

import tensorflow as tf

config = ConfigurationManager()

model = ImageCaptioningModel(
    config.get_model_config()
)

tokenizer = TokenizerBuilder(
    config.get_tokenizer_builder_config()
).build_tokenizer()

generator = CaptionGenerator(
    tokenizer=tokenizer,
    model=model,
    max_length=config.get_model_config().sequence_length,
)

dummy_features = tf.random.normal(
    (
        1,
        100,
        config.get_model_config().embed_dim,
    )
)

print("=" * 60)
print("Greedy Decoder Test")
print("=" * 60)

caption = generator.generate(
    dummy_features
)

print("Generated Caption:")
print(caption)

print()
print("Greedy Decoder Passed ✅")