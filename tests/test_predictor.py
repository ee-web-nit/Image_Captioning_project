from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.tokenizer_builder import TokenizerBuilder
from image_captioning.models.image_captioning_model import ImageCaptioningModel
from image_captioning.components.predictor import Predictor


config = ConfigurationManager()

model = ImageCaptioningModel(
    config.get_model_config()
)

tokenizer = TokenizerBuilder(
    config.get_tokenizer_builder_config()
).build_tokenizer()

predictor = Predictor(
    model=model,
    tokenizer=tokenizer,
    sequence_length=config.get_model_config().sequence_length,
)

print("=" * 60)
print("Predictor Test")
print("=" * 60)

print("Predictor initialized successfully.")
print("CNN Model      :", type(model.cnn_model).__name__)
print("Encoder        :", type(model.encoder).__name__)
print("Decoder        :", type(model.decoder).__name__)

print()
print("Predictor Test Passed ✅")