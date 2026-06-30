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

image_path = (
    config.get_data_ingestion_config().images_dir
    / "1000268201_693b08cb0e.jpg"
)

features = predictor.extract_features(image_path)

print("=" * 60)
print("Feature Extraction Test")
print("=" * 60)

print("Feature Shape :", features.shape)

print()

print("Feature Extraction Passed ✅")