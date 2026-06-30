from image_captioning.components.tokenizer_builder import TokenizerBuilder
from image_captioning.components.caption_generator import CaptionGenerator
from image_captioning.config.configuration import ConfigurationManager


class DummyModel:

    def __init__(self):
        self.decoder = None


config = ConfigurationManager()

tokenizer = TokenizerBuilder(
    config.get_tokenizer_builder_config()
).build_tokenizer()

generator = CaptionGenerator(
    tokenizer=tokenizer,
    model=DummyModel(),
    max_length=config.get_model_config().sequence_length,
)

print("=" * 60)
print("Caption Generator Test")
print("=" * 60)
print(generator.vocabulary[:20])
print(generator.vocabulary[-20:])
print("<start>" in generator.vocabulary)
print("start" in generator.vocabulary)
print("<end>" in generator.vocabulary)
print("end" in generator.vocabulary)

print("Vocabulary Size :", len(generator.vocabulary))
print("<start> index :", generator.word_to_index["<start>"])
print("<end> index   :", generator.word_to_index["<end>"])

print()

print("Initialization Successful ✅")