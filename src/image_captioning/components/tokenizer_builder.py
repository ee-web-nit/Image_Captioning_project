from image_captioning.components.text_vectorizer import TextVectorizer
from image_captioning.logger import logger


class TokenizerBuilder:

    def __init__(self, config):
        self.config = config

    def build_tokenizer(self):

        logger.info("Loading tokenizer...")

        print("=" * 60)
        print("Tokenizer sequence length:", self.config.sequence_length)
        print("=" * 60)

        vectorizer = TextVectorizer(
            max_tokens=self.config.max_tokens,
            sequence_length=self.config.sequence_length,
        )

        vocab_size = vectorizer.load_vocabulary(
            self.config.vocabulary_file
        )

        logger.info(
            f"Tokenizer initialized with {vocab_size} tokens."
        )

        return vectorizer