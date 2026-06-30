import tensorflow as tf

from image_captioning.logger import logger


class CaptionGenerator:

    def __init__(
        self,
        tokenizer,
        model,
        max_length,
    ):

        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

        self.vocabulary = (
            tokenizer.vocabulary()
        )

        self.word_to_index = {
            word: idx
            for idx, word in enumerate(self.vocabulary)
        }

        self.index_to_word = {
            idx: word
            for idx, word in enumerate(self.vocabulary)
        }

    def generate(
        self,
        image_features,
    ):

        caption = ["<start>"]

        for _ in range(self.max_length - 1):

            sequence = [
                self.word_to_index[word]
                for word in caption
            ]

            sequence = tf.keras.preprocessing.sequence.pad_sequences(
                [sequence],
                maxlen=self.max_length,
                padding="post",
            )

            predictions = self.model.decoder(
                sequence,
                image_features,
                training=False,
            )

            next_token = tf.argmax(
                predictions[0, len(caption) - 1],
                axis=-1,
            ).numpy()

            next_word = self.index_to_word.get(
                next_token,
                "[UNK]",
            )

            if next_word == "<end>":
                break

            caption.append(
                next_word
            )

        return " ".join(caption[1:])