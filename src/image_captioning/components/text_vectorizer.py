import tensorflow as tf


class TextVectorizer:

    def __init__(
        self,
        max_tokens=10000,
        sequence_length=25,
    ):

        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=sequence_length,
            standardize=None,
        )

    def adapt(self, captions):
        self.vectorizer.adapt(captions)

    def vectorize(self, captions):
        return self.vectorizer(captions)

    def vocabulary(self):
        return self.vectorizer.get_vocabulary()

    def save_vocabulary(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for word in self.vectorizer.get_vocabulary():
                f.write(word + "\n")

    def load_vocabulary(self, filepath):

        with open(filepath, "r", encoding="utf-8") as f:
            vocab = [
                line.strip()
                for line in f
                if line.strip()
            ]

        self.vectorizer.set_vocabulary(vocab)

        return len(vocab)