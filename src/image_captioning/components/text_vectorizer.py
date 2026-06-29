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