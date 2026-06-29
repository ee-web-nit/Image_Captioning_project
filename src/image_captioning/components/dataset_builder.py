import tensorflow as tf
import pandas as pd
from pathlib import Path

from image_captioning.utils.image_preprocessor import ImagePreprocessor
from image_captioning.components.text_vectorizer import TextVectorizer

class DatasetBuilder:

    def __init__(self, image_size=(299, 299)):

        self.preprocessor = ImagePreprocessor()

        self.vectorizer = TextVectorizer()

    def load_dataframe(self, caption_file):

        df = pd.read_csv(caption_file)

        return df
    def adapt_vectorizer(self, dataframe):

        captions = dataframe["caption"].tolist()

        self.vectorizer.adapt(captions)
    def load_image(self, image_path):

        return self.preprocessor.preprocess(image_path)
    def vectorize_caption(self, caption):

        return self.vectorizer.vectorize([caption])[0]
    def create_lists(self, dataframe, image_dir):

        images = []

        captions = []

        for _, row in dataframe.iterrows():

            images.append(

                str(
                    Path(image_dir) / row["image"]
                )
            )

            captions.append(row["caption"])

        return images, captions
    
    def create_dataset(
        self,
        image_paths,
        captions,
    ):

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                image_paths,
                captions,
            )
        )

        return dataset