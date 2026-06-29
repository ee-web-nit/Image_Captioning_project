import tensorflow as tf
import pandas as pd

from image_captioning.utils.image_preprocessor import ImagePreprocessor

class DatasetBuilder:

    def __init__(
        self,
        vectorizer,
        batch_size=64
    ):

        self.vectorizer = vectorizer
        self.batch_size = batch_size

    def process_sample(
        self,
        image_path,
        caption
    ):

        image = ImagePreprocessor.preprocess(image_path)

        caption = self.vectorizer(
            tf.convert_to_tensor([caption])
        )

        caption = tf.squeeze(caption)

        return image, caption
    
    def build_dataset(
        self,
        dataframe,
        images_root
    ):

        images = [
            str(images_root / name)
            for name in dataframe["image_name"]
        ]

        captions = dataframe["caption"].tolist()

        dataset = tf.data.Dataset.from_tensor_slices(
            (images, captions)
        )

        dataset = dataset.map(

            lambda image, caption:

            self.process_sample(
                image,
                caption
            ),

            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.batch(
            self.batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset