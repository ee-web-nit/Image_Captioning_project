from pathlib import Path

import pandas as pd
import tensorflow as tf

from image_captioning.utils.image_preprocessor import ImagePreprocessor


class DatasetBuilder:

    def __init__(
        self,
        vectorizer,
        batch_size=64,
    ):

        self.batch_size = batch_size
        self.vectorizer = vectorizer

    def _load_dataframe(self, caption_file):

        return pd.read_csv(caption_file)

    def _prepare_lists(self, dataframe, image_dir):

        image_paths = [
            str(Path(image_dir) / img)
            for img in dataframe["image_name"]
        ]

        captions = dataframe["caption"].tolist()

        return image_paths, captions

    def _process(self, image_path, caption):

        image = ImagePreprocessor.preprocess(image_path)

        caption = self.vectorizer.vectorize(tf.expand_dims(caption, 0))[0]

        return image, caption

    def build(
        self,
        caption_file,
        image_dir,
    ):

        dataframe = self._load_dataframe(caption_file)

        image_paths, captions = self._prepare_lists(
            dataframe,
            image_dir,
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                image_paths,
                captions,
            )
        )

        dataset = dataset.map(
            self._process,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.batch(
            self.batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset