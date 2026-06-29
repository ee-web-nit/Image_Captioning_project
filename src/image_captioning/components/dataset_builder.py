from pathlib import Path

import pandas as pd
import tensorflow as tf

from image_captioning.utils.image_preprocessor import ImagePreprocessor




class DatasetBuilder:
    """
    Builds a tf.data.Dataset for the Image Captioning model.
    """

    def __init__(
        self,
        vectorizer,
        batch_size=64,
    ):

        self.batch_size = batch_size

        self.preprocessor = ImagePreprocessor()

        self.vectorizer = vectorizer

    def _load_dataframe(self, caption_file):
        return pd.read_csv(caption_file)

    def _prepare_lists(self, dataframe, image_dir):
        image_paths = [
            str(Path(image_dir) / image_name)
            for image_name in dataframe["image_name"]
        ]

        captions = dataframe["caption"].tolist()

        return image_paths, captions

    def _process(self, image_path, caption):
        """
        Python function executed through tf.py_function.
        """

        if isinstance(image_path, bytes):
            image_path = image_path.decode("utf-8")
        elif isinstance(image_path, tf.Tensor):
            image_path = image_path.numpy().decode("utf-8")

        if isinstance(caption, bytes):
            caption = caption.decode("utf-8")
        elif isinstance(caption, tf.Tensor):
            caption = caption.numpy().decode("utf-8")

        image = self.preprocessor.preprocess(image_path)

        caption = tf.cast(
            self.vectorizer.vectorize([caption])[0],
            tf.int32,
        )

        return image, caption

    def _tf_process(self, image_path, caption):
        image, caption = tf.py_function(
            self._process,
            [image_path, caption],
            [tf.float32, tf.int32],
        )

        image.set_shape((299, 299, 3))
        caption.set_shape((25,))

        return image, caption

    def build(
        self,
        caption_file,
        image_dir,
    ):
        """
        Returns:
            tf.data.Dataset
        """

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
            self._tf_process,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.batch(
            self.batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset