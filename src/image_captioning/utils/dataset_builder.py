from pathlib import Path

import pandas as pd
import tensorflow as tf

from image_captioning.utils.image_preprocessor import ImagePreprocessor


class DatasetBuilder:
    """
    Builds TensorFlow datasets for the Image Captioning model.

    Responsibilities:
    -----------------
    1. Read caption CSV
    2. Filter dataframe using official Flickr8k split files
    3. Convert image path + caption into tensors
    4. Return tf.data.Dataset

    NOTE:
    -----
    This class DOES NOT learn the vocabulary.
    The TextVectorization layer must already be adapted and
    passed into this class.
    """

    def __init__(
        self,
        vectorizer,
        batch_size=64,
        image_size=(299, 299),
    ):

        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self.image_size = image_size

        self.preprocessor = ImagePreprocessor()

    # ==========================================================
    # DataFrame Utilities
    # ==========================================================

    def load_dataframe(self, caption_file):
        """
        Reads cleaned_captions.csv
        """
        return pd.read_csv(caption_file)

    def filter_dataframe(
        self,
        dataframe,
        split_file,
    ):
        """
        Filters dataframe using the official Flickr8k split file.

        split_file:
            Flickr_8k.trainImages.txt
            Flickr_8k.devImages.txt
            Flickr_8k.testImages.txt
        """

        with open(split_file, "r") as f:

            image_names = set(

                line.strip()

                for line in f.readlines()

            )

        dataframe = dataframe[
            dataframe["image_name"].isin(image_names)
        ]

        dataframe = dataframe.reset_index(drop=True)

        return dataframe

    # ==========================================================
    # Internal Processing
    # ==========================================================

    def _process(
        self,
        image_path,
        caption,
    ):

        # ---------------------------------------
        # Decode Tensor -> Python String
        # ---------------------------------------

        if isinstance(image_path, tf.Tensor):
            image_path = image_path.numpy().decode("utf-8")

        if isinstance(caption, tf.Tensor):
            caption = caption.numpy().decode("utf-8")

        # ---------------------------------------
        # Image
        # ---------------------------------------

        image = self.preprocessor.preprocess(
            image_path
        )

        # ---------------------------------------
        # Caption
        # ---------------------------------------

        caption = self.vectorizer.vectorize(
            [caption]
        )[0]

        return image, caption

    def _tf_process(
        self,
        image_path,
        caption,
    ):

        image, caption = tf.py_function(

            self._process,

            [image_path, caption],

            [tf.float32, tf.int64],

        )

        image.set_shape(
            (
                self.image_size[0],
                self.image_size[1],
                3,
            )
        )

        caption.set_shape((25,))

        return image, caption

    # ==========================================================
    # Dataset Creation
    # ==========================================================

    def build_from_dataframe(
        self,
        dataframe,
        image_dir,
    ):

        image_paths = [

            str(Path(image_dir) / image_name)

            for image_name in dataframe["image_name"]

        ]

        captions = dataframe["caption"].tolist()

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

    # ==========================================================
    # Convenience Method
    # ==========================================================

    def build(
        self,
        caption_file,
        split_file,
        image_dir,
    ):

        dataframe = self.load_dataframe(
            caption_file
        )

        dataframe = self.filter_dataframe(
            dataframe,
            split_file,
        )

        return self.build_from_dataframe(
            dataframe,
            image_dir,
        )