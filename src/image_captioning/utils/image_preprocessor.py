import tensorflow as tf
class ImagePreprocessor:

    IMAGE_SIZE = (299, 299)

    @staticmethod
    def preprocess(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(
            image,
            channels=3
        )
        image = tf.image.resize(
            image,
            ImagePreprocessor.IMAGE_SIZE
        )
        image = tf.keras.applications.efficientnet.preprocess_input(
            image
        )
        return image