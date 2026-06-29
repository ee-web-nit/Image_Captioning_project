import tensorflow as tf
class CNNEncoder:

    def __init__(self):

        self.model = self.build_model()

    def build_model(self):

        base_model = tf.keras.applications.EfficientNetB0(

            include_top=False,

            weights="imagenet",

            input_shape=(299, 299, 3),
        )

        base_model.trainable = False

        return tf.keras.Model(

            inputs=base_model.input,

            outputs=base_model.output,
    )