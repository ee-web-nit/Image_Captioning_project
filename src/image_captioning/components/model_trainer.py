import pickle
import tensorflow as tf
from image_captioning.components.dataset_builder import DatasetBuilder
from image_captioning.models.image_captioning_model import ImageCaptioningModel


def load_tokenizer(self):

    with open(self.config.tokenizer_path, "rb") as f:

        self.tokenizer = pickle.load(f)

    print("Tokenizer loaded.")
class ModelTrainer:

    def __init__(self, config):

        self.config = config

        self.model = None

        self.tokenizer = None

        self.train_dataset = None

        self.validation_dataset = None

        self.callbacks = None


    def load_tokenizer(self):

        with open(self.config.tokenizer_path, "rb") as f:

            self.tokenizer = pickle.load(f)

        print("Tokenizer loaded.")

    def load_dataset(self):

        dataset_builder = DatasetBuilder(self.config)

        self.train_dataset = dataset_builder.get_train_dataset()

        self.validation_dataset = dataset_builder.get_validation_dataset()

        print("Datasets loaded.")

    def build_model(self):

        self.model = ImageCaptioningModel(

            self.config
        )

        print("Model initialized.")

    def compile_model(self):

        optimizer = tf.keras.optimizers.Adam(

            learning_rate=self.config.learning_rate
        )

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(

            from_logits=False,

            reduction="none",
        )

        self.model.compile(

            optimizer=optimizer,

            loss=loss_fn,
        )

        print("Model compiled.")

    def create_callbacks(self):

        self.callbacks = [

            tf.keras.callbacks.ModelCheckpoint(

                filepath=self.config.checkpoint_dir / "best_model.keras",

                monitor="val_loss",

                save_best_only=True,
            ),

            tf.keras.callbacks.EarlyStopping(

                monitor="val_loss",

                patience=5,

                restore_best_weights=True,
            ),

            tf.keras.callbacks.ReduceLROnPlateau(

                monitor="val_loss",

                factor=0.2,

                patience=3,
            ),

            tf.keras.callbacks.TensorBoard(

                log_dir=self.config.tensorboard_dir,
            ),

            tf.keras.callbacks.CSVLogger(

                self.config.history_dir / "training.csv",
            ),
        ]

    def train(self):

        history = self.model.fit(

            self.train_dataset,

            validation_data=self.validation_dataset,

            epochs=self.config.epochs,

            callbacks=self.callbacks,
        )

        return history
    
    def save_model(self):

        self.model.save(

            self.config.model_dir / "image_captioning_model.keras"
        )

        print("Model saved successfully.")