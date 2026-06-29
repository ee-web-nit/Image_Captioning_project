
import tensorflow as tf
from image_captioning.components.dataset_builder import DatasetBuilder
from image_captioning.models.image_captioning_model import ImageCaptioningModel
from image_captioning.components.tokenizer_builder import TokenizerBuilder
from image_captioning.logger import logger
from pathlib import Path
import json
import sys
from image_captioning.exception import CustomException


class ModelTrainer:

    def __init__(

        self,

        training_config,

        model_config,

        tokenizer_config,

        splitter_config,

        data_config,
    ):

        self.training_config = training_config

        self.model_config = model_config

        self.tokenizer_config = tokenizer_config

        self.splitter_config = splitter_config

        self.data_config = data_config

        self.model = None

        self.tokenizer = None

        self.train_dataset = None

        self.validation_dataset = None

        self.callbacks = None

    def load_tokenizer(self):
        try:
            tokenizer_builder = TokenizerBuilder(
                self.tokenizer_config
            )

            self.tokenizer = tokenizer_builder.build_tokenizer()

            logger.info("Tokenizer loaded successfully.")

        except Exception as e:

            raise CustomException(e, sys)

    def load_dataset(self):
        try:

            dataset_builder = DatasetBuilder(
                vectorizer=self.tokenizer,
                batch_size=self.training_config.batch_size,
            )

            self.train_dataset = dataset_builder.build(
                caption_file=self.splitter_config.train_output,
                image_dir=self.data_config.images_dir,
            )

            self.validation_dataset = dataset_builder.build(
                caption_file=self.splitter_config.validation_output,
                image_dir=self.data_config.images_dir,
            )

            logger.info("Training and validation datasets created.")
        except Exception as e:

            raise CustomException(e, sys)

    def build_model(self):

        try:


            self.model = ImageCaptioningModel(
                self.model_config
            )

            logger.info("Model initialized.")

        except Exception as e:

            raise CustomException(e, sys)

    def compile_model(self):
        try:


            optimizer = tf.keras.optimizers.Adam(

                learning_rate=self.training_config.learning_rate
            )

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(

                from_logits=False,

                reduction="none",
            )

            self.model.compile(
                optimizer=optimizer,
                loss_fn=loss_fn,
            )

            logger.info("Model initialized.")
        except Exception as e:

            raise CustomException(e, sys)
        
    def create_callbacks(self):
        try:
            Path(self.training_config.checkpoint_dir).mkdir(
                parents=True,
                exist_ok=True,
            )

            Path(self.training_config.model_dir).mkdir(
                parents=True,
                exist_ok=True,
            )

            Path(self.training_config.history_dir).mkdir(
                parents=True,
                exist_ok=True,
            )

            Path(self.training_config.tensorboard_dir).mkdir(
                parents=True,
                exist_ok=True,
            )

            self.callbacks = [

                tf.keras.callbacks.ModelCheckpoint(

                    filepath=self.training_config.checkpoint_dir /
                    "best_model.weights.h5",

                    monitor="val_loss",

                    save_best_only=True,

                    save_weights_only=True,
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

                    log_dir=self.training_config.tensorboard_dir,
                ),

                tf.keras.callbacks.CSVLogger(

                    self.training_config.history_dir / "training.csv",
                ),
            ]
        except Exception as e:

            raise CustomException(e, sys)

    def train(self):
        try:

            self.history = self.model.fit(

                self.train_dataset,

                validation_data=self.validation_dataset,

                epochs=self.training_config.epochs,

                callbacks=self.callbacks,
            )

            history_file = (
                self.training_config.history_dir /
                "history.json"
            )

            with open(history_file, "w") as f:
                json.dump(self.history.history, f)

            logger.info(
                "Training history saved."
            )

            return self.history
        
        except Exception as e:

            raise CustomException(e, sys)
    
    def save_model(self):
        try:

            self.model.save_weights(
                self.training_config.model_dir /
                "final_model.weights.h5"
            )
            logger.info("Model saved successfully.")
        except Exception as e:

            raise CustomException(e, sys)