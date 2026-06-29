import tensorflow as tf

from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.dataset_builder import DatasetBuilder
from image_captioning.models.image_captioning_model import ImageCaptioningModel


def main():

    print("=" * 60)
    print(" IMAGE CAPTIONING MODEL TRAINING ")
    print("=" * 60)

    # -------------------------------------------------
    # Load Configuration
    # -------------------------------------------------

    config_manager = ConfigurationManager()

    model_config = config_manager.get_model_config()

    # -------------------------------------------------
    # Build Dataset
    # -------------------------------------------------

    builder = DatasetBuilder(
        batch_size=model_config.batch_size,
        sequence_length=model_config.sequence_length,
        vocab_size=model_config.vocab_size,
    )

    dataset = builder.build(
        caption_file="artifacts/caption_cleaner/cleaned_captions.csv",
        image_dir="artifacts/data_ingestion/extracted/Flicker8k_Dataset",
    )

    print("Dataset Loaded Successfully.")

    # -------------------------------------------------
    # Build Model
    # -------------------------------------------------

    model = ImageCaptioningModel(model_config)

    # -------------------------------------------------
    # Compile Model
    # -------------------------------------------------

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction="none",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss_fn=loss_fn,
    )

    print("Model Compiled Successfully.")

    # -------------------------------------------------
    # Temporary Training
    # -------------------------------------------------

    print("Starting training...")

    history = model.fit(
        dataset,
        epochs=1,
    )

    print(history.history)


if __name__ == "__main__":
    main()