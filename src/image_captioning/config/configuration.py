from pathlib import Path

from image_captioning.utils.common import read_yaml

from image_captioning.entity.config_entity import (
    DataIngestionConfig,
    ModelTrainerConfig,
    DataValidationConfig,
    CaptionParserConfig,
    CaptionCleanerConfig,
    DatasetSplitterConfig,
    VocabularyBuilderConfig,
    TokenizerBuilderConfig,
    ModelConfig,
)

from image_captioning.utils.common import (
    read_yaml,
    create_directories,
)

from image_captioning.entity.training_config import TrainingConfig


class ConfigurationManager:

    def __init__(
        self,
        config_filepath=Path("config/config.yaml"),
        params_filepath=Path("config/params.yaml"),
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns the configuration required for the Data Ingestion Pipeline.
        """

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            raw_dir=Path(config.raw_dir),
            extracted_dir=Path(config.extracted_dir),
            processed_dir=Path(config.processed_dir),
            image_zip=Path(config.image_zip),
            text_zip=Path(config.text_zip),
            images_dir=Path(config.images_dir),
            captions_file=Path(config.captions_file),
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            images_dir=Path(config.images_dir),
            captions_file=Path(config.captions_file),
            train_split=Path(config.train_split),
            validation_split=Path(config.validation_split),
            test_split=Path(config.test_split),
            status_file=Path(config.status_file),
        )

        return data_validation_config

    def get_caption_parser_config(self) -> CaptionParserConfig:

        config = self.config.caption_parser

        create_directories([config.root_dir])

        caption_parser_config = CaptionParserConfig(
            root_dir=Path(config.root_dir),
            captions_file=Path(config.captions_file),
            output_file=Path(config.output_file),
        )

        return caption_parser_config

    def get_caption_cleaner_config(self) -> CaptionCleanerConfig:

        config = self.config.caption_cleaner

        create_directories([config.root_dir])

        return CaptionCleanerConfig(
            root_dir=Path(config.root_dir),
            input_file=Path(config.input_file),
            output_file=Path(config.output_file),
        )
    
    def get_dataset_splitter_config(self) -> DatasetSplitterConfig:

        config = self.config.dataset_splitter

        create_directories([config.root_dir])

        return DatasetSplitterConfig(
            root_dir=Path(config.root_dir),
            captions_file=Path(config.captions_file),
            train_images=Path(config.train_images),
            validation_images=Path(config.validation_images),
            test_images=Path(config.test_images),
            train_output=Path(config.train_output),
            validation_output=Path(config.validation_output),
            test_output=Path(config.test_output),
        )
    
    def get_vocabulary_builder_config(self):

        config = self.config.vocabulary_builder

        create_directories([config.root_dir])

        return VocabularyBuilderConfig(
            root_dir=Path(config.root_dir),
            input_file=Path(config.input_file),
            vocabulary_file=Path(config.vocabulary_file),
            max_vocabulary_size=config.max_vocabulary_size,
        )

    def get_tokenizer_builder_config(self):

        config = self.config.tokenizer_builder

        create_directories([config.root_dir])

        return TokenizerBuilderConfig(
            root_dir=Path(config.root_dir),
            vocabulary_file=Path(config.vocabulary_file),
            tokenizer_path=Path(config.tokenizer_path),
            max_tokens=config.max_tokens,
            sequence_length=config.sequence_length,
        )
    def get_model_config(self) -> ModelConfig:

        config = self.config.model

        model_config = ModelConfig(
            image_size=config.image_size,
            embed_dim=config.embed_dim,
            ff_dim=config.ff_dim,
            num_heads=config.num_heads,
            vocab_size=config.vocab_size,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            epochs=config.epochs,
        )

        return model_config
    
    def get_training_config(self):

        cfg = self.config.training

        mode = cfg.mode

        if mode == "local":
            run_cfg = cfg.local
        else:
            run_cfg = cfg.cloud

        return TrainingConfig(

            mode=mode,

            epochs=run_cfg.epochs,

            batch_size=run_cfg.batch_size,

            shuffle_buffer=run_cfg.shuffle_buffer,

            learning_rate=cfg.learning_rate,

            checkpoint_dir=Path(cfg.checkpoint_dir),

            model_dir=Path(cfg.model_dir),

            tensorboard_dir=Path(cfg.tensorboard_dir),

            history_dir=Path(cfg.history_dir),


        )