from image_captioning.components.model_trainer import ModelTrainer
from image_captioning.config.configuration import ConfigurationManager
from image_captioning.logger import logger


class TrainingPipeline:

    def run(self):

        config = ConfigurationManager()
        trainer = ModelTrainer(

            training_config=config.get_training_config(),

            model_config=config.get_model_config(),

            tokenizer_config=config.get_tokenizer_builder_config(),

            splitter_config=config.get_dataset_splitter_config(),

            data_config=config.get_data_ingestion_config(),
        )

        trainer.load_tokenizer()
        trainer.load_dataset()
        trainer.build_model()
        trainer.compile_model()
        trainer.create_callbacks()
        trainer.train()
        trainer.save_model()

        logger.info("Training pipeline completed.")