from image_captioning.components.model_trainer import ModelTrainer
from image_captioning.config.configuration import ConfigurationManager


class TrainingPipeline:

    def run(self):

        config = ConfigurationManager()

        training_config = config.get_training_config()

        trainer = ModelTrainer(training_config)

        trainer.load_tokenizer()
        trainer.load_dataset()
        trainer.build_model()
        trainer.compile_model()
        trainer.create_callbacks()
        trainer.train()
        trainer.save_model()

        print("Training pipeline completed.")