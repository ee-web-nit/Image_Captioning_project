from image_captioning.config.configuration import ConfigurationManager


class TrainingPipeline:

    def __init__(self):

        self.config_manager = ConfigurationManager()

    def run(self):

        print("=" * 60)
        print("IMAGE CAPTIONING TRAINING PIPELINE")
        print("=" * 60)

        self.load_configuration()

        print("Pipeline Initialized Successfully.")

    def load_configuration(self):

        print("Loading configurations...")

        self.model_config = self.config_manager.get_model_config()

        print("Configurations loaded successfully.")
        
    def load_tokenizer(self):
        pass

    def prepare_dataset(self):
        pass

    def build_model(self):
        pass

    def compile_model(self):
        pass

    def train_model(self):
        pass

    def save_model(self):
        pass