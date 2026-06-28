from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:

    root_dir: Path

    raw_dir: Path

    extracted_dir: Path

    processed_dir: Path

    image_zip: Path

    text_zip: Path

    images_dir: Path

    captions_file: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration for the Model Training Pipeline.
    """

    root_dir: Path
    trained_model_path: Path