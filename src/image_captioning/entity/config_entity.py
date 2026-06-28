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
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    images_dir: Path
    captions_file: Path
    train_split: Path
    validation_split: Path
    test_split: Path
    status_file: Path

@dataclass(frozen=True)
class CaptionParserConfig:
    root_dir: Path

    captions_file: Path

    output_file: Path

@dataclass(frozen=True)
class CaptionCleanerConfig:
    root_dir: Path
    input_file: Path
    output_file: Path

@dataclass(frozen=True)
class DatasetSplitterConfig:
    root_dir: Path

    captions_file: Path

    train_images: Path
    validation_images: Path
    test_images: Path

    train_output: Path
    validation_output: Path
    test_output: Path