from pathlib import Path

from image_captioning.entity.config_entity import DataIngestionConfig


config = DataIngestionConfig(

    root_dir=Path("artifacts"),

    dataset_dir=Path("dataset"),

    images_dir=Path("dataset/images"),

    captions_file=Path("captions.txt")

)

print(config)