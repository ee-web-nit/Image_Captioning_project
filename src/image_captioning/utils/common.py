import os
import sys
import yaml

from box import ConfigBox

from image_captioning.exception import CustomException


def read_yaml(file_path: str) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox object.
    """

    try:

        with open(file_path, "r") as yaml_file:

            content = yaml.safe_load(yaml_file)

            return ConfigBox(content)

    except Exception as e:

        raise CustomException(e, sys)
    
from pathlib import Path


def create_directories(path_to_directories: list):
    """
    Creates a list of directories if they do not exist.

    Args:
        path_to_directories (list): List of directory paths.
    """

    for path in path_to_directories:
        Path(path).mkdir(parents=True, exist_ok=True)