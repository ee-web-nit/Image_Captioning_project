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