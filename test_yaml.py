from image_captioning.utils.common import read_yaml

config = read_yaml("config/config.yaml")

print(config.artifacts_root)