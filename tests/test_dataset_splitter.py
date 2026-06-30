from image_captioning.config.configuration import ConfigurationManager
from image_captioning.components.dataset_splitter import DatasetSplitter

config = ConfigurationManager().get_dataset_splitter_config()

splitter = DatasetSplitter(config)

splitter.split_dataset()

import pandas as pd

train = pd.read_csv("artifacts/dataset_splitter/train.csv")
valid = pd.read_csv("artifacts/dataset_splitter/validation.csv")
test = pd.read_csv("artifacts/dataset_splitter/test.csv")

print(train.shape)
print(valid.shape)
print(test.shape)