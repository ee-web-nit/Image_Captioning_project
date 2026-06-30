import pandas as pd

train = pd.read_csv("artifacts/dataset_splitter/train.csv")
val = pd.read_csv("artifacts/dataset_splitter/validation.csv")
test = pd.read_csv("artifacts/dataset_splitter/test.csv")

print("Train rows:", len(train))
print("Validation rows:", len(val))
print("Test rows:", len(test))

print("Unique train images:", train["image_name"].nunique())
print("Unique validation images:", val["image_name"].nunique())
print("Unique test images:", test["image_name"].nunique())