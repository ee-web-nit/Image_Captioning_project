from image_captioning.utils.image_preprocessor import ImagePreprocessor

image = ImagePreprocessor.preprocess(
    "artifacts/data_ingestion/extracted/Flicker8k_Dataset/1000268201_693b08cb0e.jpg"
)

print(image.shape)