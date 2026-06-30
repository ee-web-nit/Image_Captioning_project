from image_captioning.components.dataset_builder import DatasetBuilder

builder = DatasetBuilder(batch_size=2)

dataset = builder.build(
    caption_file="artifacts/caption_cleaner/cleaned_captions.csv",
    image_dir="artifacts/data_ingestion/extracted/Flicker8k_Dataset",
)

for images, captions in dataset.take(1):

    print(images.shape)

    print(captions.shape)

    print(captions[0])