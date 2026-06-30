from image_captioning.pipeline.prediction_pipeline import PredictionPipeline

print("=" * 60)
print("Prediction Pipeline Test")
print("=" * 60)

pipeline = PredictionPipeline()

print("Pipeline initialized successfully.")

image_path = (
    "artifacts/data_ingestion/extracted/"
    "Flickr8k_Dataset/"
    "1000268201_693b08cb0e.jpg"
)

caption = pipeline.predict(image_path)

print()
print("=" * 60)
print("Generated Caption")
print("=" * 60)
print(caption)