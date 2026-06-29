from image_captioning.pipeline.training_pipeline import TrainingPipeline


def main():

    print("=" * 70)
    print(" IMAGE CAPTIONING PROJECT ")
    print("=" * 70)

    pipeline = TrainingPipeline()

    pipeline.run()


if __name__ == "__main__":
    main()