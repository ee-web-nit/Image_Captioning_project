from flask import Blueprint
from flask import render_template
from flask import request

from image_captioning.pipeline.prediction_pipeline import PredictionPipeline

home = Blueprint(
    "home",
    __name__,
)

pipeline = PredictionPipeline()


@home.route("/")
def index():

    return render_template(
        "index.html"
    )


@home.route(
    "/predict",
    methods=["POST"],
)
def predict():

    file = request.files["image"]

    image_path = (
        "static/uploads/" +
        file.filename
    )

    file.save(image_path)

    caption = pipeline.predict(
        image_path
    )

    return render_template(

        "index.html",

        image=image_path,

        caption=caption,
    )