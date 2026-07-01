from flask import Blueprint
from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
import os

from image_captioning.pipeline.prediction_pipeline import PredictionPipeline
ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
}


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )
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
    if file.filename == "":

        return render_template(
            "index.html",
            error="Please choose an image."
        )

    if not allowed_file(file.filename):

        return render_template(
            "index.html",
            error="Only JPG, JPEG and PNG images are supported."
        )

    filename = secure_filename(file.filename)

    upload_folder = "static/uploads"

    os.makedirs(
        upload_folder,
        exist_ok=True,
    )

    image_path = os.path.join(
        upload_folder,
        filename,
    )

    file.save(image_path)

    caption, inference_time = pipeline.predict(
        image_path
    )

    return render_template(
        "index.html",
        image=image_path,
        caption=caption,
        inference_time=inference_time,
    )