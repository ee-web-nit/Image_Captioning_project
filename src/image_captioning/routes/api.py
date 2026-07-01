import os

from flask import Blueprint
from flask import jsonify
from flask import request

from werkzeug.utils import secure_filename

from image_captioning.pipeline.prediction_pipeline import PredictionPipeline


api = Blueprint(
    "api",
    __name__,
)

pipeline = PredictionPipeline()

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True,
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1,
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


@api.route(
    "/api/predict",
    methods=["POST"],
)
def predict_api():

    if "image" not in request.files:

        return jsonify(
            {
                "error":
                "No image uploaded."
            }
        ), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify(
            {
                "error":
                "No image selected."
            }
        ), 400

    if not allowed_file(file.filename):

        return jsonify(
            {
                "error":
                "Invalid image format."
            }
        ), 400

    filename = secure_filename(
        file.filename
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename,
    )

    file.save(
        image_path
    )

    caption, inference_time = pipeline.predict(
        image_path
    )

    return jsonify(
        {
            "caption": caption,
            "inference_time": inference_time,
        }
    )