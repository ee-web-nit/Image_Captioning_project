from flask import Flask

from image_captioning.routes.home import home

from image_captioning.routes.api import api


app = Flask(__name__)

app.register_blueprint(home)
app.register_blueprint(api)


if __name__ == "__main__":

    app.run(
        debug=True,
    )