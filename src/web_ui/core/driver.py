import os

from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return "<p>Hello world<p>"


if __name__ == "__main__":
    port = int(os.environ.get("WEB_APP_PORT", 50001))
    app.run(debug=True, host="0.0.0.0", port=port)
