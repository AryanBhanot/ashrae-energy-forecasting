from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Loaded once at startup, not on every request
model = joblib.load("../models/model.pkl")

FEATURES = ["hour", "dow", "month", "lag_24h", "lag_168h"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {f: float(request.form[f]) for f in FEATURES}
    except (KeyError, ValueError):
        return render_template(
            "index.html", prediction=None,
            error="Please fill in all five fields with numbers."
        )

    X = pd.DataFrame([values], columns=FEATURES)
    pred = model.predict(X)[0]

    return render_template("index.html", prediction=round(float(pred), 2), error=None)


if __name__ == "__main__":
    # host=0.0.0.0 is what makes it reachable from outside the VM
    app.run(host="0.0.0.0", port=5000)
