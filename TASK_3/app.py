"""
TASK 3: Deploy a Model as a Simple Web API
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load files relative to this script's own location, so it works
# no matter which folder you run "python app.py" from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "iris_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "iris_scaler.joblib"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "iris_label_encoder.joblib"))

FEATURE_ORDER = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris classification API is running.",
        "usage": "POST /predict with JSON body: "
                 "{sepal_length, sepal_width, petal_length, petal_width}"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        features = [float(data[f]) for f in FEATURE_ORDER]
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric."}), 400

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred_class = model.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    probabilities = model.predict_proba(X_scaled)[0]
    prob_dict = {
        label: round(float(prob), 4)
        for label, prob in zip(label_encoder.classes_, probabilities)
    }

    return jsonify({
        "input": data,
        "prediction": pred_label,
        "probabilities": prob_dict
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)