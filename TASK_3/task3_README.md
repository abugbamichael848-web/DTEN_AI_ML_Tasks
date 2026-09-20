# Task 3: Deploy a Model as a Simple Web API

## Overview
The Task 1 Iris classification model (Logistic Regression) is wrapped in a lightweight
**Flask** API with a single `/predict` endpoint that accepts flower measurements and returns
the predicted species.

## Files
- `train_and_save_model.py` — retrains the Task 1 model and saves it (`iris_model.joblib`,
  `iris_scaler.joblib`, `iris_label_encoder.joblib`) so the API can load it without retraining
- `app.py` — the Flask API
- `iris_model.joblib`, `iris_scaler.joblib`, `iris_label_encoder.joblib` — saved model artifacts
- `Iris_kaggle.csv` — dataset used to train the model

## Setup & Run

```bash
pip install flask joblib scikit-learn pandas numpy

# (Re)train and save the model — only needed once, or if you want to retrain
python train_and_save_model.py

# Start the API
python app.py
```

The server runs at `http://127.0.0.1:5000`.

## Endpoint

### `GET /`
Health check / usage info.

**Example:**
```bash
curl http://127.0.0.1:5000/
```
**Response:**
```json
{
  "message": "Iris classification API is running.",
  "usage": "POST /predict with JSON body: {sepal_length, sepal_width, petal_length, petal_width}"
}
```

### `POST /predict`
Accepts flower measurements, returns the predicted species and class probabilities.

**Request body (JSON):**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Example requests & responses (tested locally with curl):**

**1. Setosa-like input**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
```json
{
  "input": {"petal_length": 1.4, "petal_width": 0.2, "sepal_length": 5.1, "sepal_width": 3.5},
  "prediction": "Iris-setosa",
  "probabilities": {"Iris-setosa": 0.9813, "Iris-versicolor": 0.0187, "Iris-virginica": 0.0}
}
```

**2. Virginica-like input**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}'
```
```json
{
  "input": {"petal_length": 5.2, "petal_width": 2.3, "sepal_length": 6.7, "sepal_width": 3.0},
  "prediction": "Iris-virginica",
  "probabilities": {"Iris-setosa": 0.0001, "Iris-versicolor": 0.0438, "Iris-virginica": 0.9561}
}
```

**3. Versicolor-like input**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.9, "sepal_width": 2.8, "petal_length": 4.3, "petal_width": 1.3}'
```
```json
{
  "input": {"petal_length": 4.3, "petal_width": 1.3, "sepal_length": 5.9, "sepal_width": 2.8},
  "prediction": "Iris-versicolor",
  "probabilities": {"Iris-setosa": 0.0222, "Iris-versicolor": 0.8855, "Iris-virginica": 0.0923}
}
```

**4. Error handling — missing field**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}'
```
```json
{
  "error": "Missing fields: ['petal_width']"
}
```

## Testing with Postman
1. Set method to `POST`, URL to `http://127.0.0.1:5000/predict`.
2. Body → raw → JSON, paste one of the example request bodies above.
3. Send — you should get the same JSON response shown above.

## Notes / Limitations
- This runs Flask's built-in development server (`debug=True`), which is fine for local
  testing but **not suitable for production** — a production deployment would use a WSGI
  server like Gunicorn behind a reverse proxy (e.g. Nginx).
- No authentication or rate-limiting is implemented — anyone who can reach the server can
  call the endpoint.
- Input validation only checks that required fields are present and numeric; it does not
  check that values are within realistic biological ranges (e.g. negative lengths would still
  be accepted and passed to the model).
