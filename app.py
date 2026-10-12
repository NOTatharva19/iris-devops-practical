
import joblib
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

app = Flask(__name__)

# Load the trained Iris model
model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")

# Prometheus monitoring metrics
REQUEST_COUNT = Counter(
    "iris_api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "iris_api_request_duration_seconds",
    "API request processing time",
    ["endpoint"],
)


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    REQUEST_COUNT.labels("/health", "GET", "200").inc()
    return jsonify({"status": "healthy"}), 200


# Iris flower prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    with REQUEST_LATENCY.labels("/predict").time():
        data = request.get_json(silent=True) or {}
        features = data.get("features")

        if not isinstance(features, list) or len(features) != 4:
            REQUEST_COUNT.labels("/predict", "POST", "400").inc()
            return jsonify({
                "error": "Provide exactly four numeric features."
            }), 400

        try:
            features = [float(value) for value in features]
        except (TypeError, ValueError):
            REQUEST_COUNT.labels("/predict", "POST", "400").inc()
            return jsonify({
                "error": "All four features must be numeric."
            }), 400

        prediction = int(model.predict([features])[0])
        probabilities = model.predict_proba([features])[0]

        REQUEST_COUNT.labels("/predict", "POST", "200").inc()

        return jsonify({
            "species": str(target_names[prediction]),
            "prediction": prediction,
            "confidence": round(float(max(probabilities)), 4),
        }), 200


# Prometheus metrics endpoint
@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }


# Start the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

