"""
Vercel Python Serverless Function for real-time fraud prediction.
POST /api/predict
Body: { "dataset": "creditcard"|"paysim", "features": { "V1": ..., ... } }
"""

import json
import time
import os
import numpy as np

from http.server import BaseHTTPRequestHandler

# Lazy-load models to avoid cold start overhead on every import
_models = {}


def _get_models_dir():
    """Find models directory."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "models"),
        os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        "/var/task/models",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def _load_model(dataset_tag):
    """Load and cache models for a dataset."""
    if dataset_tag in _models:
        return _models[dataset_tag]

    models_dir = _get_models_dir()

    import joblib
    import xgboost as xgb

    # Load scaler
    scaler = joblib.load(os.path.join(models_dir, f"scaler_{dataset_tag}.pkl"))

    # Load XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(models_dir, f"xgboost_{dataset_tag}.json"))

    # Load Isolation Forest
    iforest = joblib.load(os.path.join(models_dir, f"iforest_{dataset_tag}.pkl"))

    # Load ensemble config
    with open(os.path.join(models_dir, f"ensemble_{dataset_tag}.json")) as f:
        ensemble_config = json.load(f)

    # Try to load autoencoder (ONNX)
    ae_session = None
    ae_input_name = None
    ae_threshold = 0
    try:
        import onnxruntime as ort
        onnx_path = os.path.join(models_dir, f"autoencoder_{dataset_tag}.onnx")
        if os.path.exists(onnx_path):
            ae_session = ort.InferenceSession(onnx_path)
            ae_input_name = ae_session.get_inputs()[0].name
            ae_config_path = os.path.join(models_dir, f"autoencoder_{dataset_tag}_config.json")
            with open(ae_config_path) as f:
                ae_threshold = json.load(f)["threshold"]
    except ImportError:
        pass

    result = {
        "scaler": scaler,
        "xgboost": xgb_model,
        "iforest": iforest,
        "ae_session": ae_session,
        "ae_input_name": ae_input_name,
        "ae_threshold": ae_threshold,
        "ensemble_config": ensemble_config,
    }
    _models[dataset_tag] = result
    return result


def predict(dataset_tag, features_dict):
    """Run ensemble prediction on a single transaction."""
    start = time.time()

    m = _load_model(dataset_tag)
    scaler = m["scaler"]
    ensemble_config = m["ensemble_config"]
    weights = ensemble_config["weights"]

    # Build feature array in correct order
    feature_names = scaler.feature_names_in_
    values = np.array([[features_dict.get(f, 0.0) for f in feature_names]], dtype=np.float64)
    scaled = scaler.transform(values)

    # XGBoost score
    xgb_prob = float(m["xgboost"].predict_proba(scaled)[0, 1])

    # Isolation Forest score
    if_score = float(-m["iforest"].decision_function(scaled)[0])
    if_range = ensemble_config["iforest_score_range"]
    if_norm = (if_score - if_range["min"]) / (if_range["max"] - if_range["min"] + 1e-10)
    if_norm = max(0, min(1, if_norm))

    # Autoencoder score
    ae_norm = 0.0
    ae_raw = 0.0
    if m["ae_session"] is not None:
        recon = m["ae_session"].run(None, {m["ae_input_name"]: scaled.astype(np.float32)})[0]
        ae_raw = float(np.mean(np.square(scaled - recon)))
        ae_range = ensemble_config["ae_score_range"]
        ae_norm = (ae_raw - ae_range["min"]) / (ae_range["max"] - ae_range["min"] + 1e-10)
        ae_norm = max(0, min(1, ae_norm))

    # Ensemble
    ensemble_score = (
        weights["xgboost"] * xgb_prob +
        weights["isolation_forest"] * if_norm +
        weights["autoencoder"] * ae_norm
    )

    threshold = ensemble_config["threshold"]
    is_fraud = ensemble_score >= threshold

    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "fraud": bool(is_fraud),
        "ensemble_score": round(float(ensemble_score), 4),
        "scores": {
            "XGBoost": round(xgb_prob, 4),
            "Isolation Forest": round(float(if_norm), 4),
            "Autoencoder": round(float(ae_norm), 4),
        },
        "threshold": round(float(threshold), 4),
        "latency_ms": latency_ms,
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            dataset = body.get("dataset", "creditcard")
            dataset_tag = "cc" if dataset == "creditcard" else "ps"
            features = body.get("features", {})

            result = predict(dataset_tag, features)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "endpoint": "POST /api/predict"}).encode())
