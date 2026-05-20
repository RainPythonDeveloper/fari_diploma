"""
Vercel Python Serverless Function for real-time fraud prediction (NEW ENSEMBLE).
POST /api/predict
Body: { "dataset": "creditcard"|"paysim", "features": { "V1": ..., ... } }

Ensemble = mean of min-max normalized scores from XGBoost + Isolation Forest + Autoencoder.
Each model's training-time score range is stored in models/ensemble_{tag}.json so
single-sample inference can replicate the same normalization used at training.

Two scalers per dataset:
  - scaler_sup_{tag}.pkl  -> applied to XGBoost input
  - scaler_unsup_{tag}.pkl -> applied to Isolation Forest + Autoencoder input
"""

import json
import time
import os
import numpy as np

from http.server import BaseHTTPRequestHandler

_models = {}


def _get_models_dir():
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
    if dataset_tag in _models:
        return _models[dataset_tag]

    models_dir = _get_models_dir()

    import joblib
    import xgboost as xgb

    scaler_sup = joblib.load(os.path.join(models_dir, f"scaler_sup_{dataset_tag}.pkl"))
    scaler_unsup = joblib.load(os.path.join(models_dir, f"scaler_unsup_{dataset_tag}.pkl"))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(models_dir, f"xgboost_{dataset_tag}.json"))

    iforest = joblib.load(os.path.join(models_dir, f"iforest_{dataset_tag}.pkl"))

    with open(os.path.join(models_dir, f"ensemble_{dataset_tag}.json")) as f:
        ensemble_config = json.load(f)

    ae_session = None
    ae_input_name = None
    ae_threshold = 0.0
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
        "scaler_sup": scaler_sup,
        "scaler_unsup": scaler_unsup,
        "xgboost": xgb_model,
        "iforest": iforest,
        "ae_session": ae_session,
        "ae_input_name": ae_input_name,
        "ae_threshold": ae_threshold,
        "ensemble_config": ensemble_config,
    }
    _models[dataset_tag] = result
    return result


def _norm(value, rng):
    span = rng["max"] - rng["min"]
    if span <= 0:
        return 0.0
    out = (value - rng["min"]) / span
    return max(0.0, min(1.0, out))


def predict(dataset_tag, features_dict):
    """Run new-logic ensemble prediction on a single transaction."""
    start = time.time()

    m = _load_model(dataset_tag)
    scaler_sup = m["scaler_sup"]
    scaler_unsup = m["scaler_unsup"]
    ensemble_config = m["ensemble_config"]

    # Build feature array in correct order (both scalers were fit on the same feature set)
    feature_names = scaler_sup.feature_names_in_
    values = np.array([[features_dict.get(f, 0.0) for f in feature_names]], dtype=np.float64)

    scaled_sup = scaler_sup.transform(values)
    scaled_unsup = scaler_unsup.transform(values)

    # XGBoost (supervised) -> raw probability
    xgb_prob = float(m["xgboost"].predict_proba(scaled_sup)[0, 1])

    # Per-transaction SHAP via XGBoost built-in
    import xgboost as xgb
    dmatrix = xgb.DMatrix(scaled_sup, feature_names=list(feature_names))
    contribs = m["xgboost"].get_booster().predict(dmatrix, pred_contribs=True)
    shap_dict = {
        str(feature_names[i]): round(float(contribs[0][i]), 4)
        for i in range(len(feature_names))
    }

    # Isolation Forest (unsupervised) -> negated decision function
    if_score = float(-m["iforest"].decision_function(scaled_unsup)[0])

    # Autoencoder (unsupervised) -> reconstruction error
    ae_score = 0.0
    if m["ae_session"] is not None:
        recon = m["ae_session"].run(None, {m["ae_input_name"]: scaled_unsup.astype(np.float32)})[0]
        ae_score = float(np.mean(np.square(scaled_unsup - recon)))

    # Apply training-time min-max normalization to each raw score
    xgb_norm = _norm(xgb_prob, ensemble_config["xgb_score_range"])
    if_norm = _norm(if_score, ensemble_config["iforest_score_range"])
    ae_norm = _norm(ae_score, ensemble_config["ae_score_range"])

    # Ensemble = simple mean
    ensemble_score = (xgb_norm + if_norm + ae_norm) / 3.0
    threshold = ensemble_config["threshold"]
    is_fraud = ensemble_score >= threshold

    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "fraud": bool(is_fraud),
        "ensemble_score": round(float(ensemble_score), 4),
        "scores": {
            "XGBoost": round(xgb_norm, 4),
            "Isolation Forest": round(if_norm, 4),
            "Autoencoder": round(ae_norm, 4),
        },
        "raw_scores": {
            "XGBoost": round(xgb_prob, 4),
            "Isolation Forest": round(if_score, 4),
            "Autoencoder": round(ae_score, 4),
        },
        "threshold": round(float(threshold), 4),
        "latency_ms": latency_ms,
        "shap_values": shap_dict,
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
