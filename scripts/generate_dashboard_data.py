"""
Generate all JSON data for the dashboard.
Loads trained models from ../models/ and generates JSON files in ../frontend/public/data/

Usage:
    python generate_dashboard_data.py
"""

import os
import sys
import json
import warnings
import random

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "frontend", "public", "data")

# ---------------------------------------------------------------------------
# Reuse loaders from train_models.py
# ---------------------------------------------------------------------------

def find_file(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_creditcard():
    path = find_file([
        os.path.join(BASE_DIR, "..", "creditcard.csv"),
        "creditcard.csv",
        "/content/Diploma_files/creditcard.csv",
    ])
    if path is None:
        sys.exit("creditcard.csv not found")
    df = pd.read_csv(path).dropna().drop_duplicates().reset_index(drop=True)
    return df


def load_paysim():
    path = find_file([
        os.path.join(BASE_DIR, "..", "PS_20174392719_1491204439457_log.csv"),
        "PS_20174392719_1491204439457_log.csv",
        "/content/Diploma_files/PS_20174392719_1491204439457_log.csv",
    ])
    if path is None:
        sys.exit("PaySim CSV not found")
    df = pd.read_csv(path).dropna().drop_duplicates().reset_index(drop=True)
    df = df.rename(columns={"isFraud": "Class"})
    if len(df) > 500_000:
        fraud = df[df["Class"] == 1]
        normal = df[df["Class"] == 0].sample(n=500_000 - len(fraud), random_state=RANDOM_STATE)
        df = pd.concat([normal, fraud]).reset_index(drop=True)
    return df


def engineer_creditcard(df):
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] % 86400) / 3600
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    features = df.drop(columns=["Class", "Time", "Amount", "Hour"])
    return features, df["Class"], df


def engineer_paysim(df):
    df = df.copy()
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["error_orig"] = df["balance_diff_orig"] - df["amount"]
    df["error_dest"] = df["balance_diff_dest"] - df["amount"]
    df["amount_log"] = np.log1p(df["amount"])
    df["type_code"] = df["type"].astype("category").cat.codes
    drop_cols = ["Class", "type", "nameOrig", "nameDest", "amount"]
    if "isFlaggedFraud" in df.columns:
        drop_cols.append("isFlaggedFraud")
    features = df.drop(columns=drop_cols)
    return features, df["Class"], df


def save_json(data, *path_parts):
    path = os.path.join(DATA_DIR, *path_parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved {path}")

# ---------------------------------------------------------------------------
# Generate data for one dataset
# ---------------------------------------------------------------------------

def generate_dataset_data(df_raw, X, y, dataset_tag, dataset_name):
    print(f"\nGenerating data for {dataset_name}...")

    contamination = float(np.clip(y.mean(), 1e-4, 0.05))

    # Split (same as training)
    X_normal = X[y == 0]
    X_fraud = X[y == 1]
    X_train_unsup, X_test_normal = train_test_split(X_normal, test_size=0.2, random_state=RANDOM_STATE)
    X_test = pd.concat([X_test_normal, X_fraud], axis=0)
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])

    X_train_sup, X_test_sup, y_train_sup, y_test_sup = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Load scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, f"scaler_{dataset_tag}.pkl"))
    X_test_scaled = scaler.transform(X_test)

    # Load models and get predictions
    models_data = {}

    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, f"xgboost_{dataset_tag}.json"))
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    models_data["XGBoost"] = {"preds": xgb_preds, "scores": xgb_probs}

    # Global SHAP feature importance for XGBoost
    _shap_data = []
    try:
        import shap as _shap_lib
        _explainer = _shap_lib.TreeExplainer(xgb_model)
        _shap_n = min(200, len(X_test_scaled))
        _shap_idx = np.random.choice(len(X_test_scaled), size=_shap_n, replace=False)
        _shap_vals = _explainer.shap_values(X_test_scaled[_shap_idx])
        _shap_mean_abs = np.mean(np.abs(_shap_vals), axis=0)
        _shap_mean = np.mean(_shap_vals, axis=0)
        _feat_names_shap = list(scaler.feature_names_in_)
        _shap_data = sorted([
            {
                "name": _feat_names_shap[i],
                "mean_abs_shap": round(float(_shap_mean_abs[i]), 4),
                "mean_shap": round(float(_shap_mean[i]), 4),
            }
            for i in range(len(_feat_names_shap))
        ], key=lambda x: -x["mean_abs_shap"])
    except ImportError:
        print("  shap not installed — skipping global SHAP. Run: pip install shap")

    # Isolation Forest
    iforest = joblib.load(os.path.join(MODELS_DIR, f"iforest_{dataset_tag}.pkl"))
    if_raw = iforest.predict(X_test_scaled)
    if_preds = np.where(if_raw == -1, 1, 0)
    if_scores = -iforest.decision_function(X_test_scaled)
    models_data["Isolation Forest"] = {"preds": if_preds, "scores": if_scores}

    # Autoencoder (via ONNX or Keras)
    ae_scores = None
    try:
        import onnxruntime as ort
        onnx_path = os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}.onnx")
        if os.path.exists(onnx_path):
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            ae_recon = sess.run(None, {input_name: X_test_scaled.astype(np.float32)})[0]
            ae_scores = np.mean(np.square(X_test_scaled - ae_recon), axis=1)
    except ImportError:
        pass

    if ae_scores is None:
        try:
            import tensorflow as tf
            keras_path = os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}.keras")
            if os.path.exists(keras_path):
                ae_model = tf.keras.models.load_model(keras_path)
                ae_recon = ae_model.predict(X_test_scaled, verbose=0)
                ae_scores = np.mean(np.square(X_test_scaled - ae_recon), axis=1)
        except ImportError:
            pass

    if ae_scores is not None:
        ae_config = json.load(open(os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}_config.json")))
        ae_preds = np.where(ae_scores > ae_config["threshold"], 1, 0)
        models_data["Autoencoder"] = {"preds": ae_preds, "scores": ae_scores}

    # Ensemble
    ensemble_config = json.load(open(os.path.join(MODELS_DIR, f"ensemble_{dataset_tag}.json")))
    w = ensemble_config["weights"]
    if_range = ensemble_config["iforest_score_range"]
    ae_range = ensemble_config["ae_score_range"]

    xgb_norm = xgb_probs
    iforest_norm = (if_scores - if_range["min"]) / (if_range["max"] - if_range["min"] + 1e-10)
    ae_norm = np.zeros_like(xgb_probs)
    if ae_scores is not None:
        ae_norm = (ae_scores - ae_range["min"]) / (ae_range["max"] - ae_range["min"] + 1e-10)

    ensemble_scores = w["xgboost"] * xgb_norm + w["isolation_forest"] * iforest_norm + w["autoencoder"] * ae_norm
    ensemble_preds = (ensemble_scores >= ensemble_config["threshold"]).astype(int)
    models_data["Ensemble"] = {"preds": ensemble_preds, "scores": ensemble_scores}

    folder = "creditcard" if dataset_tag == "cc" else "paysim"

    # --- summary.json ---
    save_json({
        "name": dataset_name,
        "total_samples": int(len(df_raw)),
        "normal": int((y == 0).sum()),
        "fraud": int((y == 1).sum()),
        "fraud_rate": round(float(y.mean()) * 100, 4),
        "features_count": int(X.shape[1]),
        "test_samples": int(len(y_test)),
        "test_fraud": int(y_test.sum()),
        "contamination": contamination,
    }, folder, "summary.json")

    # --- model_results.json ---
    model_results = []
    for model_name, mdata in models_data.items():
        preds = mdata["preds"]
        scores = mdata["scores"]
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        model_results.append({
            "model": model_name,
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
            "pr_auc": round(float(average_precision_score(y_test, scores)), 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })
    model_results.sort(key=lambda x: -x["f1"])
    for i, r in enumerate(model_results):
        r["rank"] = i + 1
    save_json(model_results, folder, "model_results.json")

    # --- confusion_matrices.json ---
    cm_data = {}
    for model_name, mdata in models_data.items():
        tn, fp, fn, tp = confusion_matrix(y_test, mdata["preds"]).ravel()
        cm_data[model_name] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    save_json(cm_data, folder, "confusion_matrices.json")

    # --- roc_curves.json ---
    roc_data = {}
    for model_name, mdata in models_data.items():
        fpr, tpr, thresh_roc = roc_curve(y_test, mdata["scores"])
        # Downsample to ~200 points for JSON size
        step = max(1, len(fpr) // 200)
        roc_data[model_name] = {
            "fpr": [round(float(v), 4) for v in fpr[::step]],
            "tpr": [round(float(v), 4) for v in tpr[::step]],
            "auc": round(float(roc_auc_score(y_test, mdata["scores"])), 4),
            "thresholds": [round(float(v), 4) for v in thresh_roc[::step]],
        }
    save_json(roc_data, folder, "roc_curves.json")

    # --- pr_curves.json ---
    pr_data = {}
    for model_name, mdata in models_data.items():
        prec, rec, thresh_pr = precision_recall_curve(y_test, mdata["scores"])
        # prec/rec have one extra element vs thresholds; align them
        prec_a, rec_a = prec[:-1], rec[:-1]
        step = max(1, len(prec_a) // 200)
        pr_data[model_name] = {
            "precision": [round(float(v), 4) for v in prec_a[::step]],
            "recall": [round(float(v), 4) for v in rec_a[::step]],
            "ap": round(float(average_precision_score(y_test, mdata["scores"])), 4),
            "thresholds": [round(float(v), 4) for v in thresh_pr[::step]],
        }
    save_json(pr_data, folder, "pr_curves.json")

    # --- transactions.json (~5000 rows) ---
    # All fraud + sample of normal
    fraud_idx = np.where(y_test == 1)[0]
    normal_idx = np.where(y_test == 0)[0]
    sample_normal = np.random.choice(normal_idx, size=min(4500, len(normal_idx)), replace=False)
    sample_idx = np.concatenate([fraud_idx, sample_normal])
    sample_idx.sort()

    X_test_reset = X_test.reset_index(drop=True)
    transactions = []
    for i in sample_idx:
        row = {"is_fraud": int(y_test[i]), "index": int(i)}
        # Add readable features
        if dataset_tag == "cc":
            row["amount"] = round(float(np.expm1(X_test_reset.iloc[i].get("Amount_log", 0))), 2)
        else:
            row["amount"] = round(float(np.expm1(X_test_reset.iloc[i].get("amount_log", 0))), 2)
            for col in ["step", "type_code", "balance_diff_orig", "balance_diff_dest", "error_orig", "error_dest"]:
                if col in X_test_reset.columns:
                    row[col] = round(float(X_test_reset.iloc[i][col]), 2)

        # Add model scores
        scores = {}
        for model_name, mdata in models_data.items():
            scores[model_name] = round(float(mdata["scores"][i]), 4)
        row["scores"] = scores
        row["ensemble_prediction"] = int(models_data["Ensemble"]["preds"][i])
        transactions.append(row)

    save_json(transactions, folder, "transactions.json")

    # --- distributions.json ---
    if dataset_tag == "cc":
        normal_amounts = df_raw[df_raw["Class"] == 0]["Amount"].values
        fraud_amounts = df_raw[df_raw["Class"] == 1]["Amount"].values
        normal_times = (df_raw[df_raw["Class"] == 0]["Time"] / 3600).values
        fraud_times = (df_raw[df_raw["Class"] == 1]["Time"] / 3600).values

        save_json({
            "amount": {
                "normal": _hist_to_json(normal_amounts, 50),
                "fraud": _hist_to_json(fraud_amounts, 50),
            },
            "time": {
                "normal": _hist_to_json(normal_times, 48),
                "fraud": _hist_to_json(fraud_times, 48),
            }
        }, folder, "distributions.json")
    else:
        normal_amounts = np.log1p(df_raw[df_raw["Class"] == 0]["amount"].values)
        fraud_amounts = np.log1p(df_raw[df_raw["Class"] == 1]["amount"].values)

        # Fraud by transaction type
        fraud_by_type = df_raw[df_raw["Class"] == 1]["type"].value_counts().to_dict()

        save_json({
            "amount": {
                "normal": _hist_to_json(normal_amounts, 50),
                "fraud": _hist_to_json(fraud_amounts, 50),
            },
            "fraud_by_type": fraud_by_type,
        }, folder, "distributions.json")

    # --- training_history.json (autoencoder) ---
    training_results_path = os.path.join(MODELS_DIR, "training_results.json")
    if os.path.exists(training_results_path):
        with open(training_results_path) as f:
            tr = json.load(f)
        # Read autoencoder history if saved during training
        # (the history is in the training script's results, we'll regenerate placeholder)
    save_json({"note": "Run train_models.py to generate autoencoder loss history"}, folder, "training_history.json")

    # --- hyperparameters.json ---
    hyperparams = [
        {"model": "XGBoost", "params": f"max_depth=6, n_estimators=200, scale_pos_weight={round(float((y==0).sum()/(y==1).sum()), 1)}"},
        {"model": "Isolation Forest", "params": f"n_estimators=300, max_samples=0.8, contamination={contamination:.6f}"},
        {"model": "Autoencoder", "params": "arch=64-32-14-32-64, epochs=100, batch=2048, dropout=0.2-0.3"},
        {"model": "Ensemble", "params": f"weights: XGB=0.6, IF=0.2, AE=0.2, threshold={ensemble_config['threshold']:.2f}"},
    ]
    save_json(hyperparams, folder, "hyperparameters.json")

    # --- sample_transactions.json (for predict page) ---
    # Save a few real fraud and normal examples
    samples = {"fraud": [], "normal": []}
    fraud_examples = X_test_reset[y_test == 1].head(5)
    normal_examples = X_test_reset[y_test == 0].head(5)
    for _, row in fraud_examples.iterrows():
        samples["fraud"].append({col: round(float(row[col]), 6) for col in X_test_reset.columns})
    for _, row in normal_examples.iterrows():
        samples["normal"].append({col: round(float(row[col]), 6) for col in X_test_reset.columns})
    save_json(samples, folder, "sample_transactions.json")

    # --- shap_values.json ---
    if _shap_data:
        save_json(_shap_data, folder, "shap_values.json")

    # --- feature_analysis.json (Cohen's d per feature) ---
    feat_cols = list(X.columns)
    X_arr = X.values
    fraud_mask = (y == 1).values
    normal_mask = (y == 0).values
    feat_analysis = []
    for _i, _fname in enumerate(feat_cols):
        _fv = X_arr[fraud_mask, _i]
        _nv = X_arr[normal_mask, _i]
        _fm, _nm = float(np.mean(_fv)), float(np.mean(_nv))
        _fs, _ns = float(np.std(_fv)), float(np.std(_nv))
        _nf, _nn = len(_fv), len(_nv)
        _pooled = np.sqrt(((_nf - 1) * _fs**2 + (_nn - 1) * _ns**2) / (_nf + _nn - 2))
        _d = (_fm - _nm) / (_pooled + 1e-10)
        feat_analysis.append({
            "name": _fname,
            "cohen_d": round(float(_d), 4),
            "abs_cohen_d": round(float(abs(_d)), 4),
            "fraud_mean": round(_fm, 4),
            "normal_mean": round(_nm, 4),
            "fraud_std": round(_fs, 4),
            "normal_std": round(_ns, 4),
        })
    feat_analysis.sort(key=lambda x: -x["abs_cohen_d"])
    save_json(feat_analysis, folder, "feature_analysis.json")

    return model_results


def _hist_to_json(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    return {
        "counts": [int(c) for c in counts],
        "edges": [round(float(e), 4) for e in edges],
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Monkey-patch numpy histogram to return JSON-serializable data
    original_histogram = np.histogram

    def json_histogram(values, bins):
        counts, edges = original_histogram(values, bins=bins)
        return {
            "counts": [int(c) for c in counts],
            "edges": [round(float(e), 4) for e in edges],
        }

    # Credit Card
    cc_df = load_creditcard()
    X_cc, y_cc, cc_raw = engineer_creditcard(cc_df)
    cc_results = generate_dataset_data(cc_raw, X_cc, y_cc, "cc", "Credit Card")

    # PaySim
    ps_df = load_paysim()
    X_ps, y_ps, ps_raw = engineer_paysim(ps_df)
    ps_results = generate_dataset_data(ps_raw, X_ps, y_ps, "ps", "PaySim")

    # --- Combined data ---
    print("\nGenerating combined data...")

    # comparison.json
    comparison = []
    for r in cc_results:
        comparison.append({**r, "dataset": "Credit Card"})
    for r in ps_results:
        comparison.append({**r, "dataset": "PaySim"})
    save_json(comparison, "combined", "comparison.json")

    # ranking.json
    ranking = []
    for dataset_name, results in [("Credit Card", cc_results), ("PaySim", ps_results)]:
        for r in results:
            ranking.append({"dataset": dataset_name, **r})
    save_json(ranking, "combined", "ranking.json")

    # best_models.json
    best = []
    for dataset_name, results in [("Credit Card", cc_results), ("PaySim", ps_results)]:
        best_model = max(results, key=lambda x: x["f1"])
        best.append({"dataset": dataset_name, **best_model})
    save_json(best, "combined", "best_models.json")

    # feature_names.json (for predict page)
    save_json({
        "creditcard": list(X_cc.columns),
        "paysim": list(X_ps.columns),
    }, "combined", "feature_names.json")

    print(f"\n{'='*60}")
    print("DASHBOARD DATA GENERATED")
    print(f"Output: {DATA_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
