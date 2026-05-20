"""
Generate all JSON data for the dashboard (NEW ENSEMBLE LOGIC).

Loads trained models from ../models/ and produces JSON files in
../frontend/public/data/.

Evaluation strategy (matches new training pipeline):
  - XGBoost is evaluated on its own SUPERVISED test set (stratified 80/20).
  - Isolation Forest / Autoencoder / Ensemble are evaluated on the SHARED
    UNSUPERVISED test set (normal_test + all fraud).
  - For the transactions table + ensemble math, XGBoost is RE-SCORED on the
    unsupervised test set so all four models share comparable scores.

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

AE_PARAMS = {
    "cc": {"encoding_dim": 14, "epochs": 20, "batch_size": 2048},
    "ps": {"encoding_dim": 8,  "epochs": 10, "batch_size": 1024},
}

# ---------------------------------------------------------------------------
# Loaders (mirror train_models.py)
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
        os.path.join(BASE_DIR, "..", "paysim.csv"),
        os.path.join(BASE_DIR, "..", "PS_20174392719_1491204439457_log.csv"),
        "paysim.csv",
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


def _hist_to_json(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    return {
        "counts": [int(c) for c in counts],
        "edges": [round(float(e), 4) for e in edges],
    }

# ---------------------------------------------------------------------------
# Generate data for one dataset
# ---------------------------------------------------------------------------

def generate_dataset_data(df_raw, X, y, dataset_tag, dataset_name):
    print(f"\nGenerating data for {dataset_name}...")

    contamination = float(max(y.mean(), 1e-4))
    if dataset_tag == "cc":
        contamination = float(np.clip(y.mean(), 1e-4, 0.05))

    # --- Reproduce both splits from training ---
    X_train_sup, X_test_sup, y_train_sup, y_test_sup = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_normal = X[y == 0]
    X_fraud = X[y == 1]
    X_train_unsup, X_test_normal = train_test_split(
        X_normal, test_size=0.2, random_state=RANDOM_STATE
    )
    X_test_unsup = pd.concat([X_test_normal, X_fraud], axis=0)
    y_test_unsup = np.concatenate([
        np.zeros(len(X_test_normal)),
        np.ones(len(X_fraud))
    ])

    # --- Load both scalers ---
    scaler_sup = joblib.load(os.path.join(MODELS_DIR, f"scaler_sup_{dataset_tag}.pkl"))
    scaler_unsup = joblib.load(os.path.join(MODELS_DIR, f"scaler_unsup_{dataset_tag}.pkl"))

    X_test_sup_scaled = scaler_sup.transform(X_test_sup)
    X_test_unsup_scaled = scaler_unsup.transform(X_test_unsup)

    # --- Load models and produce raw scores on their proper test sets ---
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, f"xgboost_{dataset_tag}.json"))

    # XGBoost on supervised test
    xgb_scores_sup = xgb_model.predict_proba(X_test_sup_scaled)[:, 1]
    xgb_preds_sup = (xgb_scores_sup >= 0.5).astype(int)

    # XGBoost RE-SCORED on unsupervised test (for ensemble + transactions)
    X_test_unsup_for_xgb = scaler_sup.transform(X_test_unsup)
    xgb_scores_unsup = xgb_model.predict_proba(X_test_unsup_for_xgb)[:, 1]

    # SHAP feature importance (computed on the unsup set for consistency)
    shap_data = []
    try:
        import shap as shap_lib
        explainer = shap_lib.TreeExplainer(xgb_model)
        n = min(200, len(X_test_unsup_for_xgb))
        idx = np.random.choice(len(X_test_unsup_for_xgb), size=n, replace=False)
        vals = explainer.shap_values(X_test_unsup_for_xgb[idx])
        mean_abs = np.mean(np.abs(vals), axis=0)
        mean_v = np.mean(vals, axis=0)
        feat_names = list(scaler_sup.feature_names_in_)
        shap_data = sorted([
            {
                "name": feat_names[i],
                "mean_abs_shap": round(float(mean_abs[i]), 4),
                "mean_shap": round(float(mean_v[i]), 4),
            }
            for i in range(len(feat_names))
        ], key=lambda x: -x["mean_abs_shap"])
    except ImportError:
        print("  shap not installed — skipping global SHAP")

    # Isolation Forest on unsupervised test
    iforest = joblib.load(os.path.join(MODELS_DIR, f"iforest_{dataset_tag}.pkl"))
    if_raw = iforest.predict(X_test_unsup_scaled)
    if_preds = np.where(if_raw == -1, 1, 0)
    if_scores = -iforest.decision_function(X_test_unsup_scaled)

    # Autoencoder on unsupervised test (ONNX or Keras fallback)
    ae_scores = None
    try:
        import onnxruntime as ort
        onnx_path = os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}.onnx")
        if os.path.exists(onnx_path):
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            ae_recon = sess.run(None, {input_name: X_test_unsup_scaled.astype(np.float32)})[0]
            ae_scores = np.mean(np.square(X_test_unsup_scaled - ae_recon), axis=1)
    except ImportError:
        pass

    if ae_scores is None:
        try:
            import tensorflow as tf
            keras_path = os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}.keras")
            if os.path.exists(keras_path):
                ae_model = tf.keras.models.load_model(keras_path)
                ae_recon = ae_model.predict(X_test_unsup_scaled, verbose=0)
                ae_scores = np.mean(np.square(X_test_unsup_scaled - ae_recon), axis=1)
        except ImportError:
            pass

    if ae_scores is None:
        sys.exit(f"Could not load Autoencoder for {dataset_tag}. Run train_models.py first.")

    ae_config = json.load(open(os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}_config.json")))
    ae_preds = np.where(ae_scores > ae_config["threshold"], 1, 0)

    # --- Ensemble: mean of min-max normalized scores ---
    ensemble_config = json.load(open(os.path.join(MODELS_DIR, f"ensemble_{dataset_tag}.json")))
    xgb_range = ensemble_config["xgb_score_range"]
    if_range = ensemble_config["iforest_score_range"]
    ae_range = ensemble_config["ae_score_range"]
    ens_threshold = ensemble_config["threshold"]

    def norm(arr, rng):
        span = rng["max"] - rng["min"]
        if span <= 0:
            return np.zeros_like(arr, dtype=np.float64)
        out = (arr - rng["min"]) / span
        return np.clip(out, 0.0, 1.0)

    xgb_norm = norm(xgb_scores_unsup, xgb_range)
    if_norm = norm(if_scores, if_range)
    ae_norm = norm(ae_scores, ae_range)

    ensemble_scores = (xgb_norm + if_norm + ae_norm) / 3.0
    ensemble_preds = (ensemble_scores >= ens_threshold).astype(int)

    # --- Per-model evaluation: each on its own proper test set ---
    # XGBoost -> supervised; IF/AE/Ensemble -> unsupervised
    eval_map = {
        "XGBoost":          {"y": y_test_sup,   "preds": xgb_preds_sup, "scores": xgb_scores_sup},
        "Isolation Forest": {"y": y_test_unsup, "preds": if_preds,      "scores": if_scores},
        "Autoencoder":      {"y": y_test_unsup, "preds": ae_preds,      "scores": ae_scores},
        "Ensemble":         {"y": y_test_unsup, "preds": ensemble_preds, "scores": ensemble_scores},
    }

    folder = "creditcard" if dataset_tag == "cc" else "paysim"

    # --- summary.json ---
    save_json({
        "name": dataset_name,
        "total_samples": int(len(df_raw)),
        "normal": int((y == 0).sum()),
        "fraud": int((y == 1).sum()),
        "fraud_rate": round(float(y.mean()) * 100, 4),
        "features_count": int(X.shape[1]),
        "test_samples": int(len(y_test_unsup)),
        "test_fraud": int(y_test_unsup.sum()),
        "contamination": contamination,
        "ensemble_method": "mean_of_minmax_normalized",
        "ensemble_threshold": round(float(ens_threshold), 4),
    }, folder, "summary.json")

    # --- model_results.json ---
    model_results = []
    for model_name, ev in eval_map.items():
        preds = ev["preds"]
        scores = ev["scores"]
        y_true = ev["y"]
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        model_results.append({
            "model": model_name,
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
            "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })
    model_results.sort(key=lambda r: -r["f1"])
    for i, r in enumerate(model_results):
        r["rank"] = i + 1
    save_json(model_results, folder, "model_results.json")

    # --- confusion_matrices.json ---
    cm_data = {}
    for model_name, ev in eval_map.items():
        tn, fp, fn, tp = confusion_matrix(ev["y"], ev["preds"]).ravel()
        cm_data[model_name] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    save_json(cm_data, folder, "confusion_matrices.json")

    # --- roc_curves.json ---
    roc_data = {}
    for model_name, ev in eval_map.items():
        fpr, tpr, thresh_roc = roc_curve(ev["y"], ev["scores"])
        step = max(1, len(fpr) // 200)
        roc_data[model_name] = {
            "fpr": [round(float(v), 4) for v in fpr[::step]],
            "tpr": [round(float(v), 4) for v in tpr[::step]],
            "auc": round(float(roc_auc_score(ev["y"], ev["scores"])), 4),
            "thresholds": [round(float(v), 4) for v in thresh_roc[::step]],
        }
    save_json(roc_data, folder, "roc_curves.json")

    # --- pr_curves.json ---
    pr_data = {}
    for model_name, ev in eval_map.items():
        prec, rec, thresh_pr = precision_recall_curve(ev["y"], ev["scores"])
        prec_a, rec_a = prec[:-1], rec[:-1]
        step = max(1, len(prec_a) // 200)
        pr_data[model_name] = {
            "precision": [round(float(v), 4) for v in prec_a[::step]],
            "recall": [round(float(v), 4) for v in rec_a[::step]],
            "ap": round(float(average_precision_score(ev["y"], ev["scores"])), 4),
            "thresholds": [round(float(v), 4) for v in thresh_pr[::step]],
        }
    save_json(pr_data, folder, "pr_curves.json")

    # --- transactions.json (~5000 rows from UNSUPERVISED test set) ---
    # All four models have comparable scores here (XGBoost re-scored on unsup).
    transactions_models = {
        "XGBoost":          {"preds": (xgb_scores_unsup >= 0.5).astype(int), "scores": xgb_scores_unsup},
        "Isolation Forest": {"preds": if_preds, "scores": if_scores},
        "Autoencoder":      {"preds": ae_preds, "scores": ae_scores},
        "Ensemble":         {"preds": ensemble_preds, "scores": ensemble_scores},
    }
    fraud_idx = np.where(y_test_unsup == 1)[0]
    normal_idx = np.where(y_test_unsup == 0)[0]
    sample_normal = np.random.choice(normal_idx, size=min(4500, len(normal_idx)), replace=False)
    sample_idx = np.concatenate([fraud_idx, sample_normal])
    sample_idx.sort()

    X_test_unsup_reset = X_test_unsup.reset_index(drop=True)
    transactions = []
    for i in sample_idx:
        row = {"is_fraud": int(y_test_unsup[i]), "index": int(i)}
        if dataset_tag == "cc":
            row["amount"] = round(float(np.expm1(X_test_unsup_reset.iloc[i].get("Amount_log", 0))), 2)
        else:
            row["amount"] = round(float(np.expm1(X_test_unsup_reset.iloc[i].get("amount_log", 0))), 2)
            for col in ["step", "type_code", "balance_diff_orig", "balance_diff_dest", "error_orig", "error_dest"]:
                if col in X_test_unsup_reset.columns:
                    row[col] = round(float(X_test_unsup_reset.iloc[i][col]), 2)

        scores = {}
        for model_name, mdata in transactions_models.items():
            scores[model_name] = round(float(mdata["scores"][i]), 4)
        row["scores"] = scores
        row["ensemble_prediction"] = int(transactions_models["Ensemble"]["preds"][i])
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
        fraud_by_type = df_raw[df_raw["Class"] == 1]["type"].value_counts().to_dict()
        save_json({
            "amount": {
                "normal": _hist_to_json(normal_amounts, 50),
                "fraud": _hist_to_json(fraud_amounts, 50),
            },
            "fraud_by_type": fraud_by_type,
        }, folder, "distributions.json")

    # --- training_history.json (real AE loss curves from training_results.json) ---
    tr_path = os.path.join(MODELS_DIR, "training_results.json")
    history_payload = {"note": "Run train_models.py to generate autoencoder loss history"}
    if os.path.exists(tr_path):
        with open(tr_path) as f:
            tr = json.load(f)
        ds_key = "creditcard" if dataset_tag == "cc" else "paysim"
        ae_entry = tr.get(ds_key, {}).get("Autoencoder", {})
        if "history" in ae_entry:
            history_payload = {
                "loss": ae_entry["history"].get("loss", []),
                "val_loss": ae_entry["history"].get("val_loss", []),
                "epochs": len(ae_entry["history"].get("loss", [])),
            }
    save_json(history_payload, folder, "training_history.json")

    # --- hyperparameters.json ---
    ae_p = AE_PARAMS[dataset_tag]
    scale_pos_ratio = round(float((y == 0).sum() / max((y == 1).sum(), 1)), 1)
    hyperparams = [
        {"model": "XGBoost", "params": f"n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight={scale_pos_ratio}, eval_metric=logloss"},
        {"model": "Isolation Forest", "params": f"n_estimators=300, max_samples=0.8, contamination={contamination:.6f}"},
        {"model": "Autoencoder", "params": f"arch=Input-Dense16-Dense{ae_p['encoding_dim']}-Dense16-Output, epochs={ae_p['epochs']}, batch_size={ae_p['batch_size']}"},
        {"model": "Ensemble", "params": f"method=mean of min-max normalized (XGB+IF+AE)/3, threshold={ens_threshold:.4f} (percentile {(1-contamination)*100:.2f})"},
    ]
    save_json(hyperparams, folder, "hyperparameters.json")

    # --- sample_transactions.json (for predict page) ---
    samples = {"fraud": [], "normal": []}
    fraud_examples = X_test_sup[y_test_sup == 1].head(5)
    normal_examples = X_test_sup[y_test_sup == 0].head(5)
    for _, row in fraud_examples.iterrows():
        samples["fraud"].append({col: round(float(row[col]), 6) for col in X_test_sup.columns})
    for _, row in normal_examples.iterrows():
        samples["normal"].append({col: round(float(row[col]), 6) for col in X_test_sup.columns})
    save_json(samples, folder, "sample_transactions.json")

    # --- shap_values.json ---
    if shap_data:
        save_json(shap_data, folder, "shap_values.json")

    # --- feature_analysis.json (Cohen's d per feature) ---
    feat_cols = list(X.columns)
    X_arr = X.values
    fraud_mask = (y == 1).values
    normal_mask = (y == 0).values
    feat_analysis = []
    for i, fname in enumerate(feat_cols):
        fv = X_arr[fraud_mask, i]
        nv = X_arr[normal_mask, i]
        fm, nm = float(np.mean(fv)), float(np.mean(nv))
        fs, ns = float(np.std(fv)), float(np.std(nv))
        nf, nn = len(fv), len(nv)
        pooled = np.sqrt(((nf - 1) * fs**2 + (nn - 1) * ns**2) / max(nf + nn - 2, 1))
        d = (fm - nm) / (pooled + 1e-10)
        feat_analysis.append({
            "name": fname,
            "cohen_d": round(float(d), 4),
            "abs_cohen_d": round(float(abs(d)), 4),
            "fraud_mean": round(fm, 4),
            "normal_mean": round(nm, 4),
            "fraud_std": round(fs, 4),
            "normal_std": round(ns, 4),
        })
    feat_analysis.sort(key=lambda x: -x["abs_cohen_d"])
    save_json(feat_analysis, folder, "feature_analysis.json")

    return model_results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cc_df = load_creditcard()
    X_cc, y_cc, cc_raw = engineer_creditcard(cc_df)
    cc_results = generate_dataset_data(cc_raw, X_cc, y_cc, "cc", "Credit Card")

    ps_df = load_paysim()
    X_ps, y_ps, ps_raw = engineer_paysim(ps_df)
    ps_results = generate_dataset_data(ps_raw, X_ps, y_ps, "ps", "PaySim")

    print("\nGenerating combined data...")

    comparison = [{**r, "dataset": "Credit Card"} for r in cc_results] + \
                 [{**r, "dataset": "PaySim"} for r in ps_results]
    save_json(comparison, "combined", "comparison.json")

    ranking = []
    for dataset_name, results in [("Credit Card", cc_results), ("PaySim", ps_results)]:
        for r in results:
            ranking.append({"dataset": dataset_name, **r})
    save_json(ranking, "combined", "ranking.json")

    best = []
    for dataset_name, results in [("Credit Card", cc_results), ("PaySim", ps_results)]:
        best_model = max(results, key=lambda x: x["f1"])
        best.append({"dataset": dataset_name, **best_model})
    save_json(best, "combined", "best_models.json")

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
