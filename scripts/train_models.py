"""
Train supervised + unsupervised models for fraud detection (NEW LOGIC).

Pipeline (matches newLogic/ notebooks):
  - TWO scalers: supervised (fit on all train) + unsupervised (fit on normal-only train)
  - XGBoost trained on stratified 80/20 split
  - Isolation Forest + Autoencoder trained on normal-only data
  - Ensemble = simple mean of min-max normalized scores from all three models
  - Threshold = percentile(ensemble_scores, 100*(1-contamination))

Per-dataset AE hyperparameters:
  - Credit Card: encoding_dim=14, epochs=20, batch_size=2048
  - PaySim:      encoding_dim=8,  epochs=10, batch_size=1024

Saves artifacts to ../models/:
  - scaler_sup_{tag}.pkl, scaler_unsup_{tag}.pkl
  - xgboost_{tag}.json, iforest_{tag}.pkl
  - autoencoder_{tag}.onnx (+ .keras fallback), autoencoder_{tag}_config.json
  - ensemble_{tag}.json (score ranges + threshold)
  - training_results.json (metrics summary)

Usage:
    cd scripts
    pip install -r requirements.txt
    python train_models.py
"""

import os
import sys
import json
import time
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
    roc_auc_score, average_precision_score, confusion_matrix
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Per-dataset AE hyperparameters (from newLogic notebooks)
AE_PARAMS = {
    "cc": {"encoding_dim": 14, "epochs": 20, "batch_size": 2048},
    "ps": {"encoding_dim": 8,  "epochs": 10, "batch_size": 1024},
}

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def find_file(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_creditcard():
    path = find_file([
        os.path.join(os.path.dirname(__file__), "..", "creditcard.csv"),
        "creditcard.csv",
        "/content/Diploma_files/creditcard.csv",
        "/content/creditcard.csv",
    ])
    if path is None:
        sys.exit("creditcard.csv not found")
    print(f"Loading Credit Card from {path}")
    df = pd.read_csv(path).dropna().drop_duplicates().reset_index(drop=True)
    print(f"  Shape: {df.shape}, Fraud: {df['Class'].sum()}")
    return df


def load_paysim():
    path = find_file([
        os.path.join(os.path.dirname(__file__), "..", "paysim.csv"),
        os.path.join(os.path.dirname(__file__), "..", "PS_20174392719_1491204439457_log.csv"),
        "paysim.csv",
        "PS_20174392719_1491204439457_log.csv",
        "/content/Diploma_files/PS_20174392719_1491204439457_log.csv",
        "/content/PS_20174392719_1491204439457_log.csv",
    ])
    if path is None:
        sys.exit("PaySim CSV not found")
    print(f"Loading PaySim from {path}")
    df = pd.read_csv(path).dropna().drop_duplicates().reset_index(drop=True)
    df = df.rename(columns={"isFraud": "Class"})

    # Sample for memory (full dataset is 6.3M rows; new notebook used 3.5M)
    if len(df) > 500_000:
        fraud = df[df["Class"] == 1]
        normal = df[df["Class"] == 0].sample(n=500_000 - len(fraud), random_state=RANDOM_STATE)
        df = pd.concat([normal, fraud]).reset_index(drop=True)
        print(f"  Sampled to {len(df)} rows (all {len(fraud)} frauds kept)")
    print(f"  Shape: {df.shape}, Fraud: {df['Class'].sum()}")
    return df

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_creditcard(df):
    """Credit Card: drop Time (per new notebook), keep V1..V28 + Amount + engineered."""
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] % 86400) / 3600
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    features = df.drop(columns=["Class", "Time", "Amount", "Hour"])
    return features, df["Class"]


def engineer_paysim(df):
    """PaySim: balance differentials + amount_log + type_code (per new notebook)."""
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
    return features, df["Class"]

# ---------------------------------------------------------------------------
# Autoencoder (Keras) — new simple architecture matching notebooks
# ---------------------------------------------------------------------------

def build_autoencoder(input_dim, encoding_dim):
    """Simple AE: Input -> Dense(16) -> Dense(encoding_dim) -> Dense(16) -> Dense(input_dim)."""
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Dense

    tf.random.set_seed(RANDOM_STATE)

    inputs = Input(shape=(input_dim,))
    x = Dense(16, activation="relu")(inputs)
    x = Dense(encoding_dim, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(input_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(X_train, contamination, encoding_dim, epochs, batch_size):
    from tensorflow.keras.callbacks import EarlyStopping

    model = build_autoencoder(X_train.shape[1], encoding_dim)
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, X_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        callbacks=[early_stop]
    )

    train_recon = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - train_recon), axis=1)
    # Per new notebook: threshold = percentile (100 * (1 - contamination))
    threshold = float(np.percentile(train_errors, 100 * (1 - contamination)))

    return model, threshold, history.history


def minmax_norm(arr):
    """Min-max normalize array to [0,1]. Returns (normalized, min, max)."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float64), mn, mx
    return (arr - mn) / (mx - mn), mn, mx

# ---------------------------------------------------------------------------
# Training pipeline for one dataset
# ---------------------------------------------------------------------------

def train_all_models(X, y, dataset_tag):
    print(f"\n{'='*60}")
    print(f"Training models for: {dataset_tag}")
    print(f"{'='*60}")

    contamination = float(max(y.mean(), 1e-4))
    if dataset_tag == "cc":
        # Clip to reasonable range for CC
        contamination = float(np.clip(y.mean(), 1e-4, 0.05))
    print(f"  Contamination: {contamination}")

    # --- Supervised split (XGBoost): stratified 80/20 on full data ---
    X_train_sup, X_test_sup, y_train_sup, y_test_sup = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Unsupervised split: train on normal only ---
    X_normal = X[y == 0]
    X_fraud = X[y == 1]
    X_train_unsup, X_test_normal = train_test_split(
        X_normal, test_size=0.2, random_state=RANDOM_STATE
    )
    # Shared unsupervised test set = normal_test + all fraud
    X_test_unsup = pd.concat([X_test_normal, X_fraud], axis=0)
    y_test_unsup = np.concatenate([
        np.zeros(len(X_test_normal)),
        np.ones(len(X_fraud))
    ])

    print(f"  XGBoost   train: {X_train_sup.shape}  test: {X_test_sup.shape}")
    print(f"  Unsup     train: {X_train_unsup.shape}  test: {X_test_unsup.shape}")
    print(f"  Fraud in unsup test: {int(y_test_unsup.sum())}")

    # --- Two scalers ---
    scaler_sup = StandardScaler()
    X_train_sup_scaled = scaler_sup.fit_transform(X_train_sup)
    X_test_sup_scaled = scaler_sup.transform(X_test_sup)

    scaler_unsup = StandardScaler()
    X_train_unsup_scaled = scaler_unsup.fit_transform(X_train_unsup)
    X_test_unsup_scaled = scaler_unsup.transform(X_test_unsup)

    joblib.dump(scaler_sup, os.path.join(MODELS_DIR, f"scaler_sup_{dataset_tag}.pkl"))
    joblib.dump(scaler_unsup, os.path.join(MODELS_DIR, f"scaler_unsup_{dataset_tag}.pkl"))
    print(f"  Saved scaler_sup_{dataset_tag}.pkl and scaler_unsup_{dataset_tag}.pkl")

    results = {}

    # --- 1. XGBoost (supervised) ---
    print("  Training XGBoost...")
    t0 = time.time()
    n_neg = int((y_train_sup == 0).sum())
    n_pos = int((y_train_sup == 1).sum())
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_sup_scaled, y_train_sup, verbose=False)
    xgb_train_time = time.time() - t0

    # Evaluate XGBoost on its OWN supervised test split (matches new notebook)
    t0 = time.time()
    xgb_probs = xgb_model.predict_proba(X_test_sup_scaled)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    xgb_test_time = time.time() - t0

    xgb_model.save_model(os.path.join(MODELS_DIR, f"xgboost_{dataset_tag}.json"))

    results["XGBoost"] = evaluate(y_test_sup, xgb_preds, xgb_probs, xgb_train_time, xgb_test_time)
    print(f"    F1={results['XGBoost']['f1']:.4f}  P={results['XGBoost']['precision']:.4f}  R={results['XGBoost']['recall']:.4f}")

    # --- 2. Isolation Forest (unsupervised) ---
    print("  Training Isolation Forest...")
    t0 = time.time()
    iforest = IsolationForest(
        n_estimators=300,
        max_samples=0.8,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_train_unsup_scaled)
    if_train_time = time.time() - t0

    t0 = time.time()
    if_raw = iforest.predict(X_test_unsup_scaled)
    if_preds = np.where(if_raw == -1, 1, 0)
    if_scores = -iforest.decision_function(X_test_unsup_scaled)
    if_test_time = time.time() - t0

    joblib.dump(iforest, os.path.join(MODELS_DIR, f"iforest_{dataset_tag}.pkl"))

    results["Isolation Forest"] = evaluate(y_test_unsup, if_preds, if_scores, if_train_time, if_test_time)
    print(f"    F1={results['Isolation Forest']['f1']:.4f}  P={results['Isolation Forest']['precision']:.4f}  R={results['Isolation Forest']['recall']:.4f}")

    # --- 3. Autoencoder (unsupervised) ---
    print("  Training Autoencoder...")
    ae_params = AE_PARAMS[dataset_tag]
    t0 = time.time()
    ae_model, ae_threshold, ae_history = train_autoencoder(
        X_train_unsup_scaled, contamination,
        encoding_dim=ae_params["encoding_dim"],
        epochs=ae_params["epochs"],
        batch_size=ae_params["batch_size"],
    )
    ae_train_time = time.time() - t0

    t0 = time.time()
    ae_recon = ae_model.predict(X_test_unsup_scaled, verbose=0)
    ae_scores = np.mean(np.square(X_test_unsup_scaled - ae_recon), axis=1)
    ae_preds = np.where(ae_scores > ae_threshold, 1, 0)
    ae_test_time = time.time() - t0

    # Save as ONNX
    try:
        import tf2onnx
        import tensorflow as tf
        onnx_path = os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}.onnx")
        spec = (tf.TensorSpec((None,) + ae_model.input_shape[1:], tf.float32, name="input"),)
        tf2onnx.convert.from_keras(ae_model, input_signature=spec, output_path=onnx_path)
        print(f"    Saved ONNX to {onnx_path}")
    except Exception as e:
        print(f"    ONNX export failed ({e}), saving as .keras")
        ae_model.save(os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}.keras"))

    with open(os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}_config.json"), "w") as f:
        json.dump({"threshold": ae_threshold}, f)

    results["Autoencoder"] = evaluate(y_test_unsup, ae_preds, ae_scores, ae_train_time, ae_test_time)
    results["Autoencoder"]["history"] = {
        "loss": [float(v) for v in ae_history.get("loss", [])],
        "val_loss": [float(v) for v in ae_history.get("val_loss", [])],
    }
    print(f"    F1={results['Autoencoder']['f1']:.4f}  P={results['Autoencoder']['precision']:.4f}  R={results['Autoencoder']['recall']:.4f}")

    # --- 4. Ensemble (mean of min-max normalized scores) ---
    # New logic: re-score XGBoost on the shared unsupervised test set, then mean-normalize
    print("  Building ensemble...")
    X_test_unsup_for_xgb = scaler_sup.transform(X_test_unsup)
    xgb_scores_ens = xgb_model.predict_proba(X_test_unsup_for_xgb)[:, 1]

    xgb_norm, xgb_min, xgb_max = minmax_norm(xgb_scores_ens)
    if_norm, if_min, if_max = minmax_norm(if_scores)
    ae_norm, ae_min, ae_max = minmax_norm(ae_scores)

    ensemble_scores = (xgb_norm + if_norm + ae_norm) / 3.0
    ensemble_threshold = float(np.percentile(ensemble_scores, 100 * (1 - contamination)))
    ensemble_preds = (ensemble_scores >= ensemble_threshold).astype(int)

    results["Ensemble"] = evaluate(y_test_unsup, ensemble_preds, ensemble_scores, 0.0, 0.0)
    print(f"    Threshold={ensemble_threshold:.4f}  F1={results['Ensemble']['f1']:.4f}  P={results['Ensemble']['precision']:.4f}  R={results['Ensemble']['recall']:.4f}")

    # --- Save ensemble config (full normalizers for runtime) ---
    ensemble_config = {
        "method": "mean_of_minmax_normalized",
        "threshold": ensemble_threshold,
        "contamination": contamination,
        "xgb_score_range": {"min": xgb_min, "max": xgb_max},
        "iforest_score_range": {"min": if_min, "max": if_max},
        "ae_score_range": {"min": ae_min, "max": ae_max},
        "ae_threshold": ae_threshold,
    }
    with open(os.path.join(MODELS_DIR, f"ensemble_{dataset_tag}.json"), "w") as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"    Saved ensemble_{dataset_tag}.json")

    return results


def evaluate(y_true, y_pred, scores, train_time, test_time):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "train_time": round(train_time, 2),
        "test_time": round(test_time, 2),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_results = {}

    cc_df = load_creditcard()
    X_cc, y_cc = engineer_creditcard(cc_df)
    all_results["creditcard"] = train_all_models(X_cc, y_cc, "cc")

    ps_df = load_paysim()
    X_ps, y_ps = engineer_paysim(ps_df)
    all_results["paysim"] = train_all_models(X_ps, y_ps, "ps")

    # Save summary (history stripped for JSON serialization)
    save_results = {}
    for ds, models in all_results.items():
        save_results[ds] = {}
        for model_name, metrics in models.items():
            save_results[ds][model_name] = {k: v for k, v in metrics.items() if k != "history"}
            if "history" in metrics:
                save_results[ds][model_name]["has_history"] = True
                save_results[ds][model_name]["history"] = metrics["history"]

    with open(os.path.join(MODELS_DIR, "training_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Artifacts saved to: {MODELS_DIR}")

    for ds_name, ds_results in all_results.items():
        print(f"\n{ds_name}:")
        for model_name, metrics in sorted(ds_results.items(), key=lambda x: -x[1]["f1"]):
            print(f"  {model_name:20s}  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
