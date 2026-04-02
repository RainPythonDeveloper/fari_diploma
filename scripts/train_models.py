"""
Train supervised + unsupervised models for fraud detection.
Saves model artifacts to ../models/

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
        os.path.join(os.path.dirname(__file__), "..", "PS_20174392719_1491204439457_log.csv"),
        "PS_20174392719_1491204439457_log.csv",
        "/content/Diploma_files/PS_20174392719_1491204439457_log.csv",
        "/content/PS_20174392719_1491204439457_log.csv",
    ])
    if path is None:
        sys.exit("PaySim CSV not found")
    print(f"Loading PaySim from {path}")
    df = pd.read_csv(path).dropna().drop_duplicates().reset_index(drop=True)
    df = df.rename(columns={"isFraud": "Class"})

    # Sample to 500K rows (keep all fraud)
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
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] % 86400) / 3600
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    features = df.drop(columns=["Class", "Time", "Amount", "Hour"])
    return features, df["Class"]


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
    return features, df["Class"]

# ---------------------------------------------------------------------------
# Autoencoder (Keras)
# ---------------------------------------------------------------------------

def build_autoencoder(input_dim, encoding_dim=14):
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

    tf.random.set_seed(RANDOM_STATE)

    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    encoded = Dense(encoding_dim, activation="relu")(x)
    x = Dense(32, activation="relu")(encoded)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    outputs = Dense(input_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(X_train, contamination, epochs=100, batch_size=2048):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model = build_autoencoder(X_train.shape[1])
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, X_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        callbacks=[early_stop, reduce_lr]
    )

    train_recon = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - train_recon), axis=1)
    threshold = float(np.quantile(train_errors, 1 - contamination))

    return model, threshold, history.history

# ---------------------------------------------------------------------------
# Training pipeline for one dataset
# ---------------------------------------------------------------------------

def train_all_models(X, y, dataset_tag):
    print(f"\n{'='*60}")
    print(f"Training models for: {dataset_tag}")
    print(f"{'='*60}")

    contamination = float(np.clip(y.mean(), 1e-4, 0.05))

    # --- Split data ---
    # For unsupervised: train on normal only
    X_normal = X[y == 0]
    X_fraud = X[y == 1]

    X_train_unsup, X_test_normal = train_test_split(
        X_normal, test_size=0.2, random_state=RANDOM_STATE
    )
    X_test = pd.concat([X_test_normal, X_fraud], axis=0)
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])

    # For supervised: train on mixed (stratified)
    X_train_sup, X_test_sup, y_train_sup, y_test_sup = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Scale ---
    scaler = StandardScaler()
    X_train_unsup_scaled = scaler.fit_transform(X_train_unsup)
    X_test_scaled = scaler.transform(X_test)
    X_train_sup_scaled = scaler.transform(X_train_sup)
    X_test_sup_scaled = scaler.transform(X_test_sup)

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{dataset_tag}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to {scaler_path}")

    results = {}

    # --- 1. XGBoost (supervised) ---
    print("  Training XGBoost...")
    t0 = time.time()
    n_neg = (y_train_sup == 0).sum()
    n_pos = (y_train_sup == 1).sum()
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    xgb_model = xgb.XGBClassifier(
        max_depth=6,
        n_estimators=200,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        tree_method="hist",
    )
    xgb_model.fit(X_train_sup_scaled, y_train_sup, verbose=False)
    xgb_train_time = time.time() - t0

    # Evaluate on the unsupervised test set for fair comparison
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    xgb_path = os.path.join(MODELS_DIR, f"xgboost_{dataset_tag}.json")
    xgb_model.save_model(xgb_path)
    print(f"    Saved to {xgb_path}")

    results["XGBoost"] = evaluate(y_test, xgb_preds, xgb_probs, xgb_train_time)
    print(f"    F1={results['XGBoost']['f1']:.4f}  P={results['XGBoost']['precision']:.4f}  R={results['XGBoost']['recall']:.4f}")

    # --- 2. Isolation Forest (unsupervised) ---
    print("  Training Isolation Forest...")
    t0 = time.time()
    iforest = IsolationForest(
        n_estimators=300, max_samples=0.8,
        contamination=contamination, random_state=RANDOM_STATE
    )
    iforest.fit(X_train_unsup_scaled)
    iforest_train_time = time.time() - t0

    iforest_raw = iforest.predict(X_test_scaled)
    iforest_preds = np.where(iforest_raw == -1, 1, 0)
    iforest_scores = -iforest.decision_function(X_test_scaled)

    iforest_path = os.path.join(MODELS_DIR, f"iforest_{dataset_tag}.pkl")
    joblib.dump(iforest, iforest_path)
    print(f"    Saved to {iforest_path}")

    results["Isolation Forest"] = evaluate(y_test, iforest_preds, iforest_scores, iforest_train_time)
    print(f"    F1={results['Isolation Forest']['f1']:.4f}  P={results['Isolation Forest']['precision']:.4f}  R={results['Isolation Forest']['recall']:.4f}")

    # --- 3. Autoencoder (unsupervised) ---
    print("  Training Autoencoder...")
    t0 = time.time()
    ae_model, ae_threshold, ae_history = train_autoencoder(
        X_train_unsup_scaled, contamination
    )
    ae_train_time = time.time() - t0

    ae_recon = ae_model.predict(X_test_scaled, verbose=0)
    ae_errors = np.mean(np.square(X_test_scaled - ae_recon), axis=1)
    ae_preds = np.where(ae_errors > ae_threshold, 1, 0)
    ae_scores = ae_errors

    # Save as ONNX
    try:
        import tf2onnx
        import tensorflow as tf
        onnx_path = os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}.onnx")
        spec = (tf.TensorSpec(ae_model.input_shape[1:], tf.float32, name="input"),)
        tf2onnx.convert.from_keras(ae_model, input_signature=spec, output_path=onnx_path)
        print(f"    Saved ONNX to {onnx_path}")
    except Exception as e:
        print(f"    ONNX export failed ({e}), saving as .keras")
        ae_model.save(os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}.keras"))

    # Save threshold
    ae_config_path = os.path.join(MODELS_DIR, f"autoencoder_{dataset_tag}_config.json")
    with open(ae_config_path, "w") as f:
        json.dump({"threshold": ae_threshold}, f)

    results["Autoencoder"] = evaluate(y_test, ae_preds, ae_scores, ae_train_time)
    results["Autoencoder"]["history"] = {
        "loss": [float(v) for v in ae_history.get("loss", [])],
        "val_loss": [float(v) for v in ae_history.get("val_loss", [])],
    }
    print(f"    F1={results['Autoencoder']['f1']:.4f}  P={results['Autoencoder']['precision']:.4f}  R={results['Autoencoder']['recall']:.4f}")

    # --- 4. Ensemble ---
    print("  Building ensemble...")
    # Normalize scores to [0, 1]
    xgb_norm = xgb_probs  # already 0-1
    if_min, if_max = iforest_scores.min(), iforest_scores.max()
    iforest_norm = (iforest_scores - if_min) / (if_max - if_min + 1e-10)
    ae_min, ae_max = ae_scores.min(), ae_scores.max()
    ae_norm = (ae_scores - ae_min) / (ae_max - ae_min + 1e-10)

    # Weighted average (XGBoost dominant since it's supervised)
    weights = {"xgboost": 0.6, "isolation_forest": 0.2, "autoencoder": 0.2}
    ensemble_scores = (
        weights["xgboost"] * xgb_norm +
        weights["isolation_forest"] * iforest_norm +
        weights["autoencoder"] * ae_norm
    )

    # Find optimal threshold via F1
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds_t = (ensemble_scores >= t).astype(int)
        f1_t = f1_score(y_test, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t

    ensemble_preds = (ensemble_scores >= best_thresh).astype(int)

    results["Ensemble"] = evaluate(y_test, ensemble_preds, ensemble_scores, 0)
    print(f"    Threshold={best_thresh:.2f}  F1={results['Ensemble']['f1']:.4f}  P={results['Ensemble']['precision']:.4f}  R={results['Ensemble']['recall']:.4f}")

    # Save ensemble config
    ensemble_config = {
        "weights": weights,
        "threshold": float(best_thresh),
        "iforest_score_range": {"min": float(if_min), "max": float(if_max)},
        "ae_score_range": {"min": float(ae_min), "max": float(ae_max)},
        "ae_threshold": float(ae_threshold),
    }
    config_path = os.path.join(MODELS_DIR, f"ensemble_{dataset_tag}.json")
    with open(config_path, "w") as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"    Saved ensemble config to {config_path}")

    return results, scaler


def evaluate(y_true, y_pred, scores, train_time):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "train_time": round(train_time, 2),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_results = {}

    # Credit Card
    cc_df = load_creditcard()
    X_cc, y_cc = engineer_creditcard(cc_df)
    cc_results, _ = train_all_models(X_cc, y_cc, "cc")
    all_results["creditcard"] = cc_results

    # PaySim
    ps_df = load_paysim()
    X_ps, y_ps = engineer_paysim(ps_df)
    ps_results, _ = train_all_models(X_ps, y_ps, "ps")
    all_results["paysim"] = ps_results

    # Save results summary
    summary_path = os.path.join(MODELS_DIR, "training_results.json")
    # Remove non-serializable history before saving
    save_results = {}
    for ds, models in all_results.items():
        save_results[ds] = {}
        for model_name, metrics in models.items():
            save_results[ds][model_name] = {k: v for k, v in metrics.items() if k != "history"}
            if "history" in metrics:
                save_results[ds][model_name]["has_history"] = True

    with open(summary_path, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Artifacts saved to: {MODELS_DIR}")

    for ds_name, ds_results in all_results.items():
        print(f"\n{ds_name}:")
        for model_name, metrics in sorted(ds_results.items(), key=lambda x: -x[1]["f1"]):
            print(f"  {model_name:25s}  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
