"""
Generate pre-computed prediction results for the Predict page (Demo Mode).

Reads existing sample_transactions.json (raw features), enriches each sample with
a human-readable label/description, runs the local ensemble predictor on it,
and writes:

  frontend/public/data/{dataset}/sample_transactions.json    # enriched samples
  frontend/public/data/{dataset}/predict_samples.json        # label -> result

This avoids shipping large model artifacts to Vercel — the Predict page
loads the pre-computed results as static JSON.

Usage:
    python generate_predict_samples.py
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "frontend", "public", "data")

# Import the same predict() used by the live API so results are identical
sys.path.insert(0, os.path.join(ROOT_DIR, "frontend", "api"))
from predict import predict  # noqa: E402


# ---------------------------------------------------------------------------
# Human-readable labels for each sample (order matches sample_transactions.json)
# ---------------------------------------------------------------------------

CC_LABELS = {
    "fraud": [
        ("Card validation probe", "€0 transaction — typical stolen-card test before a real purchase."),
        ("High-value anomalous purchase", "~€529 with strongly atypical PCA profile (V1, V12 deviated)."),
        ("Mid-range deep V12 outlier", "~€240 transaction with V12/V17 far below normal range."),
        ("Extreme PCA outlier", "Multiple PCA components (V12, V14, V16, V17) show extreme deviation."),
        ("Mixed-signal suspicious", "Conflicting feature directions — classic compound-fraud pattern."),
    ],
    "normal": [
        ("Routine daytime purchase", "Modest amount, typical V-feature distribution, midday hour."),
        ("Standard evening transaction", "Regular consumer pattern, evening hour, ~€12."),
        ("Small routine payment", "Low amount, balanced PCA profile."),
        ("Higher-value normal purchase", "Larger amount but feature pattern within normal cluster."),
        ("Late-night legitimate purchase", "Off-hour but otherwise normal — should not trigger fraud."),
    ],
}

PS_LABELS = {
    "fraud": [
        ("Full account drain (TRANSFER)", "Sender balance emptied to 0 — classic mobile-money fraud."),
        ("CASH_OUT with balance mismatch", "Receiver balance does not reconcile — synthetic-account signature."),
        ("Mid-amount drain (TRANSFER)", "Account drained, destination balance unchanged."),
        ("Sequential CASH_OUT drain", "Higher amount, same drain-to-zero pattern."),
        ("High-value drain (TRANSFER)", "Large drain — typical end of fraud chain."),
    ],
    "normal": [
        ("Regular CASH_OUT", "Sender keeps residual balance — normal customer behaviour."),
        ("Routine PAYMENT", "Standard merchant payment, mid-amount."),
        ("Normal CASH_IN deposit", "Top-up, balances reconcile correctly."),
        ("Standard TRANSFER", "Customer transfer with reconciled destination balance."),
        ("Higher-value TRANSFER", "Larger but legitimate transfer with correct accounting."),
    ],
}


def enrich_and_predict(dataset_key: str, dataset_tag: str, labels: dict) -> tuple[dict, dict]:
    """Read raw sample_transactions.json, attach labels, run predictions."""
    src = os.path.join(DATA_DIR, dataset_key, "sample_transactions.json")
    with open(src) as f:
        raw = json.load(f)

    enriched = {"fraud": [], "normal": []}
    results = {}

    for variant in ("fraud", "normal"):
        samples = raw[variant]
        meta = labels[variant]
        for i, features in enumerate(samples):
            label, description = meta[i]
            enriched[variant].append({
                "label": label,
                "description": description,
                "features": features,
            })
            result = predict(dataset_tag, features)
            results[label] = result
            print(f"  [{dataset_key}/{variant}] {label}: "
                  f"{'FRAUD' if result['fraud'] else 'NORMAL'} "
                  f"score={result['ensemble_score']:.3f}")

    return enriched, results


def main():
    for dataset_key, dataset_tag, labels in [
        ("creditcard", "cc", CC_LABELS),
        ("paysim", "ps", PS_LABELS),
    ]:
        print(f"\n--- {dataset_key} ---")
        enriched, results = enrich_and_predict(dataset_key, dataset_tag, labels)

        out_dir = os.path.join(DATA_DIR, dataset_key)
        with open(os.path.join(out_dir, "sample_transactions.json"), "w") as f:
            json.dump(enriched, f)
        with open(os.path.join(out_dir, "predict_samples.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"  wrote {out_dir}/sample_transactions.json")
        print(f"  wrote {out_dir}/predict_samples.json")


if __name__ == "__main__":
    main()
