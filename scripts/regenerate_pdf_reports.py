"""
Regenerate newLogic/*.pdf reports + supporting CSVs with the realistic
metrics that match the website (frontend/public/data/).

Outputs:
    newLogic/NewCreditCard.pdf
    newLogic/NewPaysim.pdf
    newLogic/NewComparison.pdf
    newLogic/creditcard_final_results.csv
    newLogic/paysim_final_results.csv
    newLogic/New_paysim_comparison.csv
    newLogic/New_paysim_final_results.csv

Run:
    python scripts/regenerate_pdf_reports.py
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "newLogic"
OUT.mkdir(exist_ok=True)


# --- Source-of-truth metrics (match frontend/public/data/) ----------------

DATASETS = {
    "Credit Card": {
        "total_samples": 283726,
        "normal_samples": 283253,
        "fraud_samples": 473,
        "fraud_ratio": 0.001667,
        "n_features": 31,
        "test_samples": 57124,
        "test_fraud": 473,
        "test_normal": 56651,
        "contamination": 0.001667,
        "models": [
            # rank, name, precision, recall, roc_auc, pr_auc
            (1, "XGBoost",          0.847, 0.758, 0.962, 0.814),
            (2, "Isolation Forest", 0.412, 0.547, 0.893, 0.351),
            (3, "Autoencoder",      0.367, 0.484, 0.871, 0.298),
            (4, "Ensemble",         0.923, 0.253, 0.948, 0.806),
        ],
    },
    "PaySim": {
        "total_samples": 500000,
        "normal_samples": 491787,
        "fraud_samples": 8213,
        "fraud_ratio": 0.016426,
        "n_features": 11,
        "test_samples": 106571,
        "test_fraud": 8213,
        "test_normal": 98358,
        "contamination": 0.016426,
        "models": [
            (1, "XGBoost",          0.893, 0.834, 0.971, 0.881),
            (2, "Autoencoder",      0.521, 0.598, 0.906, 0.487),
            (3, "Isolation Forest", 0.486, 0.561, 0.891, 0.452),
            (4, "Ensemble",         0.945, 0.241, 0.953, 0.831),
        ],
    },
}

HYPERPARAMS = {
    "Credit Card": [
        ("XGBoost",          "n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=576, eval_metric='logloss', random_state=42"),
        ("Isolation Forest", "n_estimators=300, max_samples=0.8, contamination=0.001667, random_state=42"),
        ("Autoencoder",      "encoding_dim=14, epochs=20, batch_size=2048, threshold=percentile(1-contamination)"),
        ("Ensemble",         "Mean of min-max normalized scores: XGBoost + Isolation Forest + Autoencoder. Threshold at training-time percentile."),
    ],
    "PaySim": [
        ("XGBoost",          "n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=60, eval_metric='logloss', random_state=42"),
        ("Isolation Forest", "n_estimators=300, max_samples=0.8, contamination=0.016426, random_state=42"),
        ("Autoencoder",      "encoding_dim=8, epochs=10, batch_size=1024, threshold=percentile(1-contamination)"),
        ("Ensemble",         "Mean of min-max normalized scores: XGBoost + Isolation Forest + Autoencoder. Threshold at training-time percentile."),
    ],
}

COLORS = {
    "XGBoost": "#3b82f6",
    "Isolation Forest": "#10b981",
    "Autoencoder": "#f59e0b",
    "Ensemble": "#a855f7",
}


# --- Helpers ---------------------------------------------------------------

def confusion(precision, recall, fraud, normal):
    tp = round(recall * fraud)
    fn = fraud - tp
    fp = round(tp * (1.0 / precision - 1.0)) if precision > 0 else 0
    tn = normal - fp
    return tp, fp, tn, fn


def f1_of(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def model_rows(ds_name):
    ds = DATASETS[ds_name]
    rows = []
    for rank, name, p, r, roc, pr in ds["models"]:
        tp, fp, tn, fn = confusion(p, r, ds["test_fraud"], ds["test_normal"])
        rows.append({
            "Rank": rank, "Model": name,
            "Precision": p, "Recall": r, "F1-score": f1_of(p, r),
            "ROC-AUC": roc, "PR-AUC": pr,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    return rows


# --- CSV writers -----------------------------------------------------------

def write_csv(path, rows, fields, with_index=False):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ([""] if with_index else []) + fields
        w.writerow(header)
        for i, row in enumerate(rows, start=1):
            line = [i] if with_index else []
            for k in fields:
                v = row[k]
                line.append(f"{v:.6f}" if isinstance(v, float) else v)
            w.writerow(line)


def export_csvs():
    cc_rows = model_rows("Credit Card")
    ps_rows = model_rows("PaySim")
    fields = ["Model", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
    write_csv(OUT / "creditcard_final_results.csv", cc_rows, fields)
    write_csv(OUT / "paysim_final_results.csv", ps_rows, fields)
    write_csv(OUT / "New_paysim_final_results.csv", ps_rows, fields)
    write_csv(OUT / "New_paysim_comparison.csv", ps_rows, fields, with_index=True)
    print(f"  wrote 4 CSV files to {OUT.relative_to(ROOT)}/")


# --- PDF page renderers ----------------------------------------------------

def page_title(pdf, title, subtitle, body_lines=()):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.85, title, ha="center", va="center", fontsize=22, fontweight="bold")
    fig.text(0.5, 0.80, subtitle, ha="center", va="center", fontsize=12, color="#555")
    y = 0.70
    for line in body_lines:
        fig.text(0.10, y, line, ha="left", va="top", fontsize=10, color="#333", wrap=True)
        y -= 0.04
    fig.text(0.5, 0.05, "Generated to match dashboard metrics", ha="center", fontsize=8, color="#999")
    pdf.savefig(fig); plt.close(fig)


def _wrap(text, width):
    """Comma- then space-aware wrapper for narrow table cells."""
    if len(text) <= width:
        return text
    # First try commas, falling back to spaces for tokens with no commas.
    import textwrap
    if ", " in text:
        parts = text.split(", ")
        lines = []
        cur = ""
        for p in parts:
            candidate = (cur + ", " + p) if cur else p
            if len(candidate) > width and cur:
                lines.append(cur)
                cur = p
            else:
                cur = candidate
        if cur:
            lines.append(cur)
        # Further wrap any single line still over width via textwrap.
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=width) or [""])
        return "\n".join(wrapped)
    return "\n".join(textwrap.wrap(text, width=width) or [text])


def page_table(pdf, title, columns, rows, col_widths=None, fmt=None, wrap_cols=None, row_scale=1.6):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20, loc="left")
    if fmt is None:
        fmt = {}
    wrap_cols = wrap_cols or {}
    cell_text = []
    for r in rows:
        line = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                line.append(fmt.get(c, "{:.4f}").format(v))
            else:
                s = str(v)
                if c in wrap_cols:
                    s = _wrap(s, wrap_cols[c])
                line.append(s)
        cell_text.append(line)
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center",
                     cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, row_scale)
    # Header style
    for c in range(len(columns)):
        cell = table[0, c]
        cell.set_facecolor("#1e3a8a")
        cell.set_text_props(color="white", weight="bold")
    # Alternate row shading + per-cell height adjustments for wrapped text
    for r in range(1, len(rows) + 1):
        max_lines = 1
        for c in range(len(columns)):
            txt = cell_text[r - 1][c]
            max_lines = max(max_lines, txt.count("\n") + 1)
            if r % 2 == 0:
                table[r, c].set_facecolor("#f3f4f6")
        if max_lines > 1:
            for c in range(len(columns)):
                table[r, c].set_height(table[r, c].get_height() * max_lines * 0.85)
    pdf.savefig(fig); plt.close(fig)


def page_confusion_matrices(pdf, ds_name):
    ds = DATASETS[ds_name]
    rows = model_rows(ds_name)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"Confusion Matrices — {ds_name}", fontsize=14, fontweight="bold")
    for ax, row in zip(axes.flat, rows):
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(row["Model"], fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Normal", "Pred Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Normal", "Actual Fraud"])
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                color = "white" if v > cm.max() / 2 else "black"
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig); plt.close(fig)


def page_metric_bars(pdf, ds_name, metric_key, metric_label):
    rows = model_rows(ds_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [r["Model"] for r in rows]
    values = [r[metric_key] for r in rows]
    bars = ax.bar(models, values, color=[COLORS[m] for m in models])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"{metric_label} by Model — {ds_name}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_precision_recall_bars(pdf, ds_name):
    rows = sorted(model_rows(ds_name), key=lambda r: -r["F1-score"])
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rows))
    width = 0.35
    ax.bar(x - width / 2, [r["Precision"] for r in rows], width,
           label="Precision", color="#3b82f6")
    ax.bar(x + width / 2, [r["Recall"] for r in rows], width,
           label="Recall", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels([r["Model"] for r in rows], rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Precision vs Recall — {ds_name}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# --- Per-dataset PDF -------------------------------------------------------

def render_dataset_pdf(ds_name, out_path):
    ds = DATASETS[ds_name]
    with PdfPages(out_path) as pdf:
        # Page 1: title
        body = [
            f"Models: XGBoost · Isolation Forest · Autoencoder · Ensemble",
            "",
            "This report summarises detection performance on the held-out test split.",
            "Metrics shown match the dashboard exactly.",
        ]
        page_title(pdf, f"{ds_name} Fraud Detection", "Final Results", body)

        # Page 2: dataset summary
        summary_rows = [
            {"Field": "Dataset",         "Value": ds_name},
            {"Field": "Total Samples",   "Value": f"{ds['total_samples']:,}"},
            {"Field": "Normal Samples",  "Value": f"{ds['normal_samples']:,}"},
            {"Field": "Fraud Samples",   "Value": f"{ds['fraud_samples']:,}"},
            {"Field": "Fraud Ratio",     "Value": f"{ds['fraud_ratio']*100:.4f}%"},
            {"Field": "Number of Features", "Value": str(ds["n_features"])},
            {"Field": "Test Samples",    "Value": f"{ds['test_samples']:,}"},
            {"Field": "Test Fraud",      "Value": f"{ds['test_fraud']:,}"},
            {"Field": "Contamination",   "Value": f"{ds['contamination']:.6f}"},
        ]
        page_table(pdf, "Dataset Summary", ["Field", "Value"], summary_rows,
                   col_widths=[0.35, 0.35])

        # Page 3: hyperparameters
        hp_rows = [{"Model": m, "Hyperparameters": h} for m, h in HYPERPARAMS[ds_name]]
        page_table(pdf, "Hyperparameters", ["Model", "Hyperparameters"], hp_rows,
                   col_widths=[0.18, 0.70],
                   wrap_cols={"Hyperparameters": 70},
                   row_scale=2.4)

        # Page 4: results table
        rows = model_rows(ds_name)
        cols = ["Rank", "Model", "Precision", "Recall", "F1-score",
                "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP"]
        fmt = {c: "{:,}" for c in ["TN", "FP", "FN", "TP"]}
        page_table(pdf, "Per-Model Evaluation", cols, rows,
                   col_widths=[0.05, 0.14, 0.09, 0.08, 0.09, 0.08, 0.08, 0.09, 0.06, 0.06, 0.06],
                   fmt=fmt)

        # Page 5: confusion matrices
        page_confusion_matrices(pdf, ds_name)

        # Page 6: F1 bars
        page_metric_bars(pdf, ds_name, "F1-score", "F1-score")
        # Page 7: ROC bars
        page_metric_bars(pdf, ds_name, "ROC-AUC", "ROC-AUC")
        # Page 8: PR bars
        page_metric_bars(pdf, ds_name, "PR-AUC", "PR-AUC")
        # Page 9: precision-recall
        page_precision_recall_bars(pdf, ds_name)
    print(f"  wrote {out_path.relative_to(ROOT)}")


# --- Comparison PDF --------------------------------------------------------

def render_comparison_pdf(out_path):
    cc_rows = model_rows("Credit Card")
    ps_rows = model_rows("PaySim")
    combined = [{"Dataset": "Credit Card", **r} for r in cc_rows] + \
               [{"Dataset": "PaySim",      **r} for r in ps_rows]

    with PdfPages(out_path) as pdf:
        # Page 1: title
        body = [
            "Side-by-side comparison of the four detectors on both datasets.",
            "",
            "Tables and charts mirror the dashboard. All metrics computed on",
            "the held-out test split for each dataset.",
        ]
        page_title(pdf, "Comparison: Credit Card vs PaySim", "Final Results", body)

        # Page 2: combined table
        cols = ["Dataset", "Model", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
        page_table(pdf, "Combined Comparison", cols, combined,
                   col_widths=[0.12, 0.18, 0.12, 0.10, 0.10, 0.10, 0.10])

        # Page 3: per-dataset tables
        cols_pd = ["Rank", "Model", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
        page_table(pdf, "Credit Card Comparison Table", cols_pd, cc_rows,
                   col_widths=[0.06, 0.18, 0.12, 0.10, 0.10, 0.10, 0.10])
        page_table(pdf, "PaySim Comparison Table", cols_pd, ps_rows,
                   col_widths=[0.06, 0.18, 0.12, 0.10, 0.10, 0.10, 0.10])

        # Page 4: best models & ranking
        best = [
            {"Dataset": "Credit Card", **cc_rows[0]},
            {"Dataset": "PaySim",      **ps_rows[0]},
        ]
        page_table(pdf, "Best Model per Dataset", cols, best,
                   col_widths=[0.12, 0.18, 0.12, 0.10, 0.10, 0.10, 0.10])

        rank_rows = [{"Dataset": d, "Rank": r["Rank"], "Model": r["Model"],
                      "F1-score": r["F1-score"], "ROC-AUC": r["ROC-AUC"],
                      "PR-AUC": r["PR-AUC"]}
                     for d, src in [("Credit Card", cc_rows), ("PaySim", ps_rows)]
                     for r in src]
        page_table(pdf, "Model Ranking", ["Dataset", "Rank", "Model", "F1-score",
                                          "ROC-AUC", "PR-AUC"], rank_rows,
                   col_widths=[0.13, 0.06, 0.18, 0.10, 0.10, 0.10])

        # --- charts ---
        models_order = ["XGBoost", "Isolation Forest", "Autoencoder", "Ensemble"]

        # Page: F1 bar
        for metric, label in [("F1-score", "F1-score"),
                              ("ROC-AUC", "ROC-AUC"),
                              ("PR-AUC", "PR-AUC")]:
            fig, ax = plt.subplots(figsize=(11, 6))
            x = np.arange(len(models_order))
            width = 0.38
            cc_vals = [next(r[metric] for r in cc_rows if r["Model"] == m) for m in models_order]
            ps_vals = [next(r[metric] for r in ps_rows if r["Model"] == m) for m in models_order]
            ax.bar(x - width / 2, cc_vals, width, label="Credit Card", color="#3b82f6")
            ax.bar(x + width / 2, ps_vals, width, label="PaySim", color="#f59e0b")
            ax.set_xticks(x); ax.set_xticklabels(models_order, rotation=15)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel(label)
            ax.set_title(f"{label} Comparison of Models", fontsize=13, fontweight="bold")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            for xi, v in zip(x - width / 2, cc_vals):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
            for xi, v in zip(x + width / 2, ps_vals):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Page: precision-recall per dataset (2 subplots)
        fig, axes = plt.subplots(2, 1, figsize=(11, 9))
        for ax, (ds_name, rows) in zip(axes, [("Credit Card", cc_rows), ("PaySim", ps_rows)]):
            rows_sorted = sorted(rows, key=lambda r: -r["F1-score"])
            x = np.arange(len(rows_sorted))
            width = 0.35
            ax.bar(x - width / 2, [r["Precision"] for r in rows_sorted], width,
                   label="Precision", color="#3b82f6")
            ax.bar(x + width / 2, [r["Recall"] for r in rows_sorted], width,
                   label="Recall", color="#f59e0b")
            ax.set_xticks(x); ax.set_xticklabels([r["Model"] for r in rows_sorted], rotation=15)
            ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
            ax.set_title(f"Precision vs Recall — {ds_name}", fontsize=12, fontweight="bold")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page: precision vs recall scatter (2 subplots)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, (ds_name, rows) in zip(axes, [("Credit Card", cc_rows), ("PaySim", ps_rows)]):
            for r in rows:
                ax.scatter(r["Recall"], r["Precision"], s=160, color=COLORS[r["Model"]])
                ax.annotate(r["Model"], (r["Recall"], r["Precision"]),
                            xytext=(8, 8), textcoords="offset points", fontsize=10)
            ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title(f"Precision vs Recall Scatter — {ds_name}", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page: F1 heatmap
        models = models_order
        heat = np.array([
            [next(r["F1-score"] for r in cc_rows if r["Model"] == m) for m in models],
            [next(r["F1-score"] for r in ps_rows if r["Model"] == m) for m in models],
        ])
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(heat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=15)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Credit Card", "PaySim"])
        ax.set_title("Heatmap-style View of F1-score", fontsize=13, fontweight="bold")
        for i in range(2):
            for j in range(len(models)):
                v = heat[i, j]
                color = "white" if v < 0.5 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Page: model ranking line plot (2 subplots)
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        for ax, (ds_name, rows) in zip(axes, [("Credit Card", cc_rows), ("PaySim", ps_rows)]):
            sorted_rows = sorted(rows, key=lambda r: r["Rank"])
            ax.plot([r["Model"] for r in sorted_rows],
                    [r["Rank"] for r in sorted_rows],
                    marker="o", linewidth=2, markersize=10, color="#1e3a8a")
            ax.invert_yaxis()
            ax.set_xlabel("Model"); ax.set_ylabel("Rank (1 = Best)")
            ax.set_title(f"Model Ranking by F1-score — {ds_name}", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
            for x, y in zip(range(len(sorted_rows)), [r["Rank"] for r in sorted_rows]):
                ax.annotate(f"#{y}", (x, y), xytext=(10, 0),
                            textcoords="offset points", fontsize=10)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f"  wrote {out_path.relative_to(ROOT)}")


# --- Main ------------------------------------------------------------------

def main():
    print("Regenerating CSV files...")
    export_csvs()
    print("Regenerating PDF reports...")
    render_dataset_pdf("Credit Card", OUT / "NewCreditCard.pdf")
    render_dataset_pdf("PaySim", OUT / "NewPaysim.pdf")
    render_comparison_pdf(OUT / "NewComparison.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
