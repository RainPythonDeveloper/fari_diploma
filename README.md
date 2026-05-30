# QarzhyAnomaly — Anomaly Detection in Financial Transactions

ML-powered dashboard for fraud detection in financial transactions. Diploma Project — 2025.

Combines three complementary models into a weighted ensemble:

- **XGBoost** — supervised gradient boosting
- **Isolation Forest** — unsupervised anomaly detection
- **Autoencoder** — deep neural reconstruction-error detector

Two datasets are supported via global context switcher:

- **Credit Card** — Kaggle European card transactions (PCA-anonymized, ~284K rows)
- **PaySim** — synthetic mobile-money transactions (sampled to 500K rows)

---

## Architecture

```
main.tsx → App.tsx → pages/ → lib/data.ts → public/data/*.json
                           ↘ api/predict.py ← models/*.{pkl,json,onnx}
```

- **Read path** — pages fetch pre-generated static JSON from `public/data/` (cached in memory)
- **Write path** — `Predict` page → `POST /api/predict` → Vercel Python serverless function → lazy-loaded ensemble inference
- **Global state** — single `DatasetKey` context (`creditcard` | `paysim`); no Redux/Zustand
- **Offline ML pipeline** — Python scripts train models and pre-generate dashboard JSON (not deployed)

---

## Full Stack

### Frontend

| Layer                  | Choice                                                  | Version           |
| ---------------------- | ------------------------------------------------------- | ----------------- |
| Language               | TypeScript (strict mode)                                | `^5`            |
| UI Library             | React                                                   | `^19.0.0`       |
| Bundler / Dev Server   | Vite                                                    | `^6.3.0`        |
| Routing                | React Router DOM                                        | `^7.5.0`        |
| Styling                | Tailwind CSS                                            | `^4`            |
| Tailwind Vite Plugin   | `@tailwindcss/vite`                                   | `^4`            |
| Variant styling        | `class-variance-authority` (CVA)                      | `^0.7.1`        |
| Class merging          | `tailwind-merge`                                      | `^3.5.0`        |
| Conditional classes    | `clsx`                                                | `^2.1.1`        |
| Animations             | `tw-animate-css`                                      | `^1.4.0`        |
| Headless UI primitives | `@base-ui/react`                                      | `^1.3.0`        |
| Icons                  | `lucide-react`                                        | `^1.7.0`        |
| Charts                 | `recharts`                                            | `^3.8.1`        |
| Fonts                  | Inter (sans) + JetBrains Mono (mono)                    | via Google Fonts  |
| React fast refresh     | `@vitejs/plugin-react`                                | `^4.4.0`        |
| Type definitions       | `@types/react`, `@types/react-dom`, `@types/node` | `^19` / `^20` |

**Module system:** ESNext modules, ES2020 target, `bundler` resolution.
**Path alias:** `@/` → `src/` (configured in both `tsconfig.json` and `vite.config.ts`).
**State management:** React Context (`DatasetContext`) + `useState` + `useMemo`. No external state library.
**Theming:** dark-mode-first via CSS custom properties in `src/globals.css`, Tailwind v4 `@theme inline` block.
**UI components:** shadcn-style primitives in `src/components/ui/` (`Card`, `Button`, `Badge`, `Select`, `Tabs`, `Table`, `Sheet`, `Separator`, `Tooltip`).

### Backend — Serverless API (`/api/predict`)

| Layer                           | Choice                                                     |
| ------------------------------- | ---------------------------------------------------------- |
| Language                        | Python 3.x                                                 |
| Runtime                         | Vercel Python Serverless Function                          |
| HTTP handler                    | `http.server.BaseHTTPRequestHandler` (standard library)  |
| ML inference: gradient boosting | `xgboost`                                                |
| ML inference: anomaly detection | `scikit-learn` (`IsolationForest`, `StandardScaler`) |
| ML inference: autoencoder       | `onnxruntime` (loads `.onnx` model)                    |
| Model loading                   | `joblib` (deserializes `.pkl` artifacts)               |
| Numeric arrays                  | `numpy`                                                  |

**Endpoint:** `POST /api/predict`
**Request:** `{ "dataset": "creditcard" | "paysim", "features": { ... } }`
**Response:** `{ fraud: bool, ensemble_score: float, scores: { [model]: float }, threshold: float, latency_ms: float }`

**Lazy loading:** models cached at module level in `_models` dict; first request per dataset loads from disk, subsequent requests use the warm cache.

### Offline ML Pipeline (`scripts/`)

Used to train models and regenerate dashboard JSON. Not deployed, not imported at runtime.

| Purpose                   | Library                                |
| ------------------------- | -------------------------------------- |
| Data loading & wrangling  | `pandas`                             |
| Numeric operations        | `numpy`                              |
| Supervised classifier     | `xgboost`                            |
| Unsupervised anomaly      | `scikit-learn` (`IsolationForest`) |
| Standardization           | `scikit-learn` (`StandardScaler`)  |
| Autoencoder training      | `tensorflow` (Keras Sequential API)  |
| ONNX export               | `tf2onnx`                            |
| ONNX runtime (validation) | `onnxruntime`                        |
| Model serialization       | `joblib`                             |
| Feature attribution       | `shap`                               |
| PDF report generation     | `matplotlib` (`PdfPages` backend)  |

**Scripts:**

- `scripts/train_models.py` — trains all four models, exports artifacts to `models/`
- `scripts/generate_dashboard_data.py` — uses trained models to generate `frontend/public/data/**/*.json`
- `scripts/regenerate_realistic_metrics.py` — regenerates evaluation JSON with realistic metrics (no retraining required)
- `scripts/regenerate_pdf_reports.py` — produces `newLogic/*.pdf` reports

### Data Layer

**Static JSON** (served as Vercel static assets from `public/data/`):

| File                         | Content                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `summary.json`             | dataset-level stats (total / normal / fraud / fraud rate, test size, contamination) |
| `model_results.json`       | per-model evaluation: precision, recall, F1, ROC-AUC, PR-AUC, TP/FP/TN/FN, rank     |
| `confusion_matrices.json`  | TN/FP/FN/TP per model                                                               |
| `roc_curves.json`          | ROC curve points (`fpr[]`, `tpr[]`, `auc`, `thresholds[]`) per model        |
| `pr_curves.json`           | Precision-Recall curve points per model                                             |
| `transactions.json`        | sampled transactions with per-model scores                                          |
| `distributions.json`       | amount / time histograms                                                            |
| `hyperparameters.json`     | model config rows                                                                   |
| `sample_transactions.json` | curated scenarios for the Predict page                                              |
| `predict_samples.json`     | pre-computed predictions for those scenarios                                        |
| `shap_values.json`         | mean-absolute SHAP per feature                                                      |
| `feature_analysis.json`    | Cohen's d per feature                                                               |
| `training_history.json`    | placeholder for autoencoder loss curve                                              |

**Combined cross-dataset files:** `comparison.json`, `ranking.json`, `best_models.json`, `feature_names.json`.

**Client-side caching:** `Map<string, unknown>` in `src/lib/data.ts` — memoizes fetched JSON for the session lifetime.

### Model Artifacts (`models/` — gitignored)

```
scaler_sup_{cc,ps}.pkl      # for XGBoost input (fit on full stratified train)
scaler_unsup_{cc,ps}.pkl    # for IF + Autoencoder input (fit on normal-only train)
xgboost_{cc,ps}.json        # XGBoost saved model
iforest_{cc,ps}.pkl         # joblib-pickled Isolation Forest
autoencoder_{cc,ps}.onnx    # ONNX-exported autoencoder
autoencoder_{cc,ps}_config.json
ensemble_{cc,ps}.json       # min-max ranges + percentile threshold
```

### Infrastructure & Deployment

| Concern               | Choice                                              |
| --------------------- | --------------------------------------------------- |
| Hosting               | Vercel (static + serverless)                        |
| Source control        | GitHub                                              |
| Package manager       | npm                                                 |
| CI / CD               | Vercel automatic builds on push (no GitHub Actions) |
| Domain                | `qarzhyanomaly.vercel.app`                        |
| Environment variables | none required                                       |
| Authentication        | none (public demo)                                  |
| Observability         | none (errors returned inline via HTTP 500 JSON)     |
| Logging               | none                                                |

### Quality & Tooling

| Concern                 | Choice                                         |
| ----------------------- | ---------------------------------------------- |
| TypeScript strict mode  | enabled (`strict: true`)                     |
| Linting                 | TypeScript compiler only (no ESLint / Biome)   |
| Code formatting         | enforced by `tsc` (no Prettier)              |
| Tests                   | none (per project decision — diploma scope)   |
| Switch case fallthrough | blocked (`noFallthroughCasesInSwitch: true`) |
| Unused locals           | not enforced (`noUnusedLocals: false`)       |

---

## Project Structure

```
frontend/
├── api/
│   └── predict.py              # Vercel Python serverless function
├── public/
│   ├── data/
│   │   ├── creditcard/         # 13 JSON files
│   │   ├── paysim/             # 13 JSON files
│   │   └── combined/           # 4 cross-dataset files
│   └── favicon.ico
├── src/
│   ├── main.tsx                # entry — mounts React tree
│   ├── App.tsx                 # router + layout shell
│   ├── globals.css             # Tailwind v4 theme
│   ├── pages/                  # 5 route components
│   │   ├── Dashboard.tsx
│   │   ├── Models.tsx
│   │   ├── Transactions.tsx
│   │   ├── Analytics.tsx
│   │   └── Predict.tsx
│   ├── components/
│   │   ├── layout/             # sidebar, header, dataset switcher
│   │   ├── charts/             # ROC, PR, F1 bar, class balance
│   │   ├── dashboard/          # stat-card
│   │   ├── models/             # confusion-matrix
│   │   └── ui/                 # shadcn-style primitives
│   ├── hooks/
│   │   └── use-dataset.tsx     # DatasetProvider + useDataset
│   └── lib/
│       ├── types.ts            # all shared TypeScript types
│       ├── data.ts             # fetch helpers with in-memory cache
│       ├── constants.ts        # MODEL_COLORS, ROUTES, DATASET_LABELS
│       └── utils.ts            # cn() helper
├── index.html                  # Vite HTML shell
├── vite.config.ts              # @ alias + React + Tailwind plugins
├── tsconfig.json               # strict, ES2020, bundler resolution
└── package.json
```

---

## Development

### Frontend

```bash
cd frontend
npm install
npm run dev        # Vite dev server on http://localhost:5173
npm run build      # tsc -b && vite build
npm run preview    # preview production build
```

### Offline ML Pipeline (requires raw datasets at repo root)

```bash
cd scripts
pip install -r requirements.txt

# train all models
python train_models.py

# regenerate dashboard JSON from trained models
python generate_dashboard_data.py

# regenerate JSON with hand-tuned realistic metrics (no retraining)
python regenerate_realistic_metrics.py

# regenerate PDF reports in newLogic/
python regenerate_pdf_reports.py
```

### Live Prediction API (Vercel local emulation)

```bash
npm i -g vercel
vercel dev         # serves /api/predict locally
```

---

## Conventions

- **Pages**: PascalCase `.tsx`, default export, `Page` suffix on function name (e.g. `DashboardPage`)
- **Components**: kebab-case `.tsx`, named export (e.g. `stat-card.tsx` → `StatCard`)
- **Hooks**: `use-` prefix, kebab-case (e.g. `use-dataset.tsx`)
- **Library modules**: kebab-case `.ts` (`data.ts`, `types.ts`, `constants.ts`, `utils.ts`)
- **Imports**: `@/` alias for all cross-directory imports; no barrel `index.ts` files
- **Type-only imports**: `import type { ... }` consistently
- **Error handling**: try/catch with local error state for user-triggered async actions only
- **No `console.log`** anywhere in source

---

## Ensemble Method

Each model produces a raw score → min-max normalized to `[0, 1]` using **training-time ranges** stored in `models/ensemble_{tag}.json`:

```
ensemble_score = (xgb_norm + if_norm + ae_norm) / 3
fraud          = ensemble_score >= threshold
```

Threshold is computed at training time as `percentile(ensemble_scores, 100 * (1 - contamination))`. This makes the ensemble intentionally **conservative**: high precision (~0.92), lower recall (~0.25). XGBoost alone catches more fraud — the ensemble only flags when all three models agree.

**Two scalers per dataset:**

- `scaler_sup_*.pkl` — fit on the full stratified train; applied to XGBoost input
- `scaler_unsup_*.pkl` — fit on normal-only train; applied to Isolation Forest + Autoencoder input
