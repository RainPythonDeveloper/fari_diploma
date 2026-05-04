# Codebase Structure

**Analysis Date:** 2026-05-04

## Directory Layout

```
fari_diploma/                        # Project root
├── frontend/                        # Vite + React SPA + Vercel Python function
│   ├── api/                         # Vercel serverless functions
│   │   └── predict.py               # POST /api/predict — ensemble ML inference
│   ├── public/                      # Static assets served at /
│   │   └── data/                    # Pre-built JSON data for dashboard
│   │       ├── creditcard/          # 10 JSON files for Credit Card dataset
│   │       ├── paysim/              # 10 JSON files for PaySim dataset
│   │       └── combined/            # Cross-dataset comparison JSON files
│   ├── src/                         # React application source
│   │   ├── main.tsx                 # App entry point — mounts React tree
│   │   ├── App.tsx                  # Router + layout shell
│   │   ├── globals.css              # Tailwind v4 theme, CSS custom properties
│   │   ├── pages/                   # Full-page route components
│   │   │   ├── Dashboard.tsx        # Overview stats, ROC, F1, class balance
│   │   │   ├── Models.tsx           # Model ranking table, confusion matrices
│   │   │   ├── Transactions.tsx     # Paginated transaction explorer
│   │   │   ├── Analytics.tsx        # ROC/PR curves, radar, scatter charts
│   │   │   └── Predict.tsx          # Real-time single-transaction prediction
│   │   ├── components/
│   │   │   ├── layout/              # Persistent chrome
│   │   │   │   ├── sidebar.tsx      # Desktop sidebar + mobile sheet drawer
│   │   │   │   ├── header.tsx       # Top bar with page title + mobile nav
│   │   │   │   └── dataset-switcher.tsx  # Toggle between creditcard/paysim
│   │   │   ├── charts/              # Recharts-based visualization components
│   │   │   │   ├── roc-chart.tsx    # ROC curve multi-model line chart
│   │   │   │   ├── pr-chart.tsx     # Precision-Recall curve chart
│   │   │   │   ├── f1-bar-chart.tsx # F1 scores horizontal bar chart
│   │   │   │   └── class-balance-chart.tsx  # Fraud vs normal pie/bar
│   │   │   ├── dashboard/
│   │   │   │   └── stat-card.tsx    # KPI stat card with icon
│   │   │   ├── models/
│   │   │   │   └── confusion-matrix.tsx  # Confusion matrix grid component
│   │   │   ├── predict/             # (empty — predict logic lives in page)
│   │   │   ├── transactions/        # (empty — transaction logic lives in page)
│   │   │   └── ui/                  # Primitive UI components (shadcn-style)
│   │   │       ├── card.tsx
│   │   │       ├── button.tsx
│   │   │       ├── badge.tsx
│   │   │       ├── select.tsx
│   │   │       ├── tabs.tsx
│   │   │       ├── table.tsx
│   │   │       ├── sheet.tsx        # Mobile drawer primitive
│   │   │       ├── separator.tsx
│   │   │       └── tooltip.tsx
│   │   ├── hooks/
│   │   │   └── use-dataset.tsx      # DatasetProvider + useDataset context hook
│   │   └── lib/
│   │       ├── types.ts             # All TypeScript interfaces and union types
│   │       ├── data.ts              # Async fetch helpers with in-memory cache
│   │       ├── constants.ts         # MODEL_COLORS, ROUTES, DATASET_LABELS
│   │       └── utils.ts             # cn() helper (clsx + tailwind-merge)
│   ├── dist/                        # Vite build output (committed, for Vercel)
│   ├── index.html                   # HTML shell with #root mount point
│   ├── vite.config.ts               # Vite config with @/ alias and Tailwind plugin
│   ├── tsconfig.json                # TypeScript config (ES2020, strict, bundler)
│   └── package.json                 # Dependencies and dev scripts
├── models/                          # Trained model artifacts (gitignored on Vercel deploy)
│   │                                # scaler_cc.pkl, xgboost_cc.json, iforest_cc.pkl,
│   │                                # autoencoder_cc.onnx, ensemble_cc.json (+ ps variants)
├── scripts/                         # Offline data pipeline (not deployed)
│   ├── train_models.py              # Train all models → save to ../models/
│   ├── generate_dashboard_data.py   # Load models → write JSON to ../frontend/public/data/
│   └── requirements.txt             # Python deps for scripts (pandas, sklearn, xgboost, etc.)
├── results/                         # Training result CSVs and Excel summary
│   ├── creditcard_results.csv
│   ├── paysim_results.csv
│   ├── combined_results.csv
│   ├── model_ranking.csv
│   ├── best_models.csv
│   ├── dataset_summary.csv
│   └── diploma_results.xlsx
├── *.ipynb                          # Jupyter exploration notebooks (not runtime)
├── creditcard.csv                   # Raw dataset (~150MB, not deployed)
├── PS_20174392719_*.csv             # Raw PaySim dataset (~493MB, not deployed)
└── .gitignore
```

## Directory Purposes

**`frontend/api/`:**
- Purpose: Vercel Python serverless functions; each `.py` file becomes an API route
- Contains: `predict.py` → `POST /api/predict`
- Key files: `frontend/api/predict.py`

**`frontend/public/data/`:**
- Purpose: Static JSON served at `/data/...` — the entire historical dashboard data
- Contains: One subdirectory per dataset (`creditcard/`, `paysim/`, `combined/`)
- Key files: `summary.json`, `model_results.json`, `transactions.json`, `roc_curves.json`, `pr_curves.json`, `confusion_matrices.json`, `distributions.json`, `hyperparameters.json`, `sample_transactions.json`, `training_history.json`; combined: `comparison.json`, `ranking.json`, `best_models.json`

**`frontend/src/pages/`:**
- Purpose: One file per route; each page owns its `useEffect` data-fetch lifecycle and local state
- Contains: Five page components mapped 1:1 to routes in `App.tsx`

**`frontend/src/components/charts/`:**
- Purpose: Recharts wrappers for specific chart types; receive typed data props, render nothing else
- Contains: Four chart components, all typed against interfaces from `lib/types.ts`

**`frontend/src/components/ui/`:**
- Purpose: Reusable design-system primitives with `cva` variants; not domain-specific
- Contains: shadcn-style components built on `@base-ui/react`

**`frontend/src/lib/`:**
- Purpose: Shared non-UI logic: types, data access, constants, utility functions
- Contains: `types.ts`, `data.ts`, `constants.ts`, `utils.ts`

**`frontend/src/hooks/`:**
- Purpose: React hooks; currently only global dataset state
- Contains: `use-dataset.tsx`

**`models/`:**
- Purpose: Serialized model artifacts referenced at runtime by `api/predict.py`
- Generated: Yes (by `scripts/train_models.py`)
- Committed: Partial (`.vercelignore` excludes large ONNX files to stay within Vercel limits)

**`scripts/`:**
- Purpose: One-time offline data pipeline; not deployed or imported at runtime
- Contains: Training and JSON-generation Python scripts with their own `requirements.txt`

## Key File Locations

**Entry Points:**
- `frontend/src/main.tsx`: React app mount; wraps tree with `BrowserRouter` and `DatasetProvider`
- `frontend/index.html`: HTML shell loaded by Vite
- `frontend/api/predict.py`: Serverless function handler for `POST /api/predict`

**Configuration:**
- `frontend/vite.config.ts`: Vite plugins (`react`, `tailwindcss`) and `@/` path alias
- `frontend/tsconfig.json`: TypeScript strict mode, `bundler` resolution, `@/*` paths
- `frontend/package.json`: Scripts (`dev`, `build`, `preview`), all dependencies

**Core Logic:**
- `frontend/src/lib/data.ts`: All data-fetching functions with in-memory cache
- `frontend/src/lib/types.ts`: All shared TypeScript types and interfaces
- `frontend/src/lib/constants.ts`: `MODEL_COLORS`, `ROUTES`, `DATASET_LABELS`
- `frontend/src/hooks/use-dataset.tsx`: Global active dataset state
- `frontend/api/predict.py`: Ensemble inference (`_load_model`, `predict`)

**Routing:**
- `frontend/src/App.tsx`: Route definitions and layout shell

**Static Data:**
- `frontend/public/data/{creditcard,paysim}/*.json`: Per-dataset dashboard data
- `frontend/public/data/combined/*.json`: Cross-dataset comparison data

## Naming Conventions

**Files:**
- Pages: PascalCase matching route label — `Dashboard.tsx`, `Predict.tsx`
- Components: kebab-case — `stat-card.tsx`, `roc-chart.tsx`, `dataset-switcher.tsx`
- Hooks: kebab-case with `use-` prefix — `use-dataset.tsx`
- Library modules: kebab-case — `data.ts`, `types.ts`, `constants.ts`, `utils.ts`
- Python serverless functions: lowercase — `predict.py`

**Directories:**
- Feature groupings under `components/`: lowercase plural — `charts/`, `layout/`, `ui/`
- Data subdirectories: lowercase matching `DatasetKey` — `creditcard/`, `paysim/`, `combined/`

**Exports:**
- Pages: default export (`export default function DashboardPage`)
- Components: named export matching PascalCase component name (`export function StatCard`)
- Hooks: named exports for provider and hook (`export function DatasetProvider`, `export function useDataset`)
- Library: named exports for each function/constant/type

**Types:**
- Interfaces: PascalCase — `ModelResult`, `Transaction`, `DatasetSummary`
- Union types: PascalCase — `DatasetKey`
- Local component prop interfaces: inline `interface` inside the file, PascalCase with `Props` suffix or `interface` inline

## Where to Add New Code

**New Page / Route:**
1. Create `frontend/src/pages/NewPage.tsx` (default export, PascalCase)
2. Add `<Route path="/new-path" element={<NewPage />} />` in `frontend/src/App.tsx`
3. Add nav entry to `NAV_ITEMS` in `frontend/src/components/layout/sidebar.tsx` and `PAGE_TITLES` in `frontend/src/components/layout/header.tsx`

**New Chart Component:**
- Implementation: `frontend/src/components/charts/my-chart.tsx` (named export)
- Props: type against interfaces from `frontend/src/lib/types.ts`
- Use `MODEL_COLORS` from `frontend/src/lib/constants.ts` for consistent coloring

**New Data JSON File:**
- Add fetch function to `frontend/src/lib/data.ts` following the `fetchJson<T>(path)` pattern
- Add corresponding TypeScript interface to `frontend/src/lib/types.ts`
- Place file at `frontend/public/data/{dataset}/filename.json` (generated by `scripts/generate_dashboard_data.py`)

**New Static Type:**
- Add to `frontend/src/lib/types.ts`

**New UI Primitive:**
- Add to `frontend/src/components/ui/` following the `cva` + `cn()` + `@base-ui/react` pattern

**New API Endpoint:**
- Create `frontend/api/endpoint-name.py` with a `handler` class extending `BaseHTTPRequestHandler`
- Vercel auto-routes it to `/api/endpoint-name`

**Shared Utilities:**
- Pure functions: `frontend/src/lib/utils.ts`
- App-wide constants: `frontend/src/lib/constants.ts`

## Special Directories

**`frontend/dist/`:**
- Purpose: Vite production build output
- Generated: Yes (by `npm run build`)
- Committed: Yes (committed to repo for Vercel deploy compatibility)

**`models/`:**
- Purpose: Serialized ML model files used by `api/predict.py` at runtime
- Generated: Yes (by `scripts/train_models.py`)
- Committed: Partially — large ONNX autoencoder files excluded via `.vercelignore`

**`results/`:**
- Purpose: CSV/XLSX outputs from model training for offline analysis
- Generated: Yes (by training scripts)
- Committed: Yes

**`frontend/public/data/`:**
- Purpose: Dashboard JSON data consumed by the React app as static assets
- Generated: Yes (by `scripts/generate_dashboard_data.py`)
- Committed: Yes (required for runtime; these are the dashboard's data source)

---

*Structure analysis: 2026-05-04*
