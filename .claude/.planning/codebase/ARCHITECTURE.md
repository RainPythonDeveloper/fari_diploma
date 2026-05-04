# Architecture

**Analysis Date:** 2026-05-04

## Pattern Overview

**Overall:** Data-driven single-page application with a Python serverless backend for real-time ML inference

**Key Characteristics:**
- React 19 SPA (Vite) with client-side routing via React Router v7
- All historical ML metrics and transaction data served as static JSON files from `frontend/public/data/`
- Live fraud prediction routed through a single Vercel Python serverless function at `POST /api/predict`
- Global dataset context (`creditcard` | `paysim`) drives which JSON files are fetched and which feature set is sent to the API
- Offline-first read path: static JSON with in-memory cache; online write path: single API endpoint

## Layers

**Entry / Bootstrap:**
- Purpose: Mount React app into the DOM, wrap with providers
- Location: `frontend/src/main.tsx`
- Contains: `BrowserRouter`, `DatasetProvider`, `App`
- Depends on: `hooks/use-dataset`, `App`
- Used by: Browser

**Routing / Shell:**
- Purpose: Define page routes and persistent layout shell (sidebar + header)
- Location: `frontend/src/App.tsx`
- Contains: `<Routes>` with five `<Route>` entries, `Sidebar`, `Header`
- Depends on: all five page components, layout components
- Used by: `main.tsx`

**Global State:**
- Purpose: Provide the active dataset key (`creditcard` | `paysim`) to any component tree node
- Location: `frontend/src/hooks/use-dataset.tsx`
- Contains: `DatasetContext`, `DatasetProvider`, `useDataset` hook
- Depends on: React context API
- Used by: all five page components, `DatasetSwitcher`

**Pages:**
- Purpose: Full-page view components, each owning their own data-fetching lifecycle
- Location: `frontend/src/pages/`
- Contains: `Dashboard.tsx`, `Models.tsx`, `Transactions.tsx`, `Analytics.tsx`, `Predict.tsx`
- Depends on: `lib/data` (fetch helpers), `lib/types`, `lib/constants`, domain components, ui components
- Used by: `App.tsx` router

**Data Access:**
- Purpose: Typed async fetch helpers with a shared in-memory cache; all requests resolve to `/data/{dataset}/*.json` or `/data/combined/*.json`
- Location: `frontend/src/lib/data.ts`
- Contains: `getSummary`, `getModelResults`, `getConfusionMatrices`, `getRocCurves`, `getPrCurves`, `getTransactions`, `getDistributions`, `getHyperparameters`, `getSampleTransactions`, `getCombinedComparison`, `getCombinedRanking`, `getBestModels`
- Depends on: native `fetch`, `lib/types`
- Used by: page components

**Domain Components:**
- Purpose: Feature-specific UI building blocks consumed by pages; charts, stat cards, confusion matrices
- Location: `frontend/src/components/charts/`, `frontend/src/components/dashboard/`, `frontend/src/components/models/`, `frontend/src/components/predict/`, `frontend/src/components/transactions/`
- Contains: `F1BarChart`, `RocChart`, `PrChart`, `ClassBalanceChart`, `StatCard`, `ConfusionMatrix`
- Depends on: Recharts, `lib/types`, `lib/constants`, ui components
- Used by: page components

**Layout Components:**
- Purpose: Persistent chrome — sidebar navigation, top header, dataset toggle
- Location: `frontend/src/components/layout/`
- Contains: `Sidebar`, `MobileSidebar`, `Header`, `DatasetSwitcher`
- Depends on: `hooks/use-dataset`, `lib/constants`, React Router `Link`/`useLocation`, ui components
- Used by: `App.tsx`

**UI Primitives:**
- Purpose: Unstyled-primitive wrappers with Tailwind CVA variants (shadcn-style)
- Location: `frontend/src/components/ui/`
- Contains: `Card`, `Button`, `Badge`, `Select`, `Tabs`, `Table`, `Sheet`, `Separator`, `Tooltip`
- Depends on: `@base-ui/react`, `class-variance-authority`, `lib/utils`
- Used by: all other component layers

**Serverless API:**
- Purpose: Run ensemble ML inference on a single submitted transaction
- Location: `frontend/api/predict.py`
- Contains: `handler` (Vercel `BaseHTTPRequestHandler`), `predict()`, `_load_model()` with lazy in-process cache
- Depends on: `joblib`, `xgboost`, `onnxruntime`, `sklearn` scaler, model artifacts in `models/`
- Used by: `Predict.tsx` page via `POST /api/predict`

**Offline ML Pipeline (not runtime):**
- Purpose: Train models and generate all static dashboard JSON; run once, not deployed
- Location: `scripts/train_models.py`, `scripts/generate_dashboard_data.py`
- Contains: XGBoost, IsolationForest, Autoencoder training; JSON export to `frontend/public/data/`
- Depends on: `pandas`, `numpy`, `sklearn`, `xgboost`, `joblib`
- Used by: developer / data scientist manually

## Data Flow

**Static Dashboard Data (read path):**

1. User opens a page → page component calls `useDataset()` to get active `DatasetKey`
2. Page issues `useEffect` → calls one or more functions from `frontend/src/lib/data.ts`
3. `data.ts` checks in-memory `Map` cache; on miss calls `fetch(/data/{dataset}/{file}.json)`
4. Browser fetches pre-built JSON from `frontend/public/data/{creditcard|paysim}/{file}.json`
5. Page component updates local state → renders charts and tables

**Real-time Prediction (write path):**

1. User selects a scenario on `PredictPage` → feature values populate local state
2. User clicks "Analyze Transaction" → `predict()` POSTs `{ dataset, features }` to `/api/predict`
3. Vercel routes request to `frontend/api/predict.py`
4. `_load_model(dataset_tag)` lazy-loads scaler + XGBoost + IsolationForest + Autoencoder ONNX from `models/`
5. Each model produces a normalized 0–1 fraud score; weighted ensemble combines them
6. Handler returns `{ fraud, ensemble_score, scores, threshold, latency_ms }` as JSON
7. `PredictPage` renders verdict, score bar, and per-model breakdown chart

**Dataset Switching:**

1. User clicks dataset toggle in `DatasetSwitcher` (sidebar)
2. `setDataset()` updates `DatasetContext`
3. All subscribed page components re-run their `useEffect([dataset])` and re-fetch JSON for the new dataset

**State Management:**
- One global piece of state: `DatasetKey` in `DatasetContext` (`frontend/src/hooks/use-dataset.tsx`)
- All other state is local to each page component via `useState`
- No external state manager (Redux, Zustand, etc.)
- Derived/filtered state computed with `useMemo` (e.g., transaction filtering in `Transactions.tsx`)

## Key Abstractions

**DatasetKey:**
- Purpose: Union type `"creditcard" | "paysim"` that gates all data fetching and feature display
- Examples: `frontend/src/lib/types.ts`, `frontend/src/hooks/use-dataset.tsx`
- Pattern: Context value flows from `DatasetSwitcher` down through `useDataset()` to every consumer

**fetchJson cache:**
- Purpose: Simple request-level memoization to avoid re-fetching the same JSON on tab revisit
- Examples: `frontend/src/lib/data.ts` (`const cache = new Map<string, unknown>()`)
- Pattern: Check cache before fetch; store result after fetch

**Ensemble Prediction:**
- Purpose: Combine XGBoost (supervised), IsolationForest (unsupervised), and Autoencoder (reconstruction error) scores via weights from `ensemble_{dataset}.json`
- Examples: `frontend/api/predict.py` `predict()` function
- Pattern: Each model score normalized to [0,1]; final score = weighted sum; compare against threshold

**MODEL_COLORS:**
- Purpose: Canonical color map for consistent model identity across all charts
- Examples: `frontend/src/lib/constants.ts`
- Pattern: Record keyed by model name string, consumed directly by Recharts `<Cell fill>` props

**shadcn-style UI primitives:**
- Purpose: Composable, slot-based card/button/badge components using `cva` + `cn()`
- Examples: `frontend/src/components/ui/card.tsx`, `frontend/src/components/ui/button.tsx`
- Pattern: Named exports (`Card`, `CardHeader`, `CardContent`, etc.); `data-slot` attributes for CSS targeting

## Entry Points

**Browser Entry:**
- Location: `frontend/src/main.tsx`
- Triggers: Vite build output served by browser or `vite preview`
- Responsibilities: Mount React tree, provide `BrowserRouter` and `DatasetProvider`

**Serverless Function Entry:**
- Location: `frontend/api/predict.py`
- Triggers: `POST /api/predict` HTTP request via Vercel routing
- Responsibilities: Parse body, resolve dataset tag, run ensemble inference, return JSON

**HTML Shell:**
- Location: `frontend/index.html`
- Triggers: Vite dev server or static file serving
- Responsibilities: Provide `<div id="root">` mount point; links to `src/main.tsx`

## Error Handling

**Strategy:** Inline local error state in page components; no global error boundary

**Patterns:**
- `Predict.tsx`: try/catch around `fetch("/api/predict")`; sets `error` state string rendered as red banner
- Pages render `<LoadingSkeleton />` while data is `null` (e.g., `Dashboard.tsx`)
- `data.ts`: no catch; failed fetches will throw and surface as unhandled promise rejections
- API handler (`predict.py`): top-level try/except returns HTTP 500 with `{"error": str(e)}`

## Cross-Cutting Concerns

**Logging:** None (no logging library; API errors returned as JSON)
**Validation:** None on frontend feature inputs (values parsed with `parseFloat` / fallback `0`); no schema validation
**Authentication:** None — application is fully public
**Theming:** Dark-mode first via CSS custom properties in `frontend/src/globals.css`; Tailwind v4 `@theme inline` block; fonts Inter (sans) and JetBrains Mono (mono)

---

*Architecture analysis: 2026-05-04*
