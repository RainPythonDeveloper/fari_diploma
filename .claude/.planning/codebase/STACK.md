# Technology Stack

**Analysis Date:** 2026-05-04

## Languages

**Primary:**
- TypeScript 5.x - All frontend source code in `frontend/src/`
- Python 3.x - Serverless API function (`frontend/api/predict.py`), ML training scripts (`scripts/train_models.py`, `scripts/generate_dashboard_data.py`)

**Secondary:**
- CSS (Tailwind utility classes) - Styling via `frontend/src/globals.css` and inline className strings
- HTML - Single entry point `frontend/index.html`

## Runtime

**Frontend Environment:**
- Node.js 24.11.1 (detected on dev machine)
- Browser target: ES2020, DOM

**Backend (Serverless):**
- Python 3.x (Vercel Python serverless runtime)
- Python 3.14.0 (detected on dev machine)

**Package Manager:**
- npm
- Lockfile: `frontend/package-lock.json` (present)

## Frameworks

**Core:**
- React 19.0.0 - UI library (`frontend/src/`)
- React Router DOM 7.5.0 - Client-side routing (`frontend/src/App.tsx`, `frontend/src/main.tsx`)

**Styling:**
- Tailwind CSS 4.x - Utility-first CSS, configured via `@tailwindcss/vite` plugin
- `class-variance-authority` 0.7.1 - Variant-based component styling
- `tailwind-merge` 3.5.0 - Conditional class merging utility
- `tw-animate-css` 1.4.0 - Animation utilities

**UI Components:**
- `@base-ui/react` 1.3.0 - Headless accessible primitives
- `lucide-react` 1.7.0 - Icon library
- Recharts 3.8.1 - Charting library for all data visualizations
- `@tanstack/react-table` 8.21.3 - Headless table library for `Transactions` page

**Build/Dev:**
- Vite 6.3.0 - Dev server and bundler (`frontend/vite.config.ts`)
- `@vitejs/plugin-react` 4.4.0 - React fast refresh plugin
- `@tailwindcss/vite` 4.x - Tailwind Vite plugin

**Testing:**
- Not detected — no test framework configured

## Key Dependencies

**Critical:**
- `xgboost` - XGBoost classifier loaded at prediction time in `frontend/api/predict.py`
- `scikit-learn` - IsolationForest, StandardScaler — loaded via `joblib` in `frontend/api/predict.py`
- `onnxruntime` - Runs Autoencoder `.onnx` model files at inference in `frontend/api/predict.py`
- `joblib` - Deserializes `.pkl` scaler and IsolationForest model files
- `numpy` - Array construction for model inference

**ML Training (scripts only, not deployed):**
- `pandas` - CSV data loading in `scripts/train_models.py`
- `tensorflow` + `tf2onnx` - Autoencoder training and ONNX export (`scripts/requirements.txt`)

**Infrastructure:**
- `react-router-dom` 7.5.0 - All page routing (`frontend/src/App.tsx`)
- Recharts 3.8.1 - Charts on Dashboard, Models, Analytics pages

## Configuration

**Environment:**
- No `.env` files detected in repository
- No environment variables required for frontend (all data is static JSON files in `public/data/`)
- No environment variables required for the serverless predict function (models loaded from filesystem path)

**Build:**
- `frontend/vite.config.ts` — Vite config with React and Tailwind plugins, `@` alias pointing to `frontend/src/`
- `frontend/tsconfig.json` — TypeScript strict mode, `bundler` module resolution, `@/*` path alias
- `frontend/.vercelignore` — Excludes `api/` directory from Vercel static asset deployment (API is deployed as serverless function separately)

**Path Alias:**
- `@/` resolves to `frontend/src/` in both TypeScript and Vite

## Platform Requirements

**Development:**
- Node.js (for Vite dev server and TypeScript compilation)
- Python 3.x with packages from `scripts/requirements.txt` for model training
- Pre-trained model artifacts must exist in `models/` directory before running the API

**Production:**
- Vercel hosting (static frontend + Python serverless function)
- Model artifacts (`.pkl`, `.json`, `.onnx`) deployed to `/var/task/models/` on Vercel
- Static JSON data files pre-generated into `frontend/public/data/` by `scripts/generate_dashboard_data.py`

---

*Stack analysis: 2026-05-04*
