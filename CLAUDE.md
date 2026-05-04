# CLAUDE.md

Guidelines for working in this codebase. Combines behavioral rules with project-specific context.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

**Project-specific traps to catch early:**
- "Update the dashboard data" — does this mean editing static JSON in `frontend/public/data/`, or re-running `scripts/generate_dashboard_data.py`? The script requires raw CSVs that are not in the repo. Ask which is meant.
- "Add a new model" — means both retraining (`scripts/train_models.py`) and updating static JSON. These are two separate offline steps. Clarify scope.
- Changes to `frontend/api/predict.py` affect live inference only. Changes to `scripts/` affect offline data generation only. They don't share code.

---

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

**Project-specific:**
- Data-fetching functions in `src/lib/data.ts` are intentionally 1–3 lines each, all delegating to `fetchJson<T>`. Keep them that way.
- Page components are large by design — sub-components live in the same file unless reused across pages. Don't extract a component just because it's long.
- There is no state manager. Don't introduce one. `useState` + `DatasetContext` is the entire state layer.

---

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

**Project-specific dead code to not touch unless asked:**
- `"use client"` directives on chart components — Next.js leftovers, harmless in Vite.
- `@tanstack/react-table` in `package.json` — installed but unused.
- `noUnusedLocals: false` in `tsconfig.json` — intentionally relaxed.

---

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

**Verification in this project (no tests exist):**
- Frontend changes: `npm run build` must pass (`tsc -b && vite build`). If it builds, types are valid.
- API changes: manually test `POST /api/predict` with a curl request or the Predict page.
- Static JSON changes: open the affected page in the browser and confirm data renders.
- Script changes: run the script and check the output files.

---

## Project Overview

A fraud detection ML dashboard — React 19 SPA (Vite) + Python Vercel serverless function.

- **Dashboard / Models / Analytics / Transactions** — read static pre-generated JSON from `frontend/public/data/`
- **Predict** — calls `POST /api/predict` → Python ensemble (XGBoost + IsolationForest + ONNX Autoencoder)
- **Two datasets**: `creditcard` and `paysim`. Global `DatasetKey` context drives all data fetching and feature display.
- **Offline pipeline** in `scripts/` — trains models, generates JSON. Not deployed, not imported at runtime.

---

## Commands

```bash
cd frontend
npm run dev        # Vite dev server
npm run build      # tsc -b && vite build
npm run preview    # Preview production build

# Offline pipeline (requires raw CSVs at project root)
cd scripts
pip install -r requirements.txt
python train_models.py              # → models/*.pkl/.json/.onnx
python generate_dashboard_data.py   # → frontend/public/data/**/*.json
```

---

## Architecture

**Read path:** page `useEffect` → `src/lib/data.ts` (in-memory Map cache) → `fetch(/data/{dataset}/*.json)` → React state

**Write path:** `Predict.tsx` → `POST /api/predict` → `frontend/api/predict.py` → lazy-loaded ensemble → JSON

**Global state:** only `DatasetKey` in `src/hooks/use-dataset.tsx`. All other state is local `useState` per page.

```
main.tsx → App.tsx → pages/ → lib/data.ts → public/data/
                           ↘ api/predict.py ← models/
```

---

## Stack

| Layer | Choice |
|---|---|
| Frontend | React 19, Vite 6, TypeScript strict |
| Routing | React Router v7 |
| Styling | Tailwind CSS v4, CVA, `cn()` |
| UI primitives | `@base-ui/react` + shadcn-style in `src/components/ui/` |
| Charts | Recharts 3 |
| API | Python `BaseHTTPRequestHandler` on Vercel |
| ML runtime | `xgboost`, `scikit-learn`, `onnxruntime`, `joblib` |

---

## Conventions

**File naming:**
- Pages: PascalCase (`Dashboard.tsx`) — default export, `Page` suffix on function name
- Components: kebab-case (`stat-card.tsx`) — named export
- Hooks: `use-` prefix (`use-dataset.tsx`) — named exports
- Lib: kebab-case (`data.ts`, `types.ts`) — named exports

**Imports:** `@/` alias for all cross-directory imports. No barrel files — exact file paths only.

**TypeScript:** `import type` for type-only imports. `strict: true`. No linter beyond `tsc`.

**No `console.log`** anywhere in source. API errors return HTTP 500 `{ "error": str(e) }`.

**Error handling (user-triggered async):**
```typescript
try {
  const res = await fetch(...);
  if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
  setResult(await res.json());
} catch (e) {
  setError(e instanceof Error ? e.message : "Failed");
}
```

---

## Where to Add Code

**New page:** `frontend/src/pages/NewPage.tsx` → `<Route>` in `App.tsx` → entry in `sidebar.tsx` (`NAV_ITEMS`) and `header.tsx` (`PAGE_TITLES`)

**New chart:** `frontend/src/components/charts/my-chart.tsx` — typed against `src/lib/types.ts`, use `MODEL_COLORS` from `src/lib/constants.ts`

**New JSON data type:** fetch function in `src/lib/data.ts` (1–3 lines, `fetchJson<T>` pattern) + interface in `src/lib/types.ts` + file at `frontend/public/data/{dataset}/file.json`

**New API endpoint:** `frontend/api/endpoint-name.py` — Vercel auto-routes to `/api/endpoint-name`

---

## Known Gotchas

**Models not in repo.** `models/` is gitignored. `/api/predict` hard-fails on fresh clone. Must run `scripts/train_models.py` first.

**`training_history.json` is a stub.** Both dataset folders contain `{"note": "placeholder"}`. Training curves won't render.

**`generate_dashboard_data.py` hard-exits if CSVs are missing.** Don't run it without `creditcard.csv` and `PS_20174392719_*.csv` at the project root.

**`filter` state not reset on dataset switch.** `TransactionsPage` keeps `fraud`/`normal` filter when dataset changes. Fix: add `setFilter("all")` to the `[dataset]` `useEffect` in `Transactions.tsx:20-25`.

**`/api/predict` has no input validation.** Unknown `dataset` values silently fall back to PaySim. Acceptable for diploma demo.
