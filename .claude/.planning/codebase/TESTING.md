# Testing Patterns

**Analysis Date:** 2026-05-04

## Test Framework

**Runner:** None — no test framework is installed or configured.

- `package.json` has no test dependencies (`jest`, `vitest`, `@testing-library/*`, `playwright`, `cypress` are all absent)
- No `jest.config.*`, `vitest.config.*`, or `playwright.config.*` files exist
- No test script defined in `package.json` `scripts` block

**Run Commands:**
```bash
# No test commands available — testing infrastructure not set up
npm run dev       # Development server (only available script beyond build)
npm run build     # Production build
npm run preview   # Preview production build
```

## Test File Organization

**Location:** No test files exist anywhere in the project.

```bash
find . -name "*.test.*" -o -name "*.spec.*"  # Returns nothing
```

**Named pattern:** Not applicable.

## Test Structure

**No test suites exist.** The codebase has zero test files as of analysis date.

## Mocking

**Framework:** Not applicable — no test framework installed.

## Fixtures and Factories

**Test Data:** Not applicable.

The project does have rich sample data files used at runtime (not for testing):
- `public/data/creditcard/sample_transactions.json` — sample fraud/normal transactions for the Predict page UI
- `public/data/{dataset}/summary.json`, `model_results.json`, etc. — static JSON served as mock API responses

These are production data assets, not test fixtures.

## Coverage

**Requirements:** None enforced.

**Coverage tool:** Not configured.

## Test Types

**Unit Tests:** Not present.

**Integration Tests:** Not present.

**E2E Tests:** Not present.

## What Exists Instead of Tests

**Manual verification approach:**

The codebase relies on a few runtime patterns rather than automated tests:

1. **TypeScript strict mode** (`"strict": true` in `tsconfig.json`) provides compile-time type safety across all components, hooks, and data fetching functions in `src/`

2. **Typed data contracts** — `src/lib/types.ts` defines all shared interfaces (`DatasetSummary`, `ModelResult`, `Transaction`, `PredictResponse`, etc.) which TypeScript enforces at component boundaries

3. **Static JSON data** — All non-predict data is served from pre-built JSON files in `public/data/` which were generated offline from ML notebooks; no runtime API calls that could fail silently

4. **Single async path tested manually** — The only live API call is `POST /api/predict` in `src/pages/Predict.tsx`. Error states are handled with `try/catch` and displayed inline, implying manual browser testing

5. **Python ML pipeline tested in Jupyter notebooks** — `Paysim.ipynb`, `anomaly_detection_diploma.ipynb`, `creditcard_final_fixed_7_models.ipynb` in the project root serve as the validation/analysis layer for model outputs

## Recommendations for Adding Tests

If tests are added, the natural stack given existing tooling would be:

**Unit/Integration:**
- Vitest (already uses Vite) + `@testing-library/react`
- Config: `vitest.config.ts` alongside `vite.config.ts`
- Test files: co-located as `*.test.tsx` next to source files

**Key candidates for first tests:**
- `src/lib/data.ts` — `fetchJson` caching logic and all data fetching wrappers
- `src/lib/utils.ts` — `cn()` utility function
- `src/hooks/use-dataset.tsx` — `DatasetProvider` and `useDataset` hook
- `src/pages/Predict.tsx` — `predict()` async function error handling path (currently the only try/catch in frontend)
- `frontend/api/predict.py` — `predict()` function with mocked numpy/joblib

---

*Testing analysis: 2026-05-04*
