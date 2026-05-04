# Coding Conventions

**Analysis Date:** 2026-05-04

## Naming Patterns

**Files:**
- React page components: PascalCase `.tsx` — `Dashboard.tsx`, `Transactions.tsx`, `Predict.tsx`
- React UI/feature components: kebab-case `.tsx` — `stat-card.tsx`, `confusion-matrix.tsx`, `dataset-switcher.tsx`
- Hooks: `use-` prefix, kebab-case `.tsx` — `use-dataset.tsx`
- Lib modules: kebab-case `.ts` — `utils.ts`, `types.ts`, `data.ts`, `constants.ts`
- Chart components: kebab-case `.tsx` — `roc-chart.tsx`, `f1-bar-chart.tsx`, `pr-chart.tsx`
- API (Python): kebab-case `.py` — `predict.py`

**Components:**
- Named exports for shared/reusable components: `export function StatCard(...)`, `export function RocChart(...)`
- Default exports for page-level components: `export default function DashboardPage()`
- Page function names use `Page` suffix: `DashboardPage`, `ModelsPage`, `TransactionsPage`
- Sub-components defined in the same file as their parent page (not extracted to separate files unless reused across pages): `LoadingSkeleton`, `ScenarioCard`, `RiskBadge`, `TypeBadge`, `ScoreBadge`, `MiniStat`

**Functions:**
- camelCase for all non-component functions: `fetchJson`, `getSummary`, `getModelResults`, `predict`, `selectScenario`
- Private helper functions prefixed with underscore in Python: `_get_models_dir`, `_load_model`

**Variables:**
- camelCase for all local variables and state: `featureValues`, `selectedScenario`, `pageData`
- SCREAMING_SNAKE_CASE for module-level constants: `NAV_ITEMS`, `PAGE_SIZE`, `PAGE_TITLES`, `FEATURE_DESCRIPTIONS`
- PascalCase for exported constant objects: `MODEL_COLORS`, `ROUTES`, `DATASET_LABELS`

**Types/Interfaces:**
- PascalCase for all interfaces and type aliases: `DatasetKey`, `DatasetSummary`, `ModelResult`, `Transaction`
- Types co-located in the same file when component-local: `PredictResult`, `Sample` in `Predict.tsx`, `Filter` in `Transactions.tsx`
- Shared types centralized in `src/lib/types.ts`
- Props interfaces named as `{ComponentName}Props` or inline inline object types: `StatCardProps`, `{ model: string; data: {...} }`

## Code Style

**Formatting:**
- No Prettier or ESLint config file detected — formatting is enforced only by TypeScript compiler
- Consistent 2-space indentation throughout all `.ts` / `.tsx` files
- Double quotes for string literals in TypeScript/React files
- Single quotes not used — double quotes only
- Semicolons present at end of statements
- Arrow functions used for callbacks and inline handlers; `function` declarations used for named exports and page components

**TypeScript:**
- `strict: true` in `tsconfig.json`
- `noUnusedLocals: false`, `noUnusedParameters: false` — unused variables are not enforced
- `noFallthroughCasesInSwitch: true`
- `import type` used consistently for type-only imports: `import type { DatasetKey } from "@/lib/types"`
- ES2020 target with `ESNext` modules

**Linting:**
- No ESLint or Biome config detected — no linting rules enforced beyond TypeScript strict mode

## Import Organization

**Order (observed pattern):**
1. React and third-party library imports (react, react-router-dom, recharts, lucide-react)
2. Internal `@/` alias imports — hooks, then lib, then components
3. CSS imports (only in `main.tsx`)

**Example:**
```typescript
import { useEffect, useState } from "react";
import { BarChart, Bar } from "recharts";
import { useDataset } from "@/hooks/use-dataset";
import { getSummary, getModelResults } from "@/lib/data";
import type { DatasetSummary, ModelResult } from "@/lib/types";
import { StatCard } from "@/components/dashboard/stat-card";
import { Card, CardContent } from "@/components/ui/card";
```

**Path Aliases:**
- `@/` maps to `src/` (configured in both `tsconfig.json` and `vite.config.ts`)
- Relative imports used only for same-directory files: `import { MobileSidebar } from "./sidebar"`
- All cross-directory imports use `@/` alias

**Barrel Files:**
- No `index.ts` barrel files used anywhere — all imports reference exact file paths

## Error Handling

**Frontend async pattern (pages):**
- Data fetching via `.then(setter)` chaining — no `try/catch` on read-only data fetches (failures are silent)
- User-triggered async actions (e.g., `predict()` in `Predict.tsx`) use `try/catch` with error state:

```typescript
const [error, setError] = useState("");

async function predict() {
  setLoading(true);
  setError("");
  try {
    const res = await fetch("/api/predict", { ... });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    const data: PredictResult = await res.json();
    setResult(data);
  } catch (e) {
    setError(e instanceof Error ? e.message : "Prediction failed");
  } finally {
    setLoading(false);
  }
}
```

- Errors displayed inline with Tailwind-styled `<p>` elements: `className="text-xs text-red-500 bg-red-500/10 px-3 py-2 rounded-md"`

**Python API (predict.py):**
- `try/except Exception as e` in HTTP handler — returns 500 with `{"error": str(e)}` JSON body
- Model loading uses lazy caching with module-level `_models` dict

**Loading states:**
- Null-check guard at render: `if (!summary) return <LoadingSkeleton />;`
- Empty array initialization for list states: `useState<ModelResult[]>([])`
- Boolean `loading` state for user-triggered actions

## Logging

**Framework:** None — no logging library in use

**Patterns:**
- No `console.log`, `console.error`, or `console.warn` calls anywhere in source files
- Python API does not use logging module — errors surfaced only via HTTP 500 response

## Comments

**When to Comment:**
- Section labels inside JSX using `{/* Comment */}` syntax — used consistently in large page components to divide regions: `{/* Summary Stats */}`, `{/* Filters + Search */}`, `{/* Left: Scenario selection + features */}`
- Python docstrings on module and function level: triple-quoted `"""..."""`
- No JSDoc/TSDoc annotations on TypeScript functions or interfaces

**Comment style observed in Python:**
```python
"""
Vercel Python Serverless Function for real-time fraud prediction.
POST /api/predict
"""

def _get_models_dir():
    """Find models directory."""
```

## Function Design

**Size:**
- Page components are large (111–364 lines) and contain all sub-components in the same file
- Data fetching functions in `src/lib/data.ts` are uniformly small (1–3 lines each), all delegating to `fetchJson<T>`
- Helper/utility functions are single-purpose and small

**Parameters:**
- Component props use destructured inline objects or named interfaces
- Props interfaces defined above the component function in the same file
- Optional props marked with `?`: `subtitle?: string`, `trend?: "up" | "down" | "neutral"`

**Return Values:**
- All data fetching functions return typed Promises: `Promise<DatasetSummary>`, `Promise<ModelResult[]>`
- Components return JSX or `null` implicitly through conditional rendering
- No explicit `return null` — uses early return with loading skeleton component instead

## Module Design

**Exports:**
- Named exports for all shared components and utility functions
- Default exports only for page-level route components
- `src/lib/utils.ts` exports single utility: `export function cn(...inputs: ClassValue[])`
- `src/lib/constants.ts` exports three constant objects with `export const`
- UI component files export multiple named values: `export { Card, CardHeader, CardContent, ... }`

**Component variants:**
- Variant styling done via `cva` (class-variance-authority) in UI primitives: `buttonVariants`, `badgeVariants`, `tabsListVariants`
- `cn()` utility (clsx + tailwind-merge) used universally for conditional className composition

**"use client" directive:**
- Present on some component files (`button.tsx`, `stat-card.tsx`, `confusion-matrix.tsx`, `roc-chart.tsx`) — a Next.js pattern carried over despite the project using Vite; has no functional effect in this Vite/React SPA context

---

*Convention analysis: 2026-05-04*
