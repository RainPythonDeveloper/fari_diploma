
import { useDataset } from "@/hooks/use-dataset";
import { DATASET_LABELS } from "@/lib/constants";
import type { DatasetKey } from "@/lib/types";

export function DatasetSwitcher() {
  const { dataset, setDataset } = useDataset();

  return (
    <div className="flex gap-1 p-1 bg-muted rounded-lg">
      {(Object.entries(DATASET_LABELS) as [DatasetKey, string][]).map(
        ([key, label]) => (
          <button
            key={key}
            onClick={() => setDataset(key)}
            className={`flex-1 text-xs py-1.5 px-2 rounded-md transition-colors ${
              dataset === key
                ? "bg-background text-foreground shadow-sm font-medium"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {label}
          </button>
        )
      )}
    </div>
  );
}
