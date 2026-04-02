"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import type { DatasetKey } from "@/lib/types";

interface DatasetContextValue {
  dataset: DatasetKey;
  setDataset: (d: DatasetKey) => void;
}

const DatasetContext = createContext<DatasetContextValue>({
  dataset: "creditcard",
  setDataset: () => {},
});

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [dataset, setDataset] = useState<DatasetKey>("creditcard");
  return (
    <DatasetContext.Provider value={{ dataset, setDataset }}>
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset() {
  return useContext(DatasetContext);
}
