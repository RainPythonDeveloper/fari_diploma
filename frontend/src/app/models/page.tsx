"use client";

import { useEffect, useState } from "react";
import { useDataset } from "@/hooks/use-dataset";
import { getModelResults, getConfusionMatrices, getHyperparameters } from "@/lib/data";
import type { ModelResult, ConfusionMatrixData, Hyperparameter } from "@/lib/types";
import { MODEL_COLORS } from "@/lib/constants";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ConfusionMatrix } from "@/components/models/confusion-matrix";

export default function ModelsPage() {
  const { dataset } = useDataset();
  const [results, setResults] = useState<ModelResult[]>([]);
  const [cm, setCm] = useState<ConfusionMatrixData | null>(null);
  const [params, setParams] = useState<Hyperparameter[]>([]);

  useEffect(() => {
    getModelResults(dataset).then(setResults);
    getConfusionMatrices(dataset).then(setCm);
    getHyperparameters(dataset).then(setParams);
  }, [dataset]);

  return (
    <div className="space-y-6">
      {/* Metrics table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Model Performance Ranking</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-2 text-muted-foreground font-medium">#</th>
                  <th className="text-left py-3 px-2 text-muted-foreground font-medium">Model</th>
                  <th className="text-right py-3 px-2 text-muted-foreground font-medium">Precision</th>
                  <th className="text-right py-3 px-2 text-muted-foreground font-medium">Recall</th>
                  <th className="text-right py-3 px-2 text-muted-foreground font-medium">F1-Score</th>
                  <th className="text-right py-3 px-2 text-muted-foreground font-medium">ROC-AUC</th>
                  <th className="text-right py-3 px-2 text-muted-foreground font-medium">PR-AUC</th>
                  <th className="text-right py-3 px-2 text-muted-foreground font-medium">TP</th>
                  <th className="text-right py-3 px-2 text-muted-foreground font-medium">FP</th>
                  <th className="text-right py-3 px-2 text-muted-foreground font-medium">FN</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.model} className="border-b border-border/50 hover:bg-accent/30 transition-colors">
                    <td className="py-3 px-2 font-mono text-muted-foreground">{r.rank}</td>
                    <td className="py-3 px-2">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: MODEL_COLORS[r.model] || "#64748b" }}
                        />
                        <span className="font-medium">{r.model}</span>
                      </div>
                    </td>
                    <td className="py-3 px-2 text-right font-mono">{r.precision.toFixed(4)}</td>
                    <td className="py-3 px-2 text-right font-mono">{r.recall.toFixed(4)}</td>
                    <td className="py-3 px-2 text-right font-mono font-bold">{r.f1.toFixed(4)}</td>
                    <td className="py-3 px-2 text-right font-mono">{r.roc_auc.toFixed(4)}</td>
                    <td className="py-3 px-2 text-right font-mono">{r.pr_auc.toFixed(4)}</td>
                    <td className="py-3 px-2 text-right font-mono">{r.tp}</td>
                    <td className="py-3 px-2 text-right font-mono">{r.fp}</td>
                    <td className="py-3 px-2 text-right font-mono">{r.fn}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Confusion Matrices */}
      {cm && (
        <div>
          <h3 className="text-sm font-medium mb-4">Confusion Matrices</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {Object.entries(cm).map(([model, data]) => (
              <ConfusionMatrix key={model} model={model} data={data} />
            ))}
          </div>
        </div>
      )}

      {/* Hyperparameters */}
      {params.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Hyperparameters</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-2 text-muted-foreground font-medium">Model</th>
                  <th className="text-left py-2 px-2 text-muted-foreground font-medium">Parameters</th>
                </tr>
              </thead>
              <tbody>
                {params.map((p) => (
                  <tr key={p.model} className="border-b border-border/50">
                    <td className="py-2 px-2 font-medium">{p.model}</td>
                    <td className="py-2 px-2 font-mono text-xs text-muted-foreground">{p.params}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
