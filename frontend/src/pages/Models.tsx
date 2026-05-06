
import { useEffect, useState } from "react";
import { useDataset } from "@/hooks/use-dataset";
import { getModelResults, getConfusionMatrices, getHyperparameters, getShapValues } from "@/lib/data";
import type { ModelResult, ConfusionMatrixData, Hyperparameter, ShapFeature } from "@/lib/types";
import { MODEL_COLORS } from "@/lib/constants";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ConfusionMatrix } from "@/components/models/confusion-matrix";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function ModelsPage() {
  const { dataset } = useDataset();
  const [results, setResults] = useState<ModelResult[]>([]);
  const [cm, setCm] = useState<ConfusionMatrixData | null>(null);
  const [params, setParams] = useState<Hyperparameter[]>([]);
  const [shapData, setShapData] = useState<ShapFeature[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    setError("");
    setResults([]);
    setCm(null);
    setShapData([]);
    getModelResults(dataset).then(setResults).catch(() => setError("Failed to load model data"));
    getConfusionMatrices(dataset).then(setCm).catch(() => {});
    getHyperparameters(dataset).then(setParams).catch(() => {});
    getShapValues(dataset).then(setShapData).catch(() => {});
  }, [dataset]);

  if (error) return <p className="text-sm text-red-500 bg-red-500/10 rounded-md px-3 py-2">{error}</p>;
  if (!results.length && !cm) return (
    <div className="space-y-6">
      <div className="h-64 bg-card rounded-lg animate-pulse" />
      <div className="h-80 bg-card rounded-lg animate-pulse" />
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Metrics table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Model Performance Ranking</CardTitle>
          <CardDescription className="text-xs">
            Models are ranked by F1-Score — the primary metric for imbalanced fraud detection.
            <span className="text-foreground"> Precision</span> = what fraction of fraud alerts were genuine.
            <span className="text-foreground"> Recall</span> = what fraction of all real frauds were caught.
            <span className="text-foreground"> ROC-AUC / PR-AUC</span> measure overall discriminative power across every possible threshold.
            <span className="text-foreground"> TP</span> = caught fraud, <span className="text-foreground">FP</span> = false alarm, <span className="text-foreground">FN</span> = missed fraud.
          </CardDescription>
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
          <div className="mb-4 space-y-1">
            <h3 className="text-sm font-medium">Confusion Matrices</h3>
            <p className="text-xs text-muted-foreground">
              Each matrix shows how a model classified the test set.
              <span className="text-foreground"> TN</span> = normal correctly cleared,
              <span className="text-foreground"> TP</span> = fraud correctly caught,
              <span className="text-foreground"> FP</span> = false alarm (normal blocked),
              <span className="text-foreground"> FN</span> = missed fraud. In financial fraud detection, minimising FN (undetected fraud) is the highest priority.
            </p>
          </div>
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
            <CardDescription className="text-xs">
              Key configuration values used to train each model. These were tuned via cross-validation to maximise F1-Score on the training set.
            </CardDescription>
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

      {/* Global SHAP Feature Importance */}
      {shapData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">XGBoost Feature Importance (SHAP)</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground mb-3">
              Mean absolute SHAP value across test samples — measures each feature's average contribution to the XGBoost model's fraud predictions.
            </p>
            <ResponsiveContainer width="100%" height={380}>
              <BarChart data={shapData.slice(0, 15)} layout="vertical" margin={{ left: 90, right: 20 }}>
                <XAxis type="number" tick={{ fill: "#a1a1aa", fontSize: 10 }} tickFormatter={(v: number) => v.toFixed(3)} />
                <YAxis type="category" dataKey="name" tick={{ fill: "#a1a1aa", fontSize: 10 }} width={90} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
                  labelStyle={{ color: "#fafafa" }}
                  formatter={(v) => [typeof v === "number" ? v.toFixed(4) : "", "mean |SHAP|"]}
                />
                <Bar dataKey="mean_abs_shap" fill={MODEL_COLORS["XGBoost"] || "#3b82f6"} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
