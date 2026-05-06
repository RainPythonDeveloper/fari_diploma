
import { useEffect, useState, useMemo } from "react";
import { useDataset } from "@/hooks/use-dataset";
import { getRocCurves, getPrCurves, getModelResults, getCombinedComparison } from "@/lib/data";
import type { RocCurveData, PrCurveData, ModelResult } from "@/lib/types";
import { RocChart } from "@/components/charts/roc-chart";
import { PrChart } from "@/components/charts/pr-chart";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { MODEL_COLORS } from "@/lib/constants";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  ScatterChart, Scatter, ZAxis,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
} from "recharts";

export default function AnalyticsPage() {
  const { dataset } = useDataset();
  const [roc, setRoc] = useState<RocCurveData | null>(null);
  const [pr, setPr] = useState<PrCurveData | null>(null);
  const [results, setResults] = useState<ModelResult[]>([]);
  const [combined, setCombined] = useState<(ModelResult & { dataset: string })[]>([]);
  const [error, setError] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [threshModel, setThreshModel] = useState("");

  useEffect(() => {
    setError("");
    setRoc(null);
    setPr(null);
    getRocCurves(dataset).then(setRoc).catch(() => setError("Failed to load analytics data"));
    getPrCurves(dataset).then(setPr).catch(() => {});
    getModelResults(dataset).then((r) => { setResults(r); if (!threshModel && r.length) setThreshModel(r[0].model); }).catch(() => {});
    getCombinedComparison().then(setCombined).catch(() => {});
  }, [dataset]); // eslint-disable-line react-hooks/exhaustive-deps

  const threshMetrics = useMemo(() => {
    if (!roc || !pr || !threshModel) return null;
    const rocModel = roc[threshModel];
    const prModel = pr[threshModel];
    if (!rocModel?.thresholds?.length || !prModel?.thresholds?.length) return null;
    const maxIdx = rocModel.thresholds.length - 1;
    const rocIdx = Math.round((1 - threshold) * maxIdx);
    const prIdx = Math.round((1 - threshold) * (prModel.thresholds.length - 1));
    const tpr = rocModel.tpr[rocIdx] ?? 0;
    const fpr = rocModel.fpr[rocIdx] ?? 0;
    const precision = prModel.precision[prIdx] ?? 0;
    const recall = prModel.recall[prIdx] ?? 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { tpr, fpr, precision, recall, f1 };
  }, [roc, pr, threshModel, threshold]);

  if (error) return <p className="text-sm text-red-500 bg-red-500/10 rounded-md px-3 py-2">{error}</p>;
  if (!roc && !pr && !results.length) return (
    <div className="space-y-6">
      {[1, 2, 3].map((i) => <div key={i} className="h-80 bg-card rounded-lg animate-pulse" />)}
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {roc && <RocChart data={roc} />}
        {pr && <PrChart data={pr} />}
      </div>

      {/* Precision vs Recall scatter */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Precision vs Recall — Operating Point Comparison</CardTitle>
          <CardDescription className="text-xs">
            Each dot is a model evaluated at its optimal threshold. The ideal point is top-right (high precision AND high recall).
            Models in the top-right catch more fraud with fewer false alarms. There is always a trade-off — moving right (higher recall) typically lowers precision.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ bottom: 20, right: 20 }}>
              <XAxis
                dataKey="recall"
                type="number"
                domain={[0, 1]}
                name="Recall"
                tick={{ fill: "#a1a1aa", fontSize: 11 }}
                label={{ value: "Recall", position: "bottom", fill: "#71717a", fontSize: 12 }}
              />
              <YAxis
                dataKey="precision"
                type="number"
                domain={[0, 1]}
                name="Precision"
                tick={{ fill: "#a1a1aa", fontSize: 11 }}
                label={{ value: "Precision", angle: -90, position: "insideLeft", fill: "#71717a", fontSize: 12 }}
              />
              <ZAxis range={[100, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
                labelStyle={{ color: "#fafafa" }} itemStyle={{ color: "#e4e4e7" }}
                formatter={(value) => typeof value === "number" ? value.toFixed(4) : value}
              />
              <Scatter
                data={results.map((r) => ({
                  recall: r.recall,
                  precision: r.precision,
                  name: r.model,
                  fill: MODEL_COLORS[r.model] || "#64748b",
                }))}
              >
                {results.map((r, i) => (
                  <Cell key={i} fill={MODEL_COLORS[r.model] || "#64748b"} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-3 mt-2 justify-center">
            {results.map((r) => (
              <div key={r.model} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: MODEL_COLORS[r.model] || "#64748b" }} />
                {r.model}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Radar chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Model Metrics Radar — Multi-Dimensional Comparison</CardTitle>
          <CardDescription className="text-xs">
            Compares all models across five metrics simultaneously. A larger filled area means better overall performance.
            No single metric tells the full story — this view reveals which model excels in which dimension.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={[
              { metric: "Precision", ...Object.fromEntries(results.map((r) => [r.model, r.precision])) },
              { metric: "Recall", ...Object.fromEntries(results.map((r) => [r.model, r.recall])) },
              { metric: "F1", ...Object.fromEntries(results.map((r) => [r.model, r.f1])) },
              { metric: "ROC-AUC", ...Object.fromEntries(results.map((r) => [r.model, r.roc_auc])) },
              { metric: "PR-AUC", ...Object.fromEntries(results.map((r) => [r.model, r.pr_auc])) },
            ]}>
              <PolarGrid stroke="#27272a" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 1]} tick={{ fill: "#52525b", fontSize: 10 }} />
              {results.slice(0, 5).map((r) => (
                <Radar
                  key={r.model}
                  name={r.model}
                  dataKey={r.model}
                  stroke={MODEL_COLORS[r.model] || "#64748b"}
                  fill={MODEL_COLORS[r.model] || "#64748b"}
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
              ))}
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </RadarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Cross-dataset F1 comparison */}
      {combined.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Cross-Dataset F1 Comparison</CardTitle>
          <CardDescription className="text-xs">
            The same models evaluated on both datasets. A drop in F1 between datasets reveals how well each model generalises beyond its training domain.
            Stable performance across Credit Card and PaySim indicates a robust, dataset-agnostic approach.
          </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={(() => {
                  const models = [...new Set(combined.map((c) => c.model))];
                  return models.map((m) => {
                    const cc = combined.find((c) => c.model === m && c.dataset === "Credit Card");
                    const ps = combined.find((c) => c.model === m && c.dataset === "PaySim");
                    return { model: m, "Credit Card": cc?.f1 || 0, PaySim: ps?.f1 || 0 };
                  });
                })()}
                margin={{ bottom: 60 }}
              >
                <XAxis dataKey="model" tick={{ fill: "#a1a1aa", fontSize: 10 }} angle={-35} textAnchor="end" />
                <YAxis domain={[0, 1]} tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
                  labelStyle={{ color: "#fafafa" }} itemStyle={{ color: "#e4e4e7" }}
                  formatter={(value) => typeof value === "number" ? value.toFixed(4) : value}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="Credit Card" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                <Bar dataKey="PaySim" fill="#f59e0b" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Threshold Sensitivity Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Threshold Sensitivity Analysis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs text-muted-foreground">
            Adjust the decision threshold to explore the Precision / Recall / F1 trade-off. Higher threshold = fewer fraud alerts but more missed frauds.
          </p>
          <div className="flex flex-wrap items-center gap-4">
            <label className="text-xs text-muted-foreground">Model:</label>
            <select
              value={threshModel}
              onChange={(e) => setThreshModel(e.target.value)}
              className="text-xs bg-muted border border-border rounded-md px-2 py-1 outline-none"
            >
              {results.map((r) => <option key={r.model} value={r.model}>{r.model}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-3">
            <label className="text-xs text-muted-foreground w-20 shrink-0">Threshold:</label>
            <input
              type="range" min={0} max={1} step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="flex-1 accent-primary"
            />
            <span className="text-xs font-mono w-10 text-right">{threshold.toFixed(2)}</span>
          </div>
          {threshMetrics ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
              {([
                ["Precision", threshMetrics.precision, "text-blue-400"],
                ["Recall (TPR)", threshMetrics.tpr, "text-emerald-400"],
                ["F1 Score", threshMetrics.f1, "text-amber-400"],
                ["FPR", threshMetrics.fpr, "text-red-400"],
              ] as [string, number, string][]).map(([label, value, color]) => (
                <div key={label} className="bg-muted rounded-lg p-3 text-center">
                  <p className="text-xs text-muted-foreground mb-1">{label}</p>
                  <p className={`text-xl font-bold font-mono ${color}`}>{(value * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground italic">
              Re-run <code className="font-mono">generate_dashboard_data.py</code> to enable live threshold analysis.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
