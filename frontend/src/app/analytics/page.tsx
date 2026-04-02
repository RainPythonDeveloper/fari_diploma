"use client";

import { useEffect, useState } from "react";
import { useDataset } from "@/hooks/use-dataset";
import { getRocCurves, getPrCurves, getModelResults, getCombinedComparison } from "@/lib/data";
import type { RocCurveData, PrCurveData, ModelResult } from "@/lib/types";
import { RocChart } from "@/components/charts/roc-chart";
import { PrChart } from "@/components/charts/pr-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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

  useEffect(() => {
    getRocCurves(dataset).then(setRoc);
    getPrCurves(dataset).then(setPr);
    getModelResults(dataset).then(setResults);
    getCombinedComparison().then(setCombined);
  }, [dataset]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {roc && <RocChart data={roc} />}
        {pr && <PrChart data={pr} />}
      </div>

      {/* Precision vs Recall scatter */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Precision vs Recall</CardTitle>
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
          <CardTitle className="text-sm font-medium">Model Metrics Radar</CardTitle>
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
    </div>
  );
}
