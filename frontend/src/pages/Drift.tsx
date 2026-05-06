
import { useEffect, useState, useMemo } from "react";
import { useDataset } from "@/hooks/use-dataset";
import { getFeatureAnalysis } from "@/lib/data";
import type { FeatureAnalysis } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";

type SortKey = "abs_cohen_d" | "name";

export default function DriftPage() {
  const { dataset } = useDataset();
  const [features, setFeatures] = useState<FeatureAnalysis[]>([]);
  const [error, setError] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("abs_cohen_d");

  useEffect(() => {
    setFeatures([]);
    setError("");
    getFeatureAnalysis(dataset)
      .then(setFeatures)
      .catch(() => setError("Failed to load feature analysis. Run generate_dashboard_data.py first."));
  }, [dataset]);

  const sorted = useMemo(() => {
    if (sortBy === "name") return [...features].sort((a, b) => a.name.localeCompare(b.name));
    return [...features].sort((a, b) => b.abs_cohen_d - a.abs_cohen_d);
  }, [features, sortBy]);

  const top15 = useMemo(() => [...features].sort((a, b) => b.abs_cohen_d - a.abs_cohen_d).slice(0, 15), [features]);
  const top3 = useMemo(() => [...features].sort((a, b) => b.abs_cohen_d - a.abs_cohen_d).slice(0, 3), [features]);

  if (error) return <p className="text-sm text-red-500 bg-red-500/10 rounded-md px-3 py-2">{error}</p>;
  if (!features.length) return (
    <div className="space-y-6">
      {[1, 2, 3].map((i) => <div key={i} className="h-64 bg-card rounded-lg animate-pulse" />)}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Info banner */}
      <Card className="border-blue-500/20 bg-blue-500/5">
        <CardContent className="pt-4 pb-4">
          <p className="text-sm font-medium text-foreground mb-1">Feature Discrimination Analysis — How Fraud Transactions Differ from Normal</p>
          <p className="text-xs text-muted-foreground">
            To detect fraud, the models rely on features that behave differently in fraudulent vs. normal transactions.
            This page quantifies that difference using <span className="text-foreground font-medium">Cohen&apos;s d</span> — a standardised effect size measuring how many standard deviations separate
            the fraud and normal distributions for each feature.
            <span className="text-foreground"> |d| &gt; 0.8</span> = strong discriminator (most useful for detection),
            <span className="text-foreground"> 0.5–0.8</span> = medium effect,
            <span className="text-foreground"> &lt; 0.5</span> = weak (little discriminative power).
            Features with high |d| are the primary signals the XGBoost and ensemble models exploit.
          </p>
        </CardContent>
      </Card>

      {/* Top 3 summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {top3.map((f, i) => (
          <Card key={f.name} className={i === 0 ? "border-emerald-500/30" : ""}>
            <CardContent className="pt-4 pb-4">
              <p className="text-xs text-muted-foreground">#{i + 1} Top Discriminator</p>
              <p className="text-lg font-bold font-mono mt-1 truncate">{f.name}</p>
              <p className="text-xs text-muted-foreground mt-1">
                |d| = <span className="text-foreground font-mono">{f.abs_cohen_d.toFixed(3)}</span>
                {" · "}
                <span className={effectColor(f.abs_cohen_d)}>{effectLabel(f.abs_cohen_d)}</span>
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Bar chart top 15 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Top 15 Features by Effect Size (|Cohen&apos;s d|)</CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            Green bars = strong effect (|d| ≥ 0.8) — these features are the most powerful fraud signals and carry the highest weight in XGBoost.
            The dashed reference lines mark the strong (0.8) and medium (0.5) effect thresholds.
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={top15} layout="vertical" margin={{ left: 90, right: 40, top: 10 }}>
              <XAxis type="number" tick={{ fill: "#a1a1aa", fontSize: 10 }} tickFormatter={(v: number) => v.toFixed(1)} />
              <YAxis type="category" dataKey="name" tick={{ fill: "#a1a1aa", fontSize: 10 }} width={90} />
              <Tooltip
                contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
                labelStyle={{ color: "#fafafa" }} itemStyle={{ color: "#e4e4e7" }}
                formatter={(v) => [typeof v === "number" ? v.toFixed(4) : "", "|Cohen's d|"]}
              />
              <ReferenceLine x={0.8} stroke="#f59e0b" strokeDasharray="4 3" label={{ value: "strong (0.8)", fill: "#f59e0b", fontSize: 9, position: "insideTopRight" }} />
              <ReferenceLine x={0.5} stroke="#64748b" strokeDasharray="4 3" label={{ value: "medium (0.5)", fill: "#64748b", fontSize: 9, position: "insideTopRight" }} />
              <Bar dataKey="abs_cohen_d" radius={[0, 4, 4, 0]}>
                {top15.map((f, i) => (
                  <Cell key={i} fill={f.abs_cohen_d >= 0.8 ? "#10b981" : f.abs_cohen_d >= 0.5 ? "#f59e0b" : "#64748b"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Full table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium">All Features ({features.length})</CardTitle>
            <div className="flex gap-2">
              {(["abs_cohen_d", "name"] as SortKey[]).map((key) => (
                <button
                  key={key}
                  onClick={() => setSortBy(key)}
                  className={`text-xs px-3 py-1 rounded-md transition-colors ${sortBy === key ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
                >
                  {key === "abs_cohen_d" ? "Sort by |d|" : "Sort by name"}
                </button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-2 text-muted-foreground font-medium">Feature</th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">Cohen&apos;s d</th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">|d|</th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">Fraud μ</th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">Normal μ</th>
                  <th className="text-left py-2 px-2 text-muted-foreground font-medium">Effect</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((f) => (
                  <tr key={f.name} className="border-b border-border/50 hover:bg-accent/30 transition-colors">
                    <td className="py-2 px-2 font-mono text-xs">{f.name}</td>
                    <td className="py-2 px-2 text-right font-mono text-xs">{f.cohen_d.toFixed(4)}</td>
                    <td className="py-2 px-2 text-right font-mono text-xs font-bold">{f.abs_cohen_d.toFixed(4)}</td>
                    <td className="py-2 px-2 text-right font-mono text-xs text-red-400">{f.fraud_mean.toFixed(4)}</td>
                    <td className="py-2 px-2 text-right font-mono text-xs text-emerald-400">{f.normal_mean.toFixed(4)}</td>
                    <td className="py-2 px-2">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${effectBadge(f.abs_cohen_d)}`}>
                        {effectLabel(f.abs_cohen_d)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function effectLabel(d: number): string {
  if (d >= 0.8) return "strong";
  if (d >= 0.5) return "medium";
  return "weak";
}

function effectColor(d: number): string {
  if (d >= 0.8) return "text-emerald-400";
  if (d >= 0.5) return "text-amber-400";
  return "text-zinc-400";
}

function effectBadge(d: number): string {
  if (d >= 0.8) return "bg-emerald-500/10 text-emerald-400";
  if (d >= 0.5) return "bg-amber-500/10 text-amber-400";
  return "bg-zinc-500/10 text-zinc-400";
}
