"use client";

import { useState, useEffect } from "react";
import { useDataset } from "@/hooks/use-dataset";
import { getSampleTransactions } from "@/lib/data";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MODEL_COLORS } from "@/lib/constants";
import { Zap, AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface PredictResult {
  fraud: boolean;
  ensemble_score: number;
  scores: Record<string, number>;
  threshold: number;
  latency_ms: number;
}

export default function PredictPage() {
  const { dataset } = useDataset();
  const [samples, setSamples] = useState<{ fraud: Record<string, number>[]; normal: Record<string, number>[] } | null>(null);
  const [features, setFeatures] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getSampleTransactions(dataset).then((data) => {
      setSamples(data);
      setResult(null);
      setFeatures({});
    });
  }, [dataset]);

  function prefill(type: "fraud" | "normal") {
    if (!samples) return;
    const list = samples[type];
    if (list.length === 0) return;
    const sample = list[Math.floor(Math.random() * list.length)];
    const stringified: Record<string, string> = {};
    for (const [k, v] of Object.entries(sample)) {
      stringified[k] = String(v);
    }
    setFeatures(stringified);
    setResult(null);
  }

  async function predict() {
    setLoading(true);
    setError("");
    setResult(null);

    const featureValues: Record<string, number> = {};
    for (const [k, v] of Object.entries(features)) {
      featureValues[k] = parseFloat(v) || 0;
    }

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, features: featureValues }),
      });
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

  const featureKeys = samples ? Object.keys(samples.fraud[0] || samples.normal[0] || {}) : [];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input form */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Transaction Features</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <button
                onClick={() => prefill("fraud")}
                className="px-3 py-1.5 text-xs rounded-md bg-red-500/10 text-red-500 hover:bg-red-500/20 transition-colors"
              >
                Load Fraud Sample
              </button>
              <button
                onClick={() => prefill("normal")}
                className="px-3 py-1.5 text-xs rounded-md bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20 transition-colors"
              >
                Load Normal Sample
              </button>
            </div>

            <div className="max-h-[400px] overflow-y-auto space-y-2 pr-2">
              {featureKeys.map((key) => (
                <div key={key} className="flex items-center gap-2">
                  <label className="text-xs text-muted-foreground w-40 shrink-0 font-mono truncate">
                    {key}
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={features[key] || ""}
                    onChange={(e) => setFeatures((prev) => ({ ...prev, [key]: e.target.value }))}
                    className="flex-1 h-8 px-2 text-xs font-mono rounded-md bg-muted border border-border focus:border-primary outline-none transition-colors"
                  />
                </div>
              ))}
            </div>

            <button
              onClick={predict}
              disabled={loading || featureKeys.length === 0}
              className="w-full py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50 flex items-center justify-center gap-2 transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              {loading ? "Predicting..." : "Predict"}
            </button>

            {error && (
              <p className="text-xs text-red-500 bg-red-500/10 px-3 py-2 rounded-md">{error}</p>
            )}
          </CardContent>
        </Card>

        {/* Result */}
        <div className="space-y-4">
          {result ? (
            <>
              {/* Decision */}
              <Card className={result.fraud ? "border-red-500/50" : "border-emerald-500/50"}>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-4">
                    {result.fraud ? (
                      <div className="w-14 h-14 rounded-full bg-red-500/10 flex items-center justify-center">
                        <AlertTriangle className="w-7 h-7 text-red-500" />
                      </div>
                    ) : (
                      <div className="w-14 h-14 rounded-full bg-emerald-500/10 flex items-center justify-center">
                        <CheckCircle className="w-7 h-7 text-emerald-500" />
                      </div>
                    )}
                    <div>
                      <p className={`text-2xl font-bold ${result.fraud ? "text-red-500" : "text-emerald-500"}`}>
                        {result.fraud ? "FRAUD DETECTED" : "NORMAL"}
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Ensemble score: <span className="font-mono">{(result.ensemble_score * 100).toFixed(1)}%</span>
                        {" · "}Threshold: <span className="font-mono">{(result.threshold * 100).toFixed(1)}%</span>
                        {" · "}Latency: <span className="font-mono">{result.latency_ms}ms</span>
                      </p>
                    </div>
                  </div>

                  {/* Score bar */}
                  <div className="mt-4">
                    <div className="h-3 bg-muted rounded-full overflow-hidden relative">
                      <div
                        className={`h-full rounded-full transition-all ${result.fraud ? "bg-red-500" : "bg-emerald-500"}`}
                        style={{ width: `${result.ensemble_score * 100}%` }}
                      />
                      <div
                        className="absolute top-0 h-full w-0.5 bg-foreground/50"
                        style={{ left: `${result.threshold * 100}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>0% (Normal)</span>
                      <span>100% (Fraud)</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Per-model scores */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Per-Model Scores</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart
                      data={Object.entries(result.scores).map(([model, score]) => ({
                        model,
                        score,
                        color: MODEL_COLORS[model] || "#64748b",
                      }))}
                      layout="vertical"
                      margin={{ left: 100 }}
                    >
                      <XAxis type="number" domain={[0, 1]} tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                      <YAxis type="category" dataKey="model" tick={{ fill: "#a1a1aa", fontSize: 11 }} width={100} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
                        formatter={(value) => [(typeof value === "number" ? (value * 100).toFixed(1) + "%" : value), "Score"]}
                      />
                      <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                        {Object.entries(result.scores).map(([model], i) => (
                          <Cell key={i} fill={MODEL_COLORS[model] || "#64748b"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="pt-6 flex flex-col items-center justify-center h-64 text-muted-foreground">
                <Zap className="w-8 h-8 mb-3 opacity-30" />
                <p className="text-sm">Load a sample transaction and click Predict</p>
                <p className="text-xs mt-1">Or fill in features manually</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
