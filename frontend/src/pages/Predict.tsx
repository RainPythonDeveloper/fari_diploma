
import { useState, useEffect } from "react";
import { useDataset } from "@/hooks/use-dataset";
import { getSampleTransactions, getPredictSamples, type PredictResult } from "@/lib/data";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MODEL_COLORS, DATASET_LABELS } from "@/lib/constants";
import { Zap, AlertTriangle, CheckCircle, Loader2, ChevronDown, ChevronUp, Info } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";

interface Sample {
  label: string;
  description: string;
  features: Record<string, number>;
}

export default function PredictPage() {
  const { dataset } = useDataset();
  const [samples, setSamples] = useState<{ fraud: Sample[]; normal: Sample[] } | null>(null);
  const [precomputed, setPrecomputed] = useState<Record<string, PredictResult>>({});
  const [features, setFeatures] = useState<Record<string, string>>({});
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    setResult(null);
    setFeatures({});
    setSelectedScenario(null);
    getSampleTransactions(dataset).then(setSamples);
    getPredictSamples(dataset).then(setPrecomputed).catch(() => setPrecomputed({}));
  }, [dataset]);

  function selectScenario(sample: Sample) {
    const stringified: Record<string, string> = {};
    for (const [k, v] of Object.entries(sample.features)) {
      stringified[k] = String(v);
    }
    setFeatures(stringified);
    setSelectedScenario(sample.label);
    setResult(null);
    setError("");
  }

  async function predict() {
    if (!selectedScenario) return;
    const stored = precomputed[selectedScenario];
    if (!stored) {
      setError("No pre-computed result for this scenario. Re-run scripts/generate_predict_samples.py.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    // Brief delay for UX feedback (mimics network round-trip)
    await new Promise((r) => setTimeout(r, 350));
    setResult(stored);
    setLoading(false);
  }

  const featureKeys = Object.keys(features);
  const hasFeatures = featureKeys.length > 0;

  return (
    <div className="space-y-6">
      {/* Intro */}
      <Card className="border-blue-500/20 bg-blue-500/5">
        <CardContent className="pt-5 pb-4">
          <div className="flex gap-3">
            <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
            <div className="text-sm text-muted-foreground space-y-1.5">
              <div className="flex items-center gap-2 flex-wrap">
                <p className="text-foreground font-medium">Fraud Prediction — Ensemble Inference</p>
                <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 font-medium">
                  Demo Mode
                </span>
              </div>
              <p>
                Select a transaction scenario below and the ensemble (XGBoost + Isolation Forest + Autoencoder)
                returns a fraud verdict, per-model score breakdown, and SHAP feature contributions explaining the decision.
              </p>
              <p className="text-xs">
                Predictions are <span className="text-foreground">pre-computed locally</span> using the trained models —
                model artifacts (~500 MB) are excluded from the Vercel deployment to stay within serverless limits.
                Results are identical to live inference; the workflow demonstrates exactly how the system would behave in production.
                {" "}Active dataset: <span className="font-medium text-foreground">{DATASET_LABELS[dataset]}</span>.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Scenario selection + features */}
        <div className="space-y-4">
          {/* Scenario Cards */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">Select a Transaction Scenario</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {samples && (
                <>
                  <p className="text-xs text-muted-foreground mb-2">Suspicious / Fraudulent:</p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {samples.fraud.map((s) => (
                      <ScenarioCard
                        key={s.label}
                        sample={s}
                        selected={selectedScenario === s.label}
                        variant="fraud"
                        onClick={() => selectScenario(s)}
                      />
                    ))}
                  </div>

                  <p className="text-xs text-muted-foreground mt-4 mb-2">Normal / Legitimate:</p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {samples.normal.map((s) => (
                      <ScenarioCard
                        key={s.label}
                        sample={s}
                        selected={selectedScenario === s.label}
                        variant="normal"
                        onClick={() => selectScenario(s)}
                      />
                    ))}
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          {/* Advanced: Raw feature editor */}
          <Card>
            <CardHeader
              className="cursor-pointer"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">
                  Advanced: Raw Features
                  {hasFeatures && <span className="ml-2 text-xs text-muted-foreground font-normal">({featureKeys.length} features loaded)</span>}
                </CardTitle>
                {showAdvanced ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
              </div>
            </CardHeader>
            {showAdvanced && (
              <CardContent className="space-y-3 pt-0">
                {dataset === "creditcard" && (
                  <p className="text-xs text-muted-foreground bg-muted p-2 rounded-md">
                    Credit Card features V1-V28 are PCA-transformed (anonymized).
                    Amount_log is log-scaled transaction amount. Hour_sin/Hour_cos encode transaction time of day.
                  </p>
                )}
                {dataset === "paysim" && (
                  <p className="text-xs text-muted-foreground bg-muted p-2 rounded-md">
                    PaySim features: step = time period, amount_log = log(amount), type_code = transaction type (0=CASH_IN, 1=CASH_OUT, 2=DEBIT, 3=PAYMENT, 4=TRANSFER),
                    balance_diff/error = balance tracking fields.
                  </p>
                )}
                <div className="max-h-[300px] overflow-y-auto space-y-2 pr-2">
                  {featureKeys.map((key) => (
                    <div key={key} className="flex items-center gap-2">
                      <label className="text-xs text-muted-foreground w-40 shrink-0 font-mono truncate" title={FEATURE_DESCRIPTIONS[key] || key}>
                        {key}
                      </label>
                      <input
                        type="number"
                        step="any"
                        value={features[key] || ""}
                        onChange={(e) => setFeatures((prev) => ({ ...prev, [key]: e.target.value }))}
                        className="flex-1 h-7 px-2 text-xs font-mono rounded-md bg-muted border border-border focus:border-primary outline-none transition-colors"
                      />
                    </div>
                  ))}
                </div>
              </CardContent>
            )}
          </Card>

          {/* Predict button */}
          <button
            onClick={predict}
            disabled={loading || !hasFeatures}
            className="w-full py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50 flex items-center justify-center gap-2 transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
            {loading ? "Analyzing Transaction..." : "Analyze Transaction"}
          </button>

          {error && (
            <p className="text-xs text-red-500 bg-red-500/10 px-3 py-2 rounded-md">{error}</p>
          )}
        </div>

        {/* Right: Result */}
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
                        {result.fraud ? "FRAUD DETECTED" : "LEGITIMATE"}
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Confidence: <span className="font-mono font-medium text-foreground">{(result.ensemble_score * 100).toFixed(1)}%</span>
                        {" · "}Latency: <span className="font-mono">{result.latency_ms}ms</span>
                      </p>
                    </div>
                  </div>

                  {/* Score bar */}
                  <div className="mt-5">
                    <div className="flex items-center justify-between text-xs text-muted-foreground mb-1.5">
                      <span>Fraud Score</span>
                      <span>Threshold: {(result.threshold * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-4 bg-muted rounded-full overflow-hidden relative">
                      <div
                        className={`h-full rounded-full transition-all ${result.fraud ? "bg-red-500" : "bg-emerald-500"}`}
                        style={{ width: `${Math.min(result.ensemble_score * 100, 100)}%` }}
                      />
                      <div
                        className="absolute top-0 h-full w-0.5 bg-foreground/70"
                        style={{ left: `${result.threshold * 100}%` }}
                        title={`Threshold: ${(result.threshold * 100).toFixed(0)}%`}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>0% Normal</span>
                      <span>100% Fraud</span>
                    </div>
                  </div>

                  <p className="text-xs text-muted-foreground mt-3 bg-muted p-2 rounded-md">
                    {result.fraud
                      ? "This transaction scored above the fraud threshold. In production, it would be blocked or sent for manual review."
                      : "This transaction scored below the fraud threshold. It would be approved automatically."}
                  </p>
                </CardContent>
              </Card>

              {/* Per-model scores */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Per-Model Breakdown</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground mb-3">
                    Each model independently scores the transaction. The ensemble combines all scores with weighted averaging.
                  </p>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart
                      data={Object.entries(result.scores).map(([model, score]) => ({
                        model,
                        score,
                      }))}
                      layout="vertical"
                      margin={{ left: 110 }}
                    >
                      <XAxis type="number" domain={[0, 1]} tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                      <YAxis type="category" dataKey="model" tick={{ fill: "#a1a1aa", fontSize: 11 }} width={110} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
                        labelStyle={{ color: "#fafafa" }} itemStyle={{ color: "#e4e4e7" }}
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

              {/* SHAP Feature Contributions */}
              {result.shap_values && (() => {
                const entries = Object.entries(result.shap_values!)
                  .sort((a, b) => Math.abs(b[1] as number) - Math.abs(a[1] as number))
                  .slice(0, 10);
                const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v as number)), 0.01);
                return (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm font-medium">Feature Contributions (XGBoost SHAP)</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-xs text-muted-foreground mb-3">
                        Red bars push toward fraud, green bars toward normal. Longer = stronger influence on this prediction.
                      </p>
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={entries.map(([name, value]) => ({ name, value }))} layout="vertical" margin={{ left: 90, right: 20 }}>
                          <XAxis type="number" domain={[-maxAbs - 0.02, maxAbs + 0.02]} tick={{ fill: "#a1a1aa", fontSize: 10 }} tickFormatter={(v: number) => v.toFixed(2)} />
                          <YAxis type="category" dataKey="name" tick={{ fill: "#a1a1aa", fontSize: 10 }} width={90} />
                          <Tooltip
                            contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
                            labelStyle={{ color: "#fafafa" }} itemStyle={{ color: "#e4e4e7" }}
                            formatter={(v) => [typeof v === "number" ? v.toFixed(4) : "", "SHAP value"]}
                          />
                          <ReferenceLine x={0} stroke="#52525b" strokeWidth={1} />
                          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {entries.map(([, value], i) => (
                              <Cell key={i} fill={(value as number) > 0 ? "#ef4444" : "#10b981"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                );
              })()}
            </>
          ) : (
            <Card className="border-dashed">
              <CardContent className="pt-6 flex flex-col items-center justify-center h-80 text-muted-foreground">
                <Zap className="w-10 h-10 mb-4 opacity-20" />
                <p className="text-sm font-medium">No prediction yet</p>
                <p className="text-xs mt-2 text-center max-w-xs">
                  Select a transaction scenario on the left, then click &quot;Analyze Transaction&quot; to see the fraud detection result.
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function ScenarioCard({ sample, selected, variant, onClick }: {
  sample: Sample; selected: boolean; variant: "fraud" | "normal"; onClick: () => void;
}) {
  const borderColor = selected
    ? variant === "fraud" ? "border-red-500" : "border-emerald-500"
    : "border-border";
  const hoverBg = variant === "fraud" ? "hover:bg-red-500/5" : "hover:bg-emerald-500/5";

  return (
    <button
      onClick={onClick}
      className={`text-left p-3 rounded-lg border ${borderColor} ${hoverBg} transition-all ${selected ? "ring-1 ring-offset-1 ring-offset-background" : ""} ${selected && variant === "fraud" ? "ring-red-500/50" : ""} ${selected && variant === "normal" ? "ring-emerald-500/50" : ""}`}
    >
      <div className="flex items-start gap-2">
        {variant === "fraud" ? (
          <AlertTriangle className="w-3.5 h-3.5 text-red-500 mt-0.5 shrink-0" />
        ) : (
          <CheckCircle className="w-3.5 h-3.5 text-emerald-500 mt-0.5 shrink-0" />
        )}
        <div>
          <p className="text-xs font-medium">{sample.label}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{sample.description}</p>
        </div>
      </div>
    </button>
  );
}

const FEATURE_DESCRIPTIONS: Record<string, string> = {
  Amount_log: "Log-scaled transaction amount",
  Hour_sin: "Time of day (sine component)",
  Hour_cos: "Time of day (cosine component)",
  amount_log: "Log-scaled transaction amount",
  step: "Time period (hours)",
  type_code: "Transaction type (0=CASH_IN, 1=CASH_OUT, 2=DEBIT, 3=PAYMENT, 4=TRANSFER)",
  balance_diff_orig: "Sender balance change",
  balance_diff_dest: "Receiver balance change",
  error_orig: "Sender balance error",
  error_dest: "Receiver balance error",
  oldbalanceOrg: "Sender balance before",
  newbalanceOrig: "Sender balance after",
  oldbalanceDest: "Receiver balance before",
  newbalanceDest: "Receiver balance after",
};
