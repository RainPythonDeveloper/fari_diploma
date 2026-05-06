
import { useEffect, useState } from "react";
import { Activity, AlertTriangle, Percent, Trophy, Shield } from "lucide-react";
import { useDataset } from "@/hooks/use-dataset";
import { getSummary, getModelResults, getRocCurves } from "@/lib/data";
import type { DatasetSummary, ModelResult, RocCurveData } from "@/lib/types";
import { StatCard } from "@/components/dashboard/stat-card";
import { F1BarChart } from "@/components/charts/f1-bar-chart";
import { RocChart } from "@/components/charts/roc-chart";
import { ClassBalanceChart } from "@/components/charts/class-balance-chart";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

export default function DashboardPage() {
  const { dataset } = useDataset();
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [results, setResults] = useState<ModelResult[]>([]);
  const [roc, setRoc] = useState<RocCurveData | null>(null);

  useEffect(() => {
    setSummary(null);
    getSummary(dataset).then(setSummary).catch(() => setSummary(null));
    getModelResults(dataset).then(setResults).catch(() => {});
    getRocCurves(dataset).then(setRoc).catch(() => {});
  }, [dataset]);

  if (!summary) return <LoadingSkeleton />;

  const bestModel = results[0];

  return (
    <div className="space-y-6">
      {/* Project overview */}
      <Card className="border-emerald-500/20 bg-emerald-500/5">
        <CardContent className="pt-5 pb-4">
          <div className="flex gap-3">
            <Shield className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
            <div className="space-y-1.5">
              <p className="text-sm font-semibold text-foreground">Automated Fraud Detection using Machine Learning — Diploma Project</p>
              <p className="text-xs text-muted-foreground">
                The system detects fraudulent financial transactions by combining three complementary ML models into a weighted ensemble:
                {" "}<span className="text-foreground font-medium">XGBoost</span> (supervised gradient boosting — learns explicit fraud patterns from labeled training data),
                {" "}<span className="text-foreground font-medium">Isolation Forest</span> (unsupervised anomaly detection — isolates rare transactions that deviate from the norm), and
                {" "}<span className="text-foreground font-medium">Autoencoder</span> (deep neural network — flags transactions with high reconstruction error, i.e. unusual feature combinations).
                Each model produces a fraud score from 0–1; the ensemble combines them with optimised weights for the final verdict.
              </p>
              <p className="text-xs text-muted-foreground">
                Two real-world datasets are supported — switch via the sidebar:
                {" "}<span className="text-foreground font-medium">Credit Card</span> (284 k European bank transactions, features PCA-anonymised for privacy) and
                {" "}<span className="text-foreground font-medium">PaySim</span> (synthetic mobile-money transfers modelled on M-Pesa behaviour).
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Transactions"
          value={summary.total_samples.toLocaleString()}
          subtitle={`${summary.features_count} features`}
          icon={Activity}
          color="text-blue-500"
        />
        <StatCard
          title="Fraud Detected"
          value={summary.fraud.toLocaleString()}
          subtitle={`in test: ${summary.test_fraud}`}
          icon={AlertTriangle}
          color="text-red-500"
        />
        <StatCard
          title="Fraud Rate"
          value={`${summary.fraud_rate}%`}
          subtitle={`contamination: ${(summary.contamination * 100).toFixed(4)}%`}
          icon={Percent}
          color="text-amber-500"
        />
        <StatCard
          title="Best Model"
          value={bestModel?.model || "—"}
          subtitle={bestModel ? `F1: ${bestModel.f1.toFixed(4)}` : ""}
          icon={Trophy}
          color="text-emerald-500"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <F1BarChart results={results} />
        </div>
        <ClassBalanceChart summary={summary} />
      </div>

      {roc && <RocChart data={roc} />}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Dataset Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Name</p>
              <p className="font-medium">{summary.name}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Total Samples</p>
              <p className="font-mono">{summary.total_samples.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Normal / Fraud</p>
              <p className="font-mono">{summary.normal.toLocaleString()} / {summary.fraud.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Test Samples</p>
              <p className="font-mono">{summary.test_samples.toLocaleString()}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-28 bg-card rounded-lg animate-pulse" />
        ))}
      </div>
      <div className="h-80 bg-card rounded-lg animate-pulse" />
    </div>
  );
}
