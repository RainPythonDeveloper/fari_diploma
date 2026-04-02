"use client";

import { useEffect, useState } from "react";
import { Activity, AlertTriangle, Percent, Trophy } from "lucide-react";
import { useDataset } from "@/hooks/use-dataset";
import { getSummary, getModelResults, getRocCurves } from "@/lib/data";
import type { DatasetSummary, ModelResult, RocCurveData } from "@/lib/types";
import { StatCard } from "@/components/dashboard/stat-card";
import { F1BarChart } from "@/components/charts/f1-bar-chart";
import { RocChart } from "@/components/charts/roc-chart";
import { ClassBalanceChart } from "@/components/charts/class-balance-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function DashboardPage() {
  const { dataset } = useDataset();
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [results, setResults] = useState<ModelResult[]>([]);
  const [roc, setRoc] = useState<RocCurveData | null>(null);

  useEffect(() => {
    getSummary(dataset).then(setSummary);
    getModelResults(dataset).then(setResults);
    getRocCurves(dataset).then(setRoc);
  }, [dataset]);

  if (!summary) return <LoadingSkeleton />;

  const bestModel = results[0];

  return (
    <div className="space-y-6">
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
