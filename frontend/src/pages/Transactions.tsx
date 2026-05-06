
import { useEffect, useState, useMemo } from "react";
import { useDataset } from "@/hooks/use-dataset";
import { getTransactions, getSummary } from "@/lib/data";
import type { Transaction, DatasetSummary } from "@/lib/types";
import { Card, CardContent } from "@/components/ui/card";
import { Activity, AlertTriangle, ShieldCheck, Search, TrendingUp, Info } from "lucide-react";

type Filter = "all" | "fraud" | "normal";
const PAGE_SIZE = 50;

export default function TransactionsPage() {
  const { dataset } = useDataset();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    setError("");
    getTransactions(dataset).then(setTransactions).catch(() => setError("Failed to load transactions"));
    getSummary(dataset).then(setSummary).catch(() => {});
    setPage(0);
    setSearch("");
    setFilter("all");
  }, [dataset]);

  const filtered = useMemo(() => {
    let list = transactions;
    if (filter === "fraud") list = list.filter((t) => t.is_fraud === 1);
    if (filter === "normal") list = list.filter((t) => t.is_fraud === 0);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter(
        (t) =>
          t.tx_id.toLowerCase().includes(q) ||
          t.amount.toString().includes(q)
      );
    }
    return list;
  }, [transactions, filter, search]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageData = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const fraudCount = transactions.filter((t) => t.is_fraud === 1).length;
  const normalCount = transactions.filter((t) => t.is_fraud === 0).length;
  const highRiskCount = transactions.filter((t) => t.risk_level === "high").length;
  const detectionRate = fraudCount > 0
    ? ((transactions.filter((t) => t.is_fraud === 1 && t.ensemble_prediction === 1).length / fraudCount) * 100).toFixed(1)
    : "0";

  if (error) return <p className="text-sm text-red-500 bg-red-500/10 rounded-md px-3 py-2">{error}</p>;
  if (!transactions.length && !summary) return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[1, 2, 3, 4].map((i) => <div key={i} className="h-20 bg-card rounded-lg animate-pulse" />)}
      </div>
      <div className="h-96 bg-card rounded-lg animate-pulse" />
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Intro */}
      <Card className="border-blue-500/20 bg-blue-500/5">
        <CardContent className="pt-4 pb-4">
          <div className="flex gap-3">
            <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
            <div className="text-xs text-muted-foreground space-y-1">
              <p className="text-sm font-medium text-foreground">Transaction-Level Fraud Scoring</p>
              <p>
                This table shows a representative sample of transactions from the selected dataset, each scored by all three models (XGBoost, Isolation Forest, Autoencoder) and the final ensemble.
                The ensemble score is a weighted combination — transactions above the trained threshold are flagged as <span className="text-red-400 font-medium">FRAUD</span>.
              </p>
              <p>
                <span className="text-foreground font-medium">Ensemble</span> = weighted average of all model scores.
                {" "}<span className="text-foreground font-medium">XGBoost</span> = supervised score (pattern match to training fraud cases).
                {" "}<span className="text-foreground font-medium">IF</span> = Isolation Forest anomaly score (statistical outlier strength).
                {" "}<span className="text-foreground font-medium">AE</span> = Autoencoder reconstruction error (how unusual the feature combination is).
                Risk level is derived from the ensemble score: High &gt; 0.7, Medium 0.3–0.7, Low &lt; 0.3.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MiniStat
          icon={Activity}
          label="Total Transactions"
          value={summary ? summary.total_samples.toLocaleString() : transactions.length.toLocaleString()}
          sub={`${transactions.length} in sample`}
          color="text-blue-500"
        />
        <MiniStat
          icon={AlertTriangle}
          label="Fraud in Sample"
          value={fraudCount.toString()}
          sub={`${((fraudCount / transactions.length) * 100).toFixed(1)}% of sample`}
          color="text-red-500"
        />
        <MiniStat
          icon={ShieldCheck}
          label="Detection Rate"
          value={`${detectionRate}%`}
          sub="of fraud correctly flagged"
          color="text-emerald-500"
        />
        <MiniStat
          icon={TrendingUp}
          label="High Risk"
          value={highRiskCount.toString()}
          sub="transactions flagged"
          color="text-amber-500"
        />
      </div>

      {/* Filters + Search */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex gap-2">
          {([
            ["all", `All (${transactions.length})`],
            ["fraud", `Fraud (${fraudCount})`],
            ["normal", `Normal (${normalCount})`],
          ] as [Filter, string][]).map(([key, label]) => (
            <button
              key={key}
              onClick={() => { setFilter(key); setPage(0); }}
              className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                filter === key
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search TX ID or amount..."
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(0); }}
              className="pl-8 pr-3 py-1.5 text-xs rounded-md bg-muted border border-border focus:border-primary outline-none transition-colors w-52"
            />
          </div>
          <span className="text-xs text-muted-foreground">
            {filtered.length} results
          </span>
        </div>
      </div>

      {/* Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/30">
                  <th className="text-left py-2.5 px-3 text-muted-foreground font-medium text-xs">TX ID</th>
                  <th className="text-left py-2.5 px-3 text-muted-foreground font-medium text-xs">Timestamp</th>
                  {dataset === "paysim" && (
                    <th className="text-left py-2.5 px-3 text-muted-foreground font-medium text-xs">Type</th>
                  )}
                  <th className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">Amount</th>
                  <th className="text-center py-2.5 px-3 text-muted-foreground font-medium text-xs">Status</th>
                  <th className="text-center py-2.5 px-3 text-muted-foreground font-medium text-xs">Risk</th>
                  <th className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">Ensemble</th>
                  <th className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">XGBoost</th>
                  <th className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">IF</th>
                  <th className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">AE</th>
                  {dataset === "paysim" && (
                    <th className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">Bal. Error</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {pageData.map((tx) => (
                  <tr key={tx.tx_id} className="border-b border-border/30 hover:bg-accent/20 transition-colors">
                    <td className="py-2 px-3 font-mono text-xs text-muted-foreground">{tx.tx_id}</td>
                    <td className="py-2 px-3 text-xs text-muted-foreground whitespace-nowrap">{tx.timestamp}</td>
                    {dataset === "paysim" && (
                      <td className="py-2 px-3">
                        <TypeBadge type={tx.type || "UNKNOWN"} />
                      </td>
                    )}
                    <td className="py-2 px-3 text-right font-mono text-xs">
                      ${tx.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </td>
                    <td className="py-2 px-3 text-center">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                          tx.is_fraud
                            ? "bg-red-500/10 text-red-500"
                            : "bg-emerald-500/10 text-emerald-500"
                        }`}
                      >
                        {tx.is_fraud ? "FRAUD" : "NORMAL"}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-center">
                      <RiskBadge level={tx.risk_level} />
                    </td>
                    <td className="py-2 px-3 text-right">
                      <ScoreBadge score={tx.scores["Ensemble"]} />
                    </td>
                    <td className="py-2 px-3 text-right">
                      <ScoreBadge score={tx.scores["XGBoost"]} />
                    </td>
                    <td className="py-2 px-3 text-right">
                      <ScoreBadge score={tx.scores["Isolation Forest"]} />
                    </td>
                    <td className="py-2 px-3 text-right">
                      <ScoreBadge score={tx.scores["Autoencoder"]} />
                    </td>
                    {dataset === "paysim" && (
                      <td className="py-2 px-3 text-right font-mono text-xs">
                        {tx.error_orig !== undefined && tx.error_orig !== 0 ? (
                          <span className="text-red-400">{tx.error_orig.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                        ) : (
                          <span className="text-muted-foreground">0.00</span>
                        )}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-3 py-1.5 text-xs rounded-md bg-muted text-muted-foreground disabled:opacity-30 hover:text-foreground transition-colors"
          >
            Previous
          </button>
          <span className="text-xs text-muted-foreground">
            Page {page + 1} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="px-3 py-1.5 text-xs rounded-md bg-muted text-muted-foreground disabled:opacity-30 hover:text-foreground transition-colors"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

function MiniStat({ icon: Icon, label, value, sub, color }: {
  icon: React.ElementType; label: string; value: string; sub: string; color: string;
}) {
  return (
    <Card>
      <CardContent className="pt-4 pb-3 px-4">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-lg bg-accent flex items-center justify-center ${color}`}>
            <Icon className="w-4 h-4" />
          </div>
          <div>
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="text-lg font-bold font-mono">{value}</p>
            <p className="text-xs text-muted-foreground">{sub}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function RiskBadge({ level }: { level: string }) {
  const styles = {
    high: "bg-red-500/10 text-red-500",
    medium: "bg-amber-500/10 text-amber-500",
    low: "bg-emerald-500/10 text-emerald-500",
  }[level] || "bg-muted text-muted-foreground";

  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium uppercase ${styles}`}>
      {level}
    </span>
  );
}

function TypeBadge({ type }: { type: string }) {
  const styles: Record<string, string> = {
    TRANSFER: "bg-blue-500/10 text-blue-400",
    CASH_OUT: "bg-purple-500/10 text-purple-400",
    PAYMENT: "bg-emerald-500/10 text-emerald-400",
    CASH_IN: "bg-cyan-500/10 text-cyan-400",
    DEBIT: "bg-amber-500/10 text-amber-400",
  };
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-mono ${styles[type] || "bg-muted text-muted-foreground"}`}>
      {type}
    </span>
  );
}

function ScoreBadge({ score }: { score: number }) {
  let color = "text-emerald-500 bg-emerald-500/10";
  if (score > 0.7) color = "text-red-500 bg-red-500/10";
  else if (score > 0.3) color = "text-amber-500 bg-amber-500/10";

  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-mono ${color}`}>
      {score.toFixed(3)}
    </span>
  );
}
