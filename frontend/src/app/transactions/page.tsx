"use client";

import { useEffect, useState, useMemo } from "react";
import { useDataset } from "@/hooks/use-dataset";
import { getTransactions } from "@/lib/data";
import type { Transaction } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type Filter = "all" | "fraud" | "normal";
const PAGE_SIZE = 50;

export default function TransactionsPage() {
  const { dataset } = useDataset();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [page, setPage] = useState(0);

  useEffect(() => {
    getTransactions(dataset).then(setTransactions);
    setPage(0);
  }, [dataset]);

  const filtered = useMemo(() => {
    if (filter === "fraud") return transactions.filter((t) => t.is_fraud === 1);
    if (filter === "normal") return transactions.filter((t) => t.is_fraud === 0);
    return transactions;
  }, [transactions, filter]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageData = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const modelNames = pageData.length > 0 ? Object.keys(pageData[0].scores) : [];

  const fraudCount = transactions.filter((t) => t.is_fraud === 1).length;
  const normalCount = transactions.filter((t) => t.is_fraud === 0).length;

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex items-center justify-between">
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
        <p className="text-xs text-muted-foreground">
          Page {page + 1} of {totalPages}
        </p>
      </div>

      {/* Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/30">
                  <th className="text-left py-2.5 px-3 text-muted-foreground font-medium text-xs">Status</th>
                  <th className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">Amount</th>
                  {dataset === "paysim" && (
                    <>
                      <th className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">Bal Diff Orig</th>
                      <th className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">Error Orig</th>
                    </>
                  )}
                  {modelNames.map((m) => (
                    <th key={m} className="text-right py-2.5 px-3 text-muted-foreground font-medium text-xs">
                      {m}
                    </th>
                  ))}
                  <th className="text-center py-2.5 px-3 text-muted-foreground font-medium text-xs">Prediction</th>
                </tr>
              </thead>
              <tbody>
                {pageData.map((tx) => (
                  <tr key={tx.index} className="border-b border-border/30 hover:bg-accent/20 transition-colors">
                    <td className="py-2 px-3">
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
                    <td className="py-2 px-3 text-right font-mono text-xs">
                      ${tx.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </td>
                    {dataset === "paysim" && (
                      <>
                        <td className="py-2 px-3 text-right font-mono text-xs">
                          {tx.balance_diff_orig?.toLocaleString(undefined, { minimumFractionDigits: 2 }) ?? "—"}
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-xs">
                          {tx.error_orig?.toFixed(2) ?? "—"}
                        </td>
                      </>
                    )}
                    {modelNames.map((m) => {
                      const score = tx.scores[m];
                      return (
                        <td key={m} className="py-2 px-3 text-right">
                          <ScoreBadge score={score} />
                        </td>
                      );
                    })}
                    <td className="py-2 px-3 text-center">
                      <span
                        className={`inline-block w-2 h-2 rounded-full ${
                          tx.ensemble_prediction ? "bg-red-500" : "bg-emerald-500"
                        }`}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Pagination */}
      <div className="flex items-center justify-center gap-2">
        <button
          onClick={() => setPage((p) => Math.max(0, p - 1))}
          disabled={page === 0}
          className="px-3 py-1.5 text-xs rounded-md bg-muted text-muted-foreground disabled:opacity-30 hover:text-foreground transition-colors"
        >
          Previous
        </button>
        <span className="text-xs text-muted-foreground">
          {page + 1} / {totalPages}
        </span>
        <button
          onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          disabled={page >= totalPages - 1}
          className="px-3 py-1.5 text-xs rounded-md bg-muted text-muted-foreground disabled:opacity-30 hover:text-foreground transition-colors"
        >
          Next
        </button>
      </div>
    </div>
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
