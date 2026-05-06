"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import type { DatasetSummary } from "@/lib/types";

export function ClassBalanceChart({ summary }: { summary: DatasetSummary }) {
  const data = [
    { name: "Normal", value: summary.normal, color: "#3b82f6" },
    { name: "Fraud", value: summary.fraud, color: "#ef4444" },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">Class Balance</CardTitle>
        <CardDescription className="text-xs">
          Fraud transactions are a tiny minority — this extreme imbalance is the core challenge.
          Models must learn to detect rare fraud without flooding normal transactions with false alarms.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
            >
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
              labelStyle={{ color: "#fafafa" }} itemStyle={{ color: "#e4e4e7" }}
              formatter={(value) => [typeof value === "number" ? value.toLocaleString() : value, ""]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
          </PieChart>
        </ResponsiveContainer>
        <p className="text-center text-xs text-muted-foreground mt-2">
          Fraud rate: {summary.fraud_rate}%
        </p>
      </CardContent>
    </Card>
  );
}
