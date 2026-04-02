"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MODEL_COLORS } from "@/lib/constants";
import type { ModelResult } from "@/lib/types";

export function F1BarChart({ results }: { results: ModelResult[] }) {
  const data = results.map((r) => ({
    model: r.model,
    f1: r.f1,
    color: MODEL_COLORS[r.model] || "#64748b",
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">F1-Score by Model</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical" margin={{ left: 120 }}>
            <XAxis type="number" domain={[0, 1]} tick={{ fill: "#a1a1aa", fontSize: 12 }} />
            <YAxis type="category" dataKey="model" tick={{ fill: "#a1a1aa", fontSize: 12 }} width={120} />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
              labelStyle={{ color: "#fafafa" }}
              formatter={(value) => [typeof value === "number" ? value.toFixed(4) : value, "F1"]}
            />
            <Bar dataKey="f1" radius={[0, 4, 4, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
