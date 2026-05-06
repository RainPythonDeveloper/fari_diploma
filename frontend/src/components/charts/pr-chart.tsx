"use client";

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { MODEL_COLORS } from "@/lib/constants";
import type { PrCurveData } from "@/lib/types";

export function PrChart({ data }: { data: PrCurveData }) {
  const models = Object.keys(data);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">Precision-Recall Curves</CardTitle>
        <CardDescription className="text-xs">
          More informative than ROC for heavily imbalanced datasets (few frauds among many normal transactions).
          AP (Average Precision) summarises the area under the curve — higher means fewer missed frauds and fewer false alarms across all thresholds.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart margin={{ bottom: 20 }}>
            <XAxis
              dataKey="recall"
              type="number"
              domain={[0, 1]}
              label={{ value: "Recall", position: "bottom", fill: "#71717a", fontSize: 12 }}
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              label={{ value: "Precision", angle: -90, position: "insideLeft", fill: "#71717a", fontSize: 12 }}
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
              labelStyle={{ color: "#fafafa" }} itemStyle={{ color: "#e4e4e7" }}
            />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
            {models.map((model) => (
              <Line
                key={model}
                data={data[model].recall.map((r, i) => ({
                  recall: r,
                  precision: data[model].precision[i],
                }))}
                dataKey="precision"
                name={`${model} (AP=${data[model].ap})`}
                stroke={MODEL_COLORS[model] || "#64748b"}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
