"use client";

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { MODEL_COLORS } from "@/lib/constants";
import type { RocCurveData } from "@/lib/types";

export function RocChart({ data }: { data: RocCurveData }) {
  // Merge all models into one array of points
  const models = Object.keys(data);
  const maxLen = Math.max(...models.map((m) => data[m].fpr.length));
  const merged = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, number> = { index: i };
    models.forEach((m) => {
      const d = data[m];
      const idx = Math.min(i, d.fpr.length - 1);
      point[`${m}_fpr`] = d.fpr[idx];
      point[`${m}_tpr`] = d.tpr[idx];
    });
    return point;
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">ROC Curves — Receiver Operating Characteristic</CardTitle>
        <CardDescription className="text-xs">
          Each curve plots True Positive Rate (fraud caught) vs. False Positive Rate (normal flagged as fraud) across all decision thresholds.
          AUC closer to 1.0 = better discrimination. The dashed diagonal line represents a random classifier (AUC = 0.5).
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart margin={{ bottom: 20 }}>
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              label={{ value: "False Positive Rate", position: "bottom", fill: "#71717a", fontSize: 12 }}
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fill: "#71717a", fontSize: 12 }}
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8 }}
              labelStyle={{ color: "#fafafa" }}
            />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
            {models.map((model) => (
              <Line
                key={model}
                data={data[model].fpr.map((fpr, i) => ({ fpr, tpr: data[model].tpr[i] }))}
                dataKey="tpr"
                name={`${model} (${data[model].auc})`}
                stroke={MODEL_COLORS[model] || "#64748b"}
                dot={false}
                strokeWidth={2}
              />
            ))}
            {/* Diagonal reference line */}
            <Line
              data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
              dataKey="tpr"
              name="Random"
              stroke="#3f3f46"
              strokeDasharray="5 5"
              dot={false}
              strokeWidth={1}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
