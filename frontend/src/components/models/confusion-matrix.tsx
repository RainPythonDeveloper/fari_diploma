"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
  model: string;
  data: { tn: number; fp: number; fn: number; tp: number };
}

export function ConfusionMatrix({ model, data }: Props) {
  const total = data.tn + data.fp + data.fn + data.tp;
  const maxVal = Math.max(data.tn, data.fp, data.fn, data.tp);

  function cellColor(value: number, isCorrect: boolean) {
    const intensity = value / maxVal;
    if (isCorrect) {
      return `rgba(34, 197, 94, ${0.1 + intensity * 0.5})`;
    }
    return `rgba(239, 68, 68, ${0.1 + intensity * 0.5})`;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">{model}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-1 max-w-[280px]">
          {/* Header row */}
          <div />
          <div className="text-center text-xs text-muted-foreground py-1">Pred Normal</div>
          <div className="text-center text-xs text-muted-foreground py-1">Pred Fraud</div>

          {/* Actual Normal */}
          <div className="text-xs text-muted-foreground flex items-center">Actual Normal</div>
          <div
            className="text-center py-3 rounded font-mono text-sm"
            style={{ backgroundColor: cellColor(data.tn, true) }}
          >
            {data.tn.toLocaleString()}
            <div className="text-xs text-muted-foreground">TN</div>
          </div>
          <div
            className="text-center py-3 rounded font-mono text-sm"
            style={{ backgroundColor: cellColor(data.fp, false) }}
          >
            {data.fp.toLocaleString()}
            <div className="text-xs text-muted-foreground">FP</div>
          </div>

          {/* Actual Fraud */}
          <div className="text-xs text-muted-foreground flex items-center">Actual Fraud</div>
          <div
            className="text-center py-3 rounded font-mono text-sm"
            style={{ backgroundColor: cellColor(data.fn, false) }}
          >
            {data.fn.toLocaleString()}
            <div className="text-xs text-muted-foreground">FN</div>
          </div>
          <div
            className="text-center py-3 rounded font-mono text-sm"
            style={{ backgroundColor: cellColor(data.tp, true) }}
          >
            {data.tp.toLocaleString()}
            <div className="text-xs text-muted-foreground">TP</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
