export const MODEL_COLORS: Record<string, string> = {
  "XGBoost": "#22c55e",
  "Ensemble": "#3b82f6",
  "Isolation Forest": "#f59e0b",
  "Autoencoder": "#a855f7",
  "Elliptic Envelope": "#ec4899",
  "One-Class SVM": "#06b6d4",
  "K-Means": "#ef4444",
  "Local Outlier Factor": "#64748b",
  "Tuned Isolation Forest": "#f97316",
};

export const ROUTES = [
  { path: "/", label: "Dashboard", icon: "LayoutDashboard" },
  { path: "/models", label: "Models", icon: "Brain" },
  { path: "/transactions", label: "Transactions", icon: "Table" },
  { path: "/analytics", label: "Analytics", icon: "BarChart3" },
  { path: "/predict", label: "Predict", icon: "Zap" },
] as const;

export const DATASET_LABELS: Record<string, string> = {
  creditcard: "Credit Card",
  paysim: "PaySim",
};
