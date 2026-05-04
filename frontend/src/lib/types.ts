export type DatasetKey = "creditcard" | "paysim";

export interface DatasetSummary {
  name: string;
  total_samples: number;
  normal: number;
  fraud: number;
  fraud_rate: number;
  features_count: number;
  test_samples: number;
  test_fraud: number;
  contamination: number;
}

export interface ModelResult {
  rank: number;
  model: string;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  dataset?: string;
}

export interface ConfusionMatrixData {
  [model: string]: { tn: number; fp: number; fn: number; tp: number };
}

export interface RocCurveData {
  [model: string]: { fpr: number[]; tpr: number[]; auc: number; thresholds?: number[] };
}

export interface PrCurveData {
  [model: string]: { precision: number[]; recall: number[]; ap: number; thresholds?: number[] };
}

export interface Transaction {
  index: number;
  tx_id: string;
  timestamp: string;
  is_fraud: number;
  amount: number;
  risk_level: "high" | "medium" | "low";
  scores: { [model: string]: number };
  ensemble_prediction: number;
  type?: string;
  step?: number;
  type_code?: number;
  balance_diff_orig?: number;
  balance_diff_dest?: number;
  error_orig?: number;
  error_dest?: number;
}

export interface SampleTransaction {
  label: string;
  description: string;
  features: Record<string, number>;
}

export interface DistributionData {
  amount: {
    normal: { counts: number[]; edges: number[] };
    fraud: { counts: number[]; edges: number[] };
  };
  time?: {
    normal: { counts: number[]; edges: number[] };
    fraud: { counts: number[]; edges: number[] };
  };
  fraud_by_type?: { [type: string]: number };
}

export interface Hyperparameter {
  model: string;
  params: string;
}

export interface PredictResponse {
  fraud: boolean;
  ensemble_score: number;
  scores: { [model: string]: number };
  threshold: number;
  latency_ms: number;
  shap_values?: Record<string, number>;
}

export interface ShapFeature {
  name: string;
  mean_abs_shap: number;
  mean_shap: number;
}

export interface FeatureAnalysis {
  name: string;
  cohen_d: number;
  abs_cohen_d: number;
  fraud_mean: number;
  normal_mean: number;
  fraud_std: number;
  normal_std: number;
}
