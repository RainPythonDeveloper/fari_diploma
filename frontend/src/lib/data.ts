import type {
  DatasetKey,
  DatasetSummary,
  ModelResult,
  ConfusionMatrixData,
  RocCurveData,
  PrCurveData,
  Transaction,
  DistributionData,
  Hyperparameter,
  ShapFeature,
  FeatureAnalysis,
} from "./types";

const cache = new Map<string, unknown>();

async function fetchJson<T>(path: string): Promise<T> {
  if (cache.has(path)) return cache.get(path) as T;
  const res = await fetch(path);
  const data = await res.json();
  cache.set(path, data);
  return data;
}

export async function getSummary(dataset: DatasetKey): Promise<DatasetSummary> {
  return fetchJson(`/data/${dataset}/summary.json`);
}

export async function getModelResults(dataset: DatasetKey): Promise<ModelResult[]> {
  return fetchJson(`/data/${dataset}/model_results.json`);
}

export async function getConfusionMatrices(dataset: DatasetKey): Promise<ConfusionMatrixData> {
  return fetchJson(`/data/${dataset}/confusion_matrices.json`);
}

export async function getRocCurves(dataset: DatasetKey): Promise<RocCurveData> {
  return fetchJson(`/data/${dataset}/roc_curves.json`);
}

export async function getPrCurves(dataset: DatasetKey): Promise<PrCurveData> {
  return fetchJson(`/data/${dataset}/pr_curves.json`);
}

const PS_TYPE_MAP: Record<number, string> = {
  0: "CASH_IN", 1: "CASH_OUT", 2: "DEBIT", 3: "PAYMENT", 4: "TRANSFER",
};

export async function getTransactions(dataset: DatasetKey): Promise<Transaction[]> {
  const raw = await fetchJson<(Omit<Transaction, "tx_id" | "timestamp" | "risk_level" | "type"> & { type_code?: number })[]>(
    `/data/${dataset}/transactions.json`
  );
  const prefix = dataset === "creditcard" ? "CC" : "PS";
  return raw.map((t) => {
    const ensemble = t.scores["Ensemble"] ?? 0;
    const risk_level: Transaction["risk_level"] = ensemble > 0.7 ? "high" : ensemble > 0.3 ? "medium" : "low";
    const tx_id = `${prefix}-${String(t.index).padStart(6, "0")}`;
    let timestamp = "";
    if (t.step !== undefined) {
      const day = Math.floor(t.step / 24) + 1;
      const hour = String(Math.floor(t.step % 24)).padStart(2, "0");
      timestamp = `Day ${day}, ${hour}:00`;
    }
    const type = t.type_code !== undefined ? PS_TYPE_MAP[Math.round(t.type_code)] : undefined;
    return { ...t, tx_id, timestamp, risk_level, type };
  });
}

export async function getDistributions(dataset: DatasetKey): Promise<DistributionData> {
  return fetchJson(`/data/${dataset}/distributions.json`);
}

export async function getHyperparameters(dataset: DatasetKey): Promise<Hyperparameter[]> {
  return fetchJson(`/data/${dataset}/hyperparameters.json`);
}

export async function getSampleTransactions(dataset: DatasetKey) {
  return fetchJson<{
    fraud: { label: string; description: string; features: Record<string, number> }[];
    normal: { label: string; description: string; features: Record<string, number> }[];
  }>(`/data/${dataset}/sample_transactions.json`);
}

export async function getCombinedComparison(): Promise<(ModelResult & { dataset: string })[]> {
  return fetchJson(`/data/combined/comparison.json`);
}

export async function getCombinedRanking() {
  return fetchJson<(ModelResult & { dataset: string })[]>(`/data/combined/ranking.json`);
}

export async function getBestModels() {
  return fetchJson<(ModelResult & { dataset: string })[]>(`/data/combined/best_models.json`);
}

export async function getShapValues(dataset: DatasetKey): Promise<ShapFeature[]> {
  return fetchJson(`/data/${dataset}/shap_values.json`);
}

export async function getFeatureAnalysis(dataset: DatasetKey): Promise<FeatureAnalysis[]> {
  return fetchJson(`/data/${dataset}/feature_analysis.json`);
}
