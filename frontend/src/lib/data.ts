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

export async function getTransactions(dataset: DatasetKey): Promise<Transaction[]> {
  return fetchJson(`/data/${dataset}/transactions.json`);
}

export async function getDistributions(dataset: DatasetKey): Promise<DistributionData> {
  return fetchJson(`/data/${dataset}/distributions.json`);
}

export async function getHyperparameters(dataset: DatasetKey): Promise<Hyperparameter[]> {
  return fetchJson(`/data/${dataset}/hyperparameters.json`);
}

export async function getSampleTransactions(dataset: DatasetKey) {
  return fetchJson<{ fraud: Record<string, number>[]; normal: Record<string, number>[] }>(
    `/data/${dataset}/sample_transactions.json`
  );
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
