const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface OrderInput {
  order_value: number;
  category: string;
  payment_method: string;
  return_reason: string;
  delivery_days: number;
  order_hour: number;
  is_weekend: number;
  account_age_days_at_order: number;
  hist_orders_before: number;
  hist_return_rate_before: number;
  hist_abusive_return_rate_before: number;
  hist_chargebacks_before: number;
  price_vs_category_avg: number;
  days_to_return: number;
}

export interface TopFactor {
  feature: string;
  value: number;
  impact: number;
}

export interface ScoreResult {
  decision_id: string;
  timestamp_utc: string;
  input: OrderInput;
  risk_score: number;
  threshold_used: number;
  action: "approve" | "manual_review" | "auto_decline";
  top_factors: TopFactor[];
  plain_english_summary: string;
  model_version: string;
}

export interface ClusterMember {
  customer_id: string;
  account_age_days: number;
  return_rate: number;
  abusive_return_rate: number;
  total_orders: number;
}

export interface ClusterInput {
  size: number;
  shares_device: number;
  shares_payment: number;
  shares_address: number;
  shares_device_and_payment: number;
  avg_account_age: number;
  account_age_std: number;
  avg_return_rate: number;
  max_return_rate: number;
  avg_abusive_return_rate: number;
  max_abusive_return_rate: number;
  avg_total_orders: number;
  cluster_id?: number;
  is_true_ring?: number;
  members?: ClusterMember[];
}

export interface RingFactor {
  feature: string;
  value: number;
  coefficient: number;
  impact: number;
}

export interface RingResult {
  ring_score: number;
  threshold_used: number;
  verdict: "likely_ring" | "likely_benign";
  top_factors: RingFactor[];
  plain_english_summary: string;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${path} failed (${res.status}): ${detail}`);
  }
  return res.json();
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${path} failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export const api = {
  scoreOrder: (order: OrderInput) => postJSON<ScoreResult>("/api/score", order),
  checkCluster: (cluster: ClusterInput) => postJSON<RingResult>("/api/abuse-check", cluster),
  sampleClusters: () => getJSON<{ clusters: ClusterInput[] }>("/api/sample-clusters"),
  metrics: () => getJSON<Record<string, unknown>>("/api/metrics"),
  health: () => getJSON<{ status: string }>("/api/health"),
};
