const BASE = "/api";

export interface FlightSummary {
  id: number;
  score: number;
  anomalous: boolean;
}

export interface PathPoint {
  lat: number;
  lon: number;
  alt: number;
  t: number;
}

export interface FlightDetail {
  id: number;
  path: PathPoint[];
  reconstructed: PathPoint[];
  scores: number[];
  window_score: number;
  threshold: number;
  step_threshold: number;
}

export interface SimulationRequest {
  id: number;
  kind: string;
  magnitude: number;
  onset: number;
}

export interface SimulationResult {
  id: number;
  kind: string;
  path: PathPoint[];
  scores: number[];
  window_score: number;
  threshold: number;
  step_threshold: number;
  onset_index: number;
  latency_seconds: number | null;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) throw new Error(`request failed: ${response.status}`);
  return response.json();
}

export interface SceneFlight {
  id: number;
  callsign: string;
  path: PathPoint[];
  scores: number[];
  anomalous: boolean;
  start_offset: number;
}

export interface InjectedFlight extends SceneFlight {
  injected: true;
  kind: string;
}

export interface Scene {
  flights: SceneFlight[];
  step_threshold: number;
  step_seconds: number;
  center: { lat: number; lon: number };
}

export function getScene(count = 12): Promise<Scene> {
  return getJson(`/scene?count=${count}`);
}

export interface MetricRow {
  model: string;
  real_roc_auc: number;
  real_pr_auc: number;
  synthetic_mean_roc_auc: number;
  synthetic_per_type: Record<string, number>;
}

export interface Metrics {
  selected_model: string | null;
  results: MetricRow[];
}

export function getFlights(
  limit = 30,
  order: "anomalous" | "normal" | "typical" = "anomalous",
): Promise<FlightSummary[]> {
  return getJson(`/flights?limit=${limit}&order=${order}`);
}

export function getMetrics(): Promise<Metrics> {
  return getJson("/metrics");
}

export function getFlight(id: number): Promise<FlightDetail> {
  return getJson(`/flights/${id}`);
}

export async function simulate(request: SimulationRequest): Promise<SimulationResult> {
  const response = await fetch(`${BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(`request failed: ${response.status}`);
  return response.json();
}
