// The only place the frontend talks to the backend. All numbers and categories arrive
// ready-made from the API; nothing here computes AQI or forecasts.

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, params: Record<string, string | number | undefined> = {}): Promise<T> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined) qs.set(k, String(v));
  const url = `${API_BASE}/api${path}${qs.toString() ? `?${qs}` : ""}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export interface Category { name: string; low: number; high: number; color: string }
export interface Freshness { data_through: string | null; last_ingest: string | null; age_minutes: number | null; stale: boolean; note: string }
export interface Pollutant {
  key: string; label: string; value: number | null; unit: string; average: number | null; average_unit: string;
  window: string; sub_index: number | null; counts_toward_aqi: boolean;
}
export interface Current {
  location: { id: string; name: string; kind: string };
  observed_at: string;
  freshness: Freshness;
  aqi: {
    value: number; category: string; color: string; advice: string; dominant: string; dominant_label: string;
    change_3h: number | null; basis: string; basis_note: string;
  };
  pollutants: Pollutant[];
  weather: { temperature_c: number | null; humidity_pct: number | null; wind_kmh: number | null; wind_direction_deg: number | null; boundary_layer_m: number | null; pressure_hpa: number | null; as_of: string | null; lag_hours: number | null };
  categories: Category[];
}
export interface Quantiles { q10: number | null; q50: number | null; q90: number | null }
export interface ForecastPoint { target_ts: string; horizon: number; pm2_5: Quantiles; pm10: Quantiles; aqi: Quantiles; category: string | null; color: string | null }
export interface Forecast {
  location: { id: string; name: string };
  issued_at: string;
  freshness: Freshness;
  basis: string;
  interval: string;
  recent: { t: string; pm2_5: number | null; aqi: number | null }[];
  points: ForecastPoint[];
  categories: Category[];
  pm25_category_bands: Category[];
  error_reference: Record<string, { model_mae: number; persistence_mae: number }> | null;
  caveat: string;
}
export interface HistoryPoint { t: string; v: number | null; min?: number | null; max?: number | null }
export interface History { location: string; metric: string; unit: string; resolution: string; timezone: string; points: HistoryPoint[] }
export interface Seasonal {
  months: string[]; month_by_hour: (number | null)[][]; year_month: { year: number; month: number; value: number | null }[];
  unit: string; timezone: string;
}
export interface Stations {
  cells: { id: string; name: string; lat: number; lon: number; aqi: number; category: string; color: string; observed_at: string }[];
  neighbourhoods: { name: string; lat: number; lon: number; cell: string; aqi: number; category: string; color: string }[];
  ground_stations: { id: number; name: string; lat: number; lon: number; pm2_5: number; pm2_5_sub_index: number; category: string | null; color: string | null; observed_at: string | null }[];
  ground_stations_enabled: boolean;
  note: string;
}
export type HorizonTable = Record<string, Record<string, { mae: number; rmse: number; n: number }>>;
export interface TargetMetrics {
  by_horizon: HorizonTable;
  skill_vs_persistence: Record<string, Record<string, number>>;
  skill_vs_best_baseline: Record<string, Record<string, number>>;
  interval: { coverage: Record<string, number>; mean_width: Record<string, number>; nominal: number };
  interval_winter: { coverage: Record<string, number>; mean_width: Record<string, number>; nominal: number };
  by_season: Record<"winter" | "other", { n: number; by_horizon: HorizonTable }>;
  by_fold: ({ fold: number; cutoff: string; n: number } & Record<string, number | string>)[];
  by_location: Record<string, Record<string, number>>;
  n_rows: number;
}
export interface ModelMetrics {
  evaluation: {
    generated_at: string; source?: string; data_range: { start: string; end: string }; scope: string;
    config: { n_folds: number; test_days: number; min_train_days: number; origin_step_h: number; horizons: number[] };
    folds: { fold: number; cutoff: string; test_end: string; train_rows: number; test_rows: number }[];
    targets: Record<string, TargetMetrics>;
    sarima: { scope: string; n_rows: number; by_horizon: HorizonTable } | null;
    feature_importance: [string, number][];
    procedure: string[];
  };
  production_model: { trained_at: string; training_rows?: number; data_through?: string } | null;
  aqi_basis: string;
  limitations: string[];
}
export interface Health { status: string; data_through: string | null; last_ingest: string | null; last_forecast: string | null; model_trained_at: string | null; aqi_basis: string; stale: boolean }
