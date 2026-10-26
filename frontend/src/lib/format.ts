export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v) ? "undefined" : `${(v * 100).toFixed(digits)}%`;

export const signedPct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v) ? "undefined" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });

export const int = (v: number | null | undefined): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Math.round(v).toLocaleString("en-US");

export const shortDate = (iso: string): string => {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

export const longDate = (iso: string): string => {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
};

/** Display metadata per model. Colors follow the entity (model), never the rank. */
export const MODEL_META: Record<string, { label: string; color: string; kind: "baseline" | "ml" | "experiment" }> = {
  xgboost: { label: "XGBoost", color: "var(--s1)", kind: "ml" },
  moving_average_28: { label: "28-day moving average", color: "var(--s2)", kind: "baseline" },
  seasonal_naive: { label: "Seasonal naive", color: "var(--s3)", kind: "baseline" },
  naive: { label: "Naive (last value)", color: "var(--s4)", kind: "baseline" },
  xgboost_v2: { label: "XGBoost v2 (experiment)", color: "var(--s5)", kind: "experiment" },
};

export const modelLabel = (m: string): string => MODEL_META[m]?.label ?? m;
export const modelColor = (m: string): string => MODEL_META[m]?.color ?? "var(--muted)";

export const MODEL_ORDER = ["xgboost", "xgboost_v2", "moving_average_28", "seasonal_naive", "naive"];
export const sortModels = (ms: string[]): string[] =>
  [...ms].sort((a, b) => MODEL_ORDER.indexOf(a) - MODEL_ORDER.indexOf(b));
