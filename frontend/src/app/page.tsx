"use client";

import { useMemo, useState } from "react";
import { ForecastChart } from "@/components/forecast-chart";
import { SeriesPicker, useSeriesSelection } from "@/components/series-picker";
import { Async, Card, EmptyState, ErrorState, Select, Spinner, StatTile } from "@/components/ui";
import { useApi } from "@/lib/api";
import { longDate, modelColor, modelLabel, num, pct, signedPct, sortModels } from "@/lib/format";
import type { ForecastResponse, MetricsResponse } from "@/lib/types";

export default function ForecastExplorer() {
  const sel = useSeriesSelection();
  const [fold, setFold] = useState("0");
  const [historyDays, setHistoryDays] = useState("90");
  const [picked, setPicked] = useState<string[] | null>(null);
  const [intervalChoice, setIntervalChoice] = useState<string | null>(null);

  const runRes = useApi<MetricsResponse>("/metrics?segment_type=overall&fold=0&model=naive");
  const path = sel.current
    ? `/forecast?item=${encodeURIComponent(sel.current.item_id)}&store=${encodeURIComponent(sel.current.store_id)}&fold=${fold}&history_days=${historyDays}`
    : null;
  const fcRes = useApi<ForecastResponse>(path);

  if (sel.error) return <ErrorState message={sel.error} onRetry={sel.reload} />;
  if (sel.loading && !sel.items.length) return <Spinner label="Loading series" />;
  if (!sel.items.length) {
    return (
      <EmptyState title="No series available">
        Train a model run first (<code>dfs train</code>) so forecasts exist in the database.
      </EmptyState>
    );
  }

  const run = runRes.data?.run;
  const foldOptions = [
    { value: "0", label: run ? `Next 28 days (${longDate(run.future_start)} →)` : "Next 28 days" },
    ...(run?.folds ?? []).map((f) => ({
      value: String(f.fold),
      label: `Validation fold ${f.fold} (${longDate(f.eval_start)} – ${longDate(f.eval_end)})`,
    })),
  ];

  return (
    <div className="space-y-5">
      <Card
        title="Forecast Explorer"
        subtitle="Daily demand 28 days ahead for one item-store series, with baselines and an empirical 80% prediction interval."
      >
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <SeriesPicker sel={sel} />
          <Select label="Forecast window" value={fold} onChange={setFold} options={foldOptions} />
          <Select
            label="History shown"
            value={historyDays}
            onChange={setHistoryDays}
            options={[28, 56, 90, 180, 365].map((d) => ({ value: String(d), label: `${d} days` }))}
          />
        </div>
      </Card>

      <Async res={fcRes} label="Loading forecast" isEmpty={(d) => Object.keys(d.models).length === 0} empty={<EmptyState title="No forecast stored for this series" />}>
        {(fc) => {
          const models = sortModels(Object.keys(fc.models));
          const best = fc.best_model ?? "xgboost";
          const chosen = (picked ?? [best, "moving_average_28"]).filter((m) => models.includes(m));
          const intervalModel = intervalChoice === "none" ? null : (intervalChoice && chosen.includes(intervalChoice) ? intervalChoice : (chosen.includes(best) ? best : chosen[0] ?? null));
          const main = fc.models[best] ?? [];
          const total = main.reduce((s, p) => s + p.y_pred, 0);
          const sm = fc.series_metrics;
          const mWape = sm[best]?.wape ?? null;
          const maWape = sm["moving_average_28"]?.wape ?? null;
          const intervalHasData = intervalModel !== null && (fc.models[intervalModel] ?? []).some((p) => p.lower_80 !== null);

          return (
            <div className="space-y-5">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <StatTile
                  label={`Forecast total, ${modelLabel(best)}`}
                  value={`${num(total, 0)} units`}
                  sub={`${longDate(fc.horizon_start)} – ${longDate(fc.horizon_end)} · ${num(total / Math.max(main.length, 1), 1)}/day`}
                />
                <StatTile
                  label={`Validation WAPE, ${modelLabel(best)}`}
                  value={pct(mWape)}
                  sub="This series, pooled over the 3 validation folds (84 days)"
                />
                <StatTile
                  label="vs 28-day moving average"
                  value={mWape !== null && maWape !== null ? `${mWape <= maWape ? "−" : "+"}${(Math.abs(mWape - maWape) * 100).toFixed(1)} pts` : "n/a"}
                  tone={mWape !== null && maWape !== null ? (mWape <= maWape ? "good" : "bad") : "neutral"}
                  sub={maWape !== null ? `Moving average WAPE ${pct(maWape)} (lower is better)` : "Baseline WAPE undefined"}
                />
              </div>

              <Card
                title={`${fc.item_id} · ${fc.store_id}`}
                subtitle={
                  fc.fold === 0
                    ? "Future forecast from the last observed day. No actuals exist yet; the interval uses residuals from all three validation folds."
                    : `Validation fold ${fc.fold}: models saw data only up to the origin, then forecast the next 28 days; actuals are shown for comparison.${
                        intervalHasData ? "" : " Fold 1 has no earlier residuals, so no interval is available."
                      }`
                }
                actions={
                  <div className="flex flex-wrap items-end gap-3">
                    <Select
                      label="Prediction interval"
                      value={intervalModel ?? "none"}
                      onChange={setIntervalChoice}
                      options={[{ value: "none", label: "Hidden" }, ...chosen.map((m) => ({ value: m, label: modelLabel(m) }))]}
                    />
                  </div>
                }
              >
                <fieldset className="mb-3 flex flex-wrap gap-2">
                  <legend className="sr-only">Models shown</legend>
                  {models.map((m) => {
                    const on = chosen.includes(m);
                    return (
                      <label
                        key={m}
                        className={`flex cursor-pointer items-center gap-2 rounded-full border px-3 py-1 text-[13px] ${
                          on ? "border-axis bg-page text-ink" : "border-line text-muted"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={on}
                          onChange={() => setPicked(on ? chosen.filter((x) => x !== m) : [...chosen, m])}
                        />
                        <i aria-hidden className="series-key inline-block h-2.5 w-2.5 rounded-full" style={{ background: modelColor(m) }} />
                        {modelLabel(m)}
                        {m === best && <span className="text-[11.5px] text-muted">best</span>}
                      </label>
                    );
                  })}
                </fieldset>
                {chosen.length === 0 ? (
                  <EmptyState title="Select at least one model" />
                ) : (
                  <ForecastChart fc={fc} models={chosen} intervalModel={intervalModel} />
                )}
                <DataTable fc={fc} models={chosen} />
              </Card>

              <Card
                title="Accuracy for this series"
                subtitle="Pooled over the 3 validation folds, identical dates for every model. Bias: positive = over-forecast."
              >
                <table className="w-full text-left text-[13.5px]">
                  <thead className="text-[12.5px] text-ink2">
                    <tr className="border-b border-line">
                      <th className="py-2 font-medium">Model</th>
                      <th className="py-2 text-right font-medium">WAPE</th>
                      <th className="py-2 text-right font-medium">MAE</th>
                      <th className="py-2 text-right font-medium">RMSE</th>
                      <th className="py-2 text-right font-medium">Bias</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((m) => {
                      const x = sm[m];
                      return (
                        <tr key={m} className="border-b border-line/60 last:border-0">
                          <td className="py-2">
                            <span className="inline-flex items-center gap-2">
                              <i aria-hidden className="series-key inline-block h-2.5 w-2.5 rounded-full" style={{ background: modelColor(m) }} />
                              {modelLabel(m)}
                            </span>
                          </td>
                          <td className="py-2 text-right font-medium">{x ? pct(x.wape) : "—"}</td>
                          <td className="py-2 text-right">{x ? num(x.mae) : "—"}</td>
                          <td className="py-2 text-right">{x ? num(x.rmse) : "—"}</td>
                          <td className="py-2 text-right">{x ? signedPct(x.bias_pct) : "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </Card>
            </div>
          );
        }}
      </Async>
    </div>
  );
}

function DataTable({ fc, models }: { fc: ForecastResponse; models: string[] }) {
  const rows = useMemo(() => {
    const first = fc.models[models[0]] ?? [];
    return first.map((p, i) => ({ date: p.date, actual: p.actual, values: models.map((m) => fc.models[m]?.[i]?.y_pred) }));
  }, [fc, models]);
  if (!models.length) return null;
  return (
    <details className="mt-4 text-[13px]">
      <summary className="cursor-pointer text-ink2">View forecast table</summary>
      <div className="mt-2 max-h-72 overflow-auto rounded-md border border-line">
        <table className="w-full text-right">
          <thead className="sticky top-0 bg-surface text-ink2">
            <tr>
              <th className="px-3 py-1.5 text-left font-medium">Date</th>
              <th className="px-3 py-1.5 font-medium">Actual</th>
              {models.map((m) => (
                <th key={m} className="px-3 py-1.5 font-medium">
                  {modelLabel(m)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.date} className="border-t border-line/60">
                <td className="px-3 py-1 text-left">{r.date}</td>
                <td className="px-3 py-1">{r.actual ?? "—"}</td>
                {r.values.map((v, i) => (
                  <td key={i} className="px-3 py-1">
                    {v === undefined ? "—" : num(v, 1)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
