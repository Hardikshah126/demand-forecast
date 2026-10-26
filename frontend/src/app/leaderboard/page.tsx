"use client";

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Async, Badge, Card, EmptyState, Segmented, SeriesKey } from "@/components/ui";
import { useApi } from "@/lib/api";
import { longDate, MODEL_META, modelColor, modelLabel, num, pct, signedPct, sortModels } from "@/lib/format";
import type { MetricRow, MetricsResponse } from "@/lib/types";

type Metric = "wape" | "mae" | "rmse" | "bias_pct";
const METRICS: { value: Metric; label: string; fmt: (v: number | null) => string }[] = [
  { value: "wape", label: "WAPE", fmt: (v) => pct(v) },
  { value: "mae", label: "MAE", fmt: (v) => num(v, 2) },
  { value: "rmse", label: "RMSE", fmt: (v) => num(v, 2) },
  { value: "bias_pct", label: "Bias", fmt: (v) => signedPct(v) },
];

export default function Leaderboard() {
  const res = useApi<MetricsResponse>("/metrics?segment_type=overall");
  return (
    <Async res={res} label="Loading leaderboard" isEmpty={(d) => d.rows.length === 0} empty={<EmptyState title="No metrics stored yet">Run <code>dfs train</code>.</EmptyState>}>
      {(d) => <Content data={d} />}
    </Async>
  );
}

function Content({ data }: { data: MetricsResponse }) {
  const [metric, setMetric] = useState<Metric>("wape");
  const fmt = METRICS.find((m) => m.value === metric)!.fmt;
  const run = data.run;

  const byModel = useMemo(() => {
    const out: Record<string, Record<number, MetricRow>> = {};
    for (const r of data.rows) (out[r.model] ??= {})[r.fold] = r;
    return out;
  }, [data.rows]);
  const models = sortModels(Object.keys(byModel));
  const ranked = [...models].sort((a, b) => (byModel[a][0]?.wape ?? Infinity) - (byModel[b][0]?.wape ?? Infinity));
  const best = ranked[0];
  const baselines = ranked.filter((m) => MODEL_META[m]?.kind === "baseline");
  const bestBaseline = baselines[0];
  const bestML = ranked.find((m) => m === "xgboost");

  const rel =
    bestML && bestBaseline && byModel[bestML][0].wape !== null && byModel[bestBaseline][0].wape !== null
      ? 1 - byModel[bestML][0].wape! / byModel[bestBaseline][0].wape!
      : null;

  const chartData = [1, 2, 3, 0].map((f) => ({
    name: f === 0 ? "All 3 folds" : `Fold ${f}`,
    ...Object.fromEntries(models.map((m) => [m, byModel[m][f]?.[metric] ?? null])),
  }));

  return (
    <div className="space-y-5">
      <Card
        title="Model Leaderboard"
        subtitle={
          <>
            Four forecasting strategies (plus one feature experiment) evaluated on three chronological 28-day folds,
            identical dates and identical {run.n_series} series for every model. Primary metric: WAPE (lower is better).
          </>
        }
      >
        <div className="grid gap-4 md:grid-cols-[1.2fr_1fr]">
          <div className="rounded-lg border border-line bg-page p-4">
            <div className="text-[12.5px] text-ink2">Best model by pooled WAPE (computed from stored results)</div>
            <div className="mt-1 flex flex-wrap items-center gap-3">
              <span className="text-[26px] font-semibold text-ink">{modelLabel(best)}</span>
              <Badge tone="good">✓ WAPE {pct(byModel[best][0].wape)}</Badge>
            </div>
            <p className="mt-2 text-[13.5px] text-ink2">
              {rel === null
                ? "Improvement versus the best baseline is undefined."
                : rel > 0
                  ? `XGBoost's pooled WAPE is ${(rel * 100).toFixed(1)}% lower (relative) than the best baseline, ${modelLabel(bestBaseline)} (${pct(byModel[bestBaseline][0].wape)}).`
                  : `XGBoost does not beat the best baseline: ${modelLabel(bestBaseline)} has the lower pooled WAPE (${pct(byModel[bestBaseline][0].wape)} vs ${pct(byModel[bestML!][0].wape)}).`}
            </p>
          </div>
          <div className="rounded-lg border border-line bg-page p-4 text-[13px] text-ink2">
            <div className="mb-1 text-[12.5px] font-medium text-ink">Evaluation period</div>
            <ul className="space-y-0.5">
              {run.folds.map((f) => (
                <li key={f.fold}>
                  <b className="text-ink">Fold {f.fold}</b>: trained through {longDate(f.origin)}, forecast {longDate(f.eval_start)} – {longDate(f.eval_end)}
                </li>
              ))}
            </ul>
            <p className="mt-1.5 text-muted">Expanding window; no random splits. Data ends {longDate(run.last_date)}.</p>
          </div>
        </div>
      </Card>

      <Card title="Results" subtitle="Metrics pooled over all item-store-days. Bias: positive = over-forecast. Coverage: share of actuals inside the empirical 80% interval, out-of-sample (folds 2–3 only).">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-[13.5px]">
            <thead className="text-[12.5px] text-ink2">
              <tr className="border-b border-line">
                <th className="py-2 font-medium">#</th>
                <th className="py-2 font-medium">Model</th>
                <th className="py-2 text-right font-medium">WAPE</th>
                <th className="py-2 text-right font-medium">MAE</th>
                <th className="py-2 text-right font-medium">RMSE</th>
                <th className="py-2 text-right font-medium">Bias</th>
                <th className="py-2 text-right font-medium">80% interval coverage</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((m, i) => {
                const r = byModel[m][0];
                return (
                  <tr key={m} className={`border-b border-line/60 last:border-0 ${m === best ? "bg-[var(--hover)]" : ""}`}>
                    <td className="py-2.5 pr-2 text-muted">{i + 1}</td>
                    <td className="py-2.5">
                      <span className="inline-flex items-center gap-2 font-medium">
                        <i aria-hidden className="series-key inline-block h-2.5 w-2.5 rounded-full" style={{ background: modelColor(m) }} />
                        {modelLabel(m)}
                        {m === best && <Badge tone="good">best</Badge>}
                        {MODEL_META[m]?.kind === "experiment" && <Badge>experiment</Badge>}
                      </span>
                    </td>
                    <td className="py-2.5 text-right font-semibold">{pct(r.wape)}</td>
                    <td className="py-2.5 text-right">{num(r.mae)}</td>
                    <td className="py-2.5 text-right">{num(r.rmse)}</td>
                    <td className="py-2.5 text-right">{signedPct(r.bias_pct)}</td>
                    <td className="py-2.5 text-right">{r.interval_coverage_80 === null ? "n/a" : pct(r.interval_coverage_80, 0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card
        title="By validation fold"
        subtitle="Each model on each of the three folds. A model that wins only one fold is less convincing than one that wins all three."
        actions={<Segmented ariaLabel="Metric" value={metric} onChange={(v) => setMetric(v as Metric)} options={METRICS.map((m) => ({ value: m.value, label: m.label }))} />}
      >
        <div className="mb-2 flex flex-wrap gap-x-5 gap-y-1">
          {models.map((m) => (
            <SeriesKey key={m} color={modelColor(m)} label={modelLabel(m)} />
          ))}
        </div>
        <div role="img" aria-label={`${metric} by fold and model`} className="h-[340px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barGap={2} barCategoryGap="22%">
              <CartesianGrid vertical={false} />
              <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: "var(--axis)" }} />
              <YAxis tickLine={false} axisLine={false} width={48} tickFormatter={(v: number) => (metric === "mae" || metric === "rmse" ? num(v, 1) : `${(v * 100).toFixed(0)}%`)} />
              <Tooltip
                cursor={{ fill: "var(--hover)" }}
                formatter={(v, name) => [fmt(typeof v === "number" ? v : null), modelLabel(String(name))]}
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12.5 }}
                itemStyle={{ color: "var(--ink)" }}
                labelStyle={{ color: "var(--ink)", fontWeight: 600 }}
              />
              {models.map((m) => (
                <Bar key={m} dataKey={m} fill={modelColor(m)} radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[560px] text-left text-[13.5px]">
            <thead className="text-[12.5px] text-ink2">
              <tr className="border-b border-line">
                <th className="py-2 font-medium">Model</th>
                {[1, 2, 3].map((f) => (
                  <th key={f} className="py-2 text-right font-medium">Fold {f}</th>
                ))}
                <th className="py-2 text-right font-medium">Pooled</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((m) => {
                const winners = [1, 2, 3].map((f) => ranked.reduce((a, b) => ((byModel[a][f]?.wape ?? Infinity) <= (byModel[b][f]?.wape ?? Infinity) ? a : b)));
                return (
                  <tr key={m} className="border-b border-line/60 last:border-0">
                    <td className="py-2">{modelLabel(m)}</td>
                    {[1, 2, 3].map((f, i) => (
                      <td key={f} className={`py-2 text-right ${winners[i] === m && metric === "wape" ? "font-semibold text-good" : ""}`}>
                        {fmt(byModel[m][f]?.[metric] ?? null)}
                        {winners[i] === m && metric === "wape" ? " ✓" : ""}
                      </td>
                    ))}
                    <td className="py-2 text-right font-medium">{fmt(byModel[m][0]?.[metric] ?? null)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {metric === "wape" && <p className="mt-2 text-[12.5px] text-muted">✓ marks the lowest WAPE on that fold.</p>}
        </div>
      </Card>
    </div>
  );
}
