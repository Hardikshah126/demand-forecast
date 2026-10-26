"use client";

import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Async, Badge, Card, EmptyState, Segmented, Select, SeriesKey } from "@/components/ui";
import { useApi } from "@/lib/api";
import { modelColor, modelLabel, num, pct, signedPct, sortModels } from "@/lib/format";
import type { MetricRow, MetricsResponse, TopError, TopErrorsResponse } from "@/lib/types";

const SEGMENTS = [
  { value: "store", label: "Store" },
  { value: "category", label: "Category" },
  { value: "dept", label: "Department" },
  { value: "weekday", label: "Weekday" },
  { value: "horizon", label: "Horizon (days ahead)" },
];
const WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

function Direction({ d }: { d: TopError["direction"] }) {
  if (d === "overforecast") return <Badge tone="warn">▲ over-forecast</Badge>;
  if (d === "underforecast") return <Badge tone="bad">▼ under-forecast</Badge>;
  if (d === "balanced") return <Badge>● balanced</Badge>;
  return <Badge>undefined</Badge>;
}

export default function ErrorExplorer() {
  const overall = useApi<MetricsResponse>("/metrics?segment_type=overall&fold=0");
  const [model, setModel] = useState<string | null>(null);
  const best = overall.data?.run.best_model ?? null;
  const active = model ?? best;
  const models = overall.data ? sortModels([...new Set(overall.data.rows.map((r) => r.model))]) : [];

  return (
    <div className="space-y-5">
      <Card
        title="Error Explorer"
        subtitle="Where does the forecast fail, and in which direction? All numbers are validation results pooled over three 28-day folds."
        actions={
          <Select
            label="Model"
            value={active ?? ""}
            disabled={!models.length}
            onChange={setModel}
            options={models.map((m) => ({ value: m, label: modelLabel(m) }))}
          />
        }
      >
        <Async res={overall} label="Loading" isEmpty={(d) => d.rows.length === 0} empty={<EmptyState title="No metrics stored yet">Run <code>dfs train</code>.</EmptyState>}>
          {(d) => {
            const rows = d.rows.filter((r) => r.model === active);
            const o = rows[0];
            if (!o) return <EmptyState title="No results for this model" />;
            const dir =
              o.bias_pct === null
                ? "undefined"
                : o.bias_pct > 0.02
                  ? "over-forecasts"
                  : o.bias_pct < -0.02
                    ? "under-forecasts"
                    : "is roughly unbiased";
            return (
              <p className="text-[14px] text-ink2">
                <b className="text-ink">{modelLabel(active!)}</b> {dir} overall: total error is {signedPct(o.bias_pct)} of total demand (WAPE {pct(o.wape)}, MAE {num(o.mae)} units/day).
                Positive bias means the model forecasts more than was sold, which would tie up stock; negative bias risks stockouts.
              </p>
            );
          }}
        </Async>
      </Card>

      {active && <Breakdown model={active} />}
      {active && <TopSeries model={active} />}
    </div>
  );
}

function Breakdown({ model }: { model: string }) {
  const [seg, setSeg] = useState("category");
  const res = useApi<MetricsResponse>(`/metrics?segment_type=${seg}&fold=0`);
  const compare = "moving_average_28";

  return (
    <Card
      title="Error breakdown"
      subtitle={`WAPE by segment for ${modelLabel(model)} against the 28-day moving average, and the bias direction of each segment.`}
      actions={<Segmented ariaLabel="Breakdown dimension" value={seg} onChange={setSeg} options={SEGMENTS} />}
    >
      <Async res={res} label="Loading breakdown" isEmpty={(d) => d.rows.length === 0} empty={<EmptyState title="No breakdown available">This dimension could not be calculated for the current run.</EmptyState>}>
        {(d) => {
          const keys = [...new Set(d.rows.map((r) => r.segment_value))];
          const order = seg === "weekday" ? WEEKDAYS : seg === "horizon" ? keys.sort((a, b) => parseInt(a) - parseInt(b)) : keys.sort();
          const get = (m: string, k: string): MetricRow | undefined => d.rows.find((r) => r.model === m && r.segment_value === k);
          const data = order.map((k) => ({
            name: k,
            main: get(model, k)?.wape ?? null,
            base: get(compare, k)?.wape ?? null,
            bias: get(model, k)?.bias_pct ?? null,
            undefinedWape: get(model, k)?.wape === null,
          }));
          const showCompare = model !== compare;
          return (
            <div className="space-y-6">
              <div>
                <div className="mb-2 flex flex-wrap gap-x-5">
                  <SeriesKey color={modelColor(model)} label={modelLabel(model)} />
                  {showCompare && <SeriesKey color={modelColor(compare)} label={modelLabel(compare)} />}
                </div>
                <div role="img" aria-label={`WAPE by ${seg}`} className="h-[260px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barGap={2} barCategoryGap="25%">
                      <CartesianGrid vertical={false} />
                      <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: "var(--axis)" }} interval={0} />
                      <YAxis tickLine={false} axisLine={false} width={44} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip
                        cursor={{ fill: "var(--hover)" }}
                        formatter={(v, n) => [typeof v === "number" ? pct(v) : "undefined", n === "main" ? modelLabel(model) : modelLabel(compare)]}
                        contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12.5 }}
                        itemStyle={{ color: "var(--ink)" }}
                        labelStyle={{ color: "var(--ink)", fontWeight: 600 }}
                      />
                      <Bar dataKey="main" fill={modelColor(model)} radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
                      {showCompare && <Bar dataKey="base" fill={modelColor(compare)} radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div>
                <div className="mb-1 text-[13px] font-medium text-ink">Bias of {modelLabel(model)} (error ÷ demand)</div>
                <p className="mb-2 text-[12.5px] text-muted">Above zero = over-forecast, below zero = under-forecast.</p>
                <div role="img" aria-label={`Bias by ${seg}`} className="h-[200px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barCategoryGap="35%">
                      <CartesianGrid vertical={false} />
                      <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} />
                      <YAxis tickLine={false} axisLine={false} width={48} tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                      <ReferenceLine y={0} stroke="var(--axis)" />
                      <Tooltip
                        cursor={{ fill: "var(--hover)" }}
                        formatter={(v) => [typeof v === "number" ? signedPct(v) : "undefined", "Bias"]}
                        contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12.5 }}
                        itemStyle={{ color: "var(--ink)" }}
                        labelStyle={{ color: "var(--ink)", fontWeight: 600 }}
                      />
                      <Bar dataKey="bias" radius={[4, 4, 4, 4]} maxBarSize={24} isAnimationActive={false}>
                        {data.map((x) => (
                          <Cell key={x.name} fill={(x.bias ?? 0) >= 0 ? "var(--s2)" : "var(--s1)"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex gap-5 text-[12.5px] text-ink2">
                  <SeriesKey color="var(--s2)" label="over-forecast" />
                  <SeriesKey color="var(--s1)" label="under-forecast" />
                </div>
              </div>

              <table className="w-full text-left text-[13.5px]">
                <thead className="text-[12.5px] text-ink2">
                  <tr className="border-b border-line">
                    <th className="py-2 font-medium">Segment</th>
                    <th className="py-2 text-right font-medium">WAPE</th>
                    <th className="py-2 text-right font-medium">MAE</th>
                    <th className="py-2 text-right font-medium">Bias</th>
                    <th className="py-2 text-right font-medium">Demand (units)</th>
                    <th className="py-2 pl-4 font-medium">Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {order.map((k) => {
                    const r = get(model, k);
                    if (!r) return null;
                    const dirn = r.bias_pct === null ? "undefined" : r.bias_pct > 0.05 ? "overforecast" : r.bias_pct < -0.05 ? "underforecast" : "balanced";
                    return (
                      <tr key={k} className="border-b border-line/60 last:border-0">
                        <td className="py-2">{k}</td>
                        <td className="py-2 text-right font-medium">{pct(r.wape)}</td>
                        <td className="py-2 text-right">{num(r.mae)}</td>
                        <td className="py-2 text-right">{signedPct(r.bias_pct)}</td>
                        <td className="py-2 text-right">{num(r.total_actual, 0)}</td>
                        <td className="py-2 pl-4">
                          <Direction d={dirn} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {data.some((x) => x.undefinedWape) && (
                <p className="text-[12.5px] text-muted">WAPE is shown as “undefined” where a segment had zero total demand (division by zero is not reported as 0).</p>
              )}
            </div>
          );
        }}
      </Async>
    </Card>
  );
}

function TopSeries({ model }: { model: string }) {
  const [sortBy, setSortBy] = useState("wape");
  const res = useApi<TopErrorsResponse>(`/errors/top?n=10&model=${model}&sort_by=${sortBy}`);
  return (
    <Card
      title="Worst-performing series"
      subtitle={
        sortBy === "wape"
          ? "Highest WAPE among series with at least one unit sold per day on average (84-day validation window)."
          : "Largest total absolute error in units: where the forecast misses cost the most."
      }
      actions={
        <Segmented
          ariaLabel="Sort"
          value={sortBy}
          onChange={setSortBy}
          options={[
            { value: "wape", label: "By WAPE" },
            { value: "abs_error", label: "By absolute error" },
          ]}
        />
      }
    >
      <Async res={res} label="Loading worst series" isEmpty={(d) => d.items.length === 0} empty={<EmptyState title="No series to show" />}>
        {(d) => (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-[13.5px]">
              <thead className="text-[12.5px] text-ink2">
                <tr className="border-b border-line">
                  <th className="py-2 font-medium">Item</th>
                  <th className="py-2 font-medium">Store</th>
                  <th className="py-2 font-medium">Category</th>
                  <th className="py-2 text-right font-medium">WAPE</th>
                  <th className="py-2 text-right font-medium">Abs. error (units)</th>
                  <th className="py-2 text-right font-medium">Bias</th>
                  <th className="py-2 pl-4 font-medium">Direction</th>
                </tr>
              </thead>
              <tbody>
                {d.items.map((r) => (
                  <tr key={`${r.item_id}|${r.store_id}`} className="border-b border-line/60 last:border-0">
                    <td className="py-2 font-medium">{r.item_id}</td>
                    <td className="py-2">{r.store_id}</td>
                    <td className="py-2">{r.cat_id}</td>
                    <td className="py-2 text-right font-semibold">{pct(r.wape)}</td>
                    <td className="py-2 text-right">{num(r.abs_error, 0)}</td>
                    <td className="py-2 text-right">{signedPct(r.bias_pct)}</td>
                    <td className="py-2 pl-4">
                      <Direction d={r.direction} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Async>
    </Card>
  );
}
