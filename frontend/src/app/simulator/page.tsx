"use client";

import { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SeriesPicker, useSeriesSelection } from "@/components/series-picker";
import { Async, Badge, Card, EmptyState, ErrorState, SeriesKey, Slider, Spinner, StatTile } from "@/components/ui";
import { useApi, useDebouncedPost } from "@/lib/api";
import { int, num, pct, shortDate } from "@/lib/format";
import type { BacktestResponse, SimulateRequest, SimulateResponse } from "@/lib/types";

export default function Simulator() {
  const sel = useSeriesSelection();
  const [demand, setDemand] = useState(0);
  const [lead, setLead] = useState(7);
  const [service, setService] = useState(95);
  const [cover, setCover] = useState(14);

  const body: SimulateRequest | null = sel.current
    ? {
        item: sel.current.item_id,
        store: sel.current.store_id,
        demand_change_pct: demand,
        lead_time_days: lead,
        service_level: service / 100,
        initial_cover_days: cover,
      }
    : null;
  const sim = useDebouncedPost<SimulateRequest, SimulateResponse>("/simulate", body);

  if (sel.error) return <ErrorState message={sel.error} onRetry={sel.reload} />;
  if (sel.loading && !sel.items.length) return <Spinner label="Loading series" />;
  if (!sel.items.length) return <EmptyState title="No series available">Run <code>dfs train</code> first.</EmptyState>;

  return (
    <div className="space-y-5">
      <Card
        title="Stock Simulator"
        subtitle="10,000 Monte Carlo demand paths over the next 28 days, built from the stored forecast plus resampled validation residual paths. Move a control and the simulation re-runs on the backend."
      >
        <SeriesPicker sel={sel} />
        <div className="mt-5 grid gap-x-8 gap-y-5 md:grid-cols-2 xl:grid-cols-4">
          <Slider label="Demand change" value={demand} min={-30} max={50} step={5} onChange={setDemand} display={`${demand > 0 ? "+" : ""}${demand}%`} hint="−30% … +50%" />
          <Slider label="Lead time" value={lead} min={1} max={28} step={1} onChange={setLead} display={`${lead} day${lead === 1 ? "" : "s"}`} hint="1 … 28 days" />
          <Slider label="Target service level" value={service} min={80} max={99.5} step={0.5} onChange={setService} display={`${service.toFixed(1)}%`} hint="80% … 99.5%" />
          <Slider label="Starting stock" value={cover} min={0} max={60} step={1} onChange={setCover} display={`${cover} days of cover`} hint="0 … 60 days" />
        </div>
      </Card>

      {sim.error && <ErrorState message={sim.error} onRetry={sim.reload} />}
      {!sim.data && sim.loading && <Spinner label="Running simulation" />}
      {sim.data && <Results r={sim.data} stale={sim.loading} />}

      <Backtest />
    </div>
  );
}

function Results({ r, stale }: { r: SimulateResponse; stale: boolean }) {
  const rec = r.recommended;
  const cur = r.current;
  const data = useMemo(
    () =>
      r.dates.map((d, i) => ({
        date: d,
        p50: r.trajectory_percentiles.p50[i],
        band: [r.trajectory_percentiles.p10[i], r.trajectory_percentiles.p90[i]] as [number, number],
      })),
    [r],
  );
  const better = rec.stockout_probability <= cur.stockout_probability;

  return (
    <div className={`space-y-5 transition-opacity ${stale ? "opacity-60" : ""}`}>
      <div className="flex items-center gap-2 text-[12.5px] text-muted">
        <Badge>Simulated estimate</Badge>
        {r.item_id} · {r.store_id} · {shortDate(r.horizon_start)} – {shortDate(r.horizon_end)} · {r.n_paths.toLocaleString()} paths · seed {r.seed} · forecast model {r.model}
      </div>

      <div className="grid grid-cols-2 gap-4 xl:grid-cols-5">
        <StatTile
          label="P(stockout) with recommended ROP"
          value={pct(rec.stockout_probability)}
          tone={rec.stockout_probability > 0.5 ? "bad" : "neutral"}
          sub={`at least one stockout day in 28 days · baseline ${pct(cur.stockout_probability)}`}
        />
        <StatTile label="Expected fill rate" value={pct(rec.expected_fill_rate, 2)} sub={`baseline ${pct(cur.expected_fill_rate, 2)}`} />
        <StatTile label="Recommended safety stock" value={`${int(r.safety_stock)} units`} sub={`z = ${r.z.toFixed(2)} × σ ${num(r.sigma_daily_error)} × √${r.order_quantity > 0 ? "L" : "L"}`} />
        <StatTile label="Recommended reorder point" value={`${int(r.reorder_point)} units`} sub={`baseline ${int(cur.reorder_point)}`} />
        <StatTile
          label="Expected lead-time demand"
          value={`${int(r.expected_lead_time_demand)} units`}
          sub={`${num(r.mean_daily_demand, 1)}/day · simulated ${int(r.simulated_lead_time_demand)}`}
        />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.4fr_1fr]">
        <Card
          title="Projected on-hand inventory"
          subtitle="Recommended policy. Line: median across paths; band: 10th–90th percentile. Dashed line: reorder point."
        >
          <div className="mb-2 flex gap-5">
            <SeriesKey color="var(--s1)" label="Median on-hand" />
            <span className="inline-flex items-center gap-1.5 text-[12.5px] text-ink2">
              <span aria-hidden className="series-key inline-block h-3 w-5 rounded-sm" style={{ background: "var(--wash)" }} />
              P10–P90
            </span>
            <SeriesKey color="var(--muted)" label="Reorder point" dashed />
          </div>
          <div role="img" aria-label="Projected inventory over 28 days" className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="date" tickFormatter={shortDate} minTickGap={28} tickLine={false} axisLine={{ stroke: "var(--axis)" }} />
                <YAxis tickLine={false} axisLine={false} width={44} />
                <Tooltip
                  cursor={{ stroke: "var(--axis)" }}
                  labelFormatter={(l) => shortDate(String(l))}
                  formatter={(v, n) => [Array.isArray(v) ? `${int(v[0])} – ${int(v[1])}` : int(Number(v)), n === "band" ? "P10–P90" : "Median"]}
                  contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12.5 }}
                  itemStyle={{ color: "var(--ink)" }}
                  labelStyle={{ color: "var(--ink)", fontWeight: 600 }}
                />
                <Area dataKey="band" stroke="none" fill="var(--wash)" isAnimationActive={false} activeDot={false} />
                <ReferenceLine y={rec.reorder_point} stroke="var(--muted)" strokeDasharray="5 4" />
                <Line dataKey="p50" stroke="var(--s1)" strokeWidth={2} strokeLinecap="round" dot={false} activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Current vs recommended" subtitle="Same starting stock and demand paths; only the reorder point differs.">
          <table className="w-full text-left text-[13.5px]">
            <thead className="text-[12.5px] text-ink2">
              <tr className="border-b border-line">
                <th className="py-2 font-medium" />
                <th className="py-2 text-right font-medium">Baseline</th>
                <th className="py-2 text-right font-medium">Recommended</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Reorder point", int(cur.reorder_point), int(rec.reorder_point)],
                ["Safety stock", "0", int(r.safety_stock)],
                ["P(stockout, 28 d)", pct(cur.stockout_probability), pct(rec.stockout_probability)],
                ["Expected fill rate", pct(cur.expected_fill_rate, 2), pct(rec.expected_fill_rate, 2)],
                ["Expected lost units", num(cur.expected_lost_units, 1), num(rec.expected_lost_units, 1)],
                ["Avg ending inventory", num(cur.expected_ending_inventory, 1), num(rec.expected_ending_inventory, 1)],
                ["Expected orders placed", num(cur.expected_orders, 1), num(rec.expected_orders, 1)],
              ].map(([k, a, b]) => (
                <tr key={k} className="border-b border-line/60 last:border-0">
                  <td className="py-2 text-ink2">{k}</td>
                  <td className="py-2 text-right">{a}</td>
                  <td className="py-2 text-right font-medium">{b}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-[13px] text-ink2">
            {better
              ? `Adding safety stock cuts simulated stockout probability from ${pct(cur.stockout_probability)} to ${pct(rec.stockout_probability)} and raises the expected fill rate by ${((rec.expected_fill_rate - cur.expected_fill_rate) * 100).toFixed(2)} percentage points, at the cost of ${num(rec.expected_ending_inventory - cur.expected_ending_inventory, 1)} more units held on average at the end.`
              : "In this scenario the recommended reorder point does not reduce stockout risk versus the baseline."}{" "}
            Baseline = reorder at expected lead-time demand, no safety stock.
          </p>
        </Card>
      </div>

      <Card title="Assumptions" subtitle="What this simulation does and does not claim.">
        <ul className="list-disc space-y-1.5 pl-5 text-[13.5px] text-ink2">
          {r.assumptions.map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      </Card>
    </div>
  );
}

function Backtest() {
  const res = useApi<BacktestResponse>("/simulation/backtest");
  return (
    <Card
      title="How well does the simulator match reality?"
      subtitle="Measured on held-out folds: for each series the simulator used only information available at the forecast origin, then the same policy was replayed on the demand that actually happened."
    >
      <Async res={res} label="Loading backtest" isEmpty={(d) => Object.keys(d.by_policy).length === 0} empty={<EmptyState title="No backtest stored" />}>
        {(d) => {
          const policies = Object.entries(d.by_policy);
          const rec = d.by_policy["recommended"];
          const points = (rec?.reliability ?? []).map((b) => ({ x: b.mean_predicted, y: b.realized_frequency, n: b.n, bin: b.bin }));
          return (
            <div className="space-y-5">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-left text-[13.5px]">
                  <thead className="text-[12.5px] text-ink2">
                    <tr className="border-b border-line">
                      <th className="py-2 font-medium">Policy (backtest, folds {d.folds.join(" & ")})</th>
                      <th className="py-2 text-right font-medium">Cases</th>
                      <th className="py-2 text-right font-medium">Predicted P(stockout)</th>
                      <th className="py-2 text-right font-medium">Realized frequency</th>
                      <th className="py-2 text-right font-medium">Predicted fill</th>
                      <th className="py-2 text-right font-medium">Realized fill</th>
                      <th className="py-2 text-right font-medium">Brier (vs constant)</th>
                      <th className="py-2 text-right font-medium">28-day demand inside P10–P90</th>
                    </tr>
                  </thead>
                  <tbody>
                    {policies.map(([k, p]) => (
                      <tr key={k} className="border-b border-line/60 last:border-0">
                        <td className="py-2 font-medium">{k === "recommended" ? "Recommended ROP" : "No safety stock"}</td>
                        <td className="py-2 text-right">{int(p.n)}</td>
                        <td className="py-2 text-right">{pct(p.mean_predicted_stockout_prob)}</td>
                        <td className="py-2 text-right">{pct(p.realized_stockout_frequency)}</td>
                        <td className="py-2 text-right">{pct(p.mean_predicted_fill_rate, 2)}</td>
                        <td className="py-2 text-right">{pct(p.mean_realized_fill_rate, 2)}</td>
                        <td className="py-2 text-right">
                          {num(p.brier_score, 3)} ({num(p.brier_score_constant_baseline, 3)})
                        </td>
                        <td className="py-2 text-right">{p.demand_total_coverage_80 === null ? "n/a" : pct(p.demand_total_coverage_80, 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {points.length > 0 && (
                <div className="grid gap-5 md:grid-cols-[1fr_1fr]">
                  <div>
                    <div className="mb-1 text-[13px] font-medium text-ink">Reliability (recommended ROP)</div>
                    <p className="mb-2 text-[12.5px] text-muted">Predicted vs realized stockout frequency per probability bin. Points on the diagonal = calibrated.</p>
                    <div role="img" aria-label="Reliability diagram" className="h-[260px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                          <CartesianGrid />
                          <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tickLine={false} axisLine={{ stroke: "var(--axis)" }} label={{ value: "predicted", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 12 }} />
                          <YAxis type="number" dataKey="y" domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tickLine={false} axisLine={false} width={44} />
                          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--muted)" strokeDasharray="5 4" />
                          <Tooltip
                            cursor={false}
                            formatter={(v, n) => [pct(Number(v)), n === "x" ? "Predicted" : "Realized"]}
                            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12.5 }}
                            itemStyle={{ color: "var(--ink)" }}
                          />
                          <Scatter data={points} fill="var(--s1)" stroke="var(--surface)" strokeWidth={2} isAnimationActive={false} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <table className="h-fit w-full text-left text-[13px]">
                    <thead className="text-[12.5px] text-ink2">
                      <tr className="border-b border-line">
                        <th className="py-2 font-medium">Predicted bin</th>
                        <th className="py-2 text-right font-medium">Cases</th>
                        <th className="py-2 text-right font-medium">Predicted</th>
                        <th className="py-2 text-right font-medium">Realized</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rec.reliability.map((b) => (
                        <tr key={b.bin} className="border-b border-line/60 last:border-0">
                          <td className="py-1.5">{b.bin}</td>
                          <td className="py-1.5 text-right">{int(b.n)}</td>
                          <td className="py-1.5 text-right">{pct(b.mean_predicted, 0)}</td>
                          <td className="py-1.5 text-right">{pct(b.realized_frequency, 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="text-[12.5px] text-muted">{d.description} Backtest model: {d.model}; each case is one series × fold × lead time × service level × starting-cover combination.</p>
            </div>
          );
        }}
      </Async>
    </Card>
  );
}
