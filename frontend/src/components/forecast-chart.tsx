"use client";

import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { modelColor, modelLabel, num, shortDate, sortModels } from "@/lib/format";
import type { ForecastResponse } from "@/lib/types";
import { SeriesKey } from "./ui";

type Row = { date: string; actual?: number; band?: [number, number] } & Record<string, number | string | [number, number] | undefined>;

export function buildRows(fc: ForecastResponse, models: string[], intervalModel: string | null): Row[] {
  const rows = new Map<string, Row>();
  const row = (d: string) => {
    let r = rows.get(d);
    if (!r) rows.set(d, (r = { date: d }));
    return r;
  };
  fc.history.forEach((h) => (row(h.date).actual = h.actual));
  for (const m of models) {
    for (const p of fc.models[m] ?? []) {
      const r = row(p.date);
      r[m] = p.y_pred;
      if (p.actual !== null) r.actual = p.actual;
      if (m === intervalModel && p.lower_80 !== null && p.upper_80 !== null) r.band = [p.lower_80, p.upper_80];
    }
  }
  return [...rows.values()].sort((a, b) => a.date.localeCompare(b.date));
}

function ChartTooltip({
  active,
  payload,
  label,
  models,
  intervalModel,
}: {
  active?: boolean;
  payload?: { payload: Row }[];
  label?: string;
  models: string[];
  intervalModel: string | null;
}) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-[12.5px] shadow-md">
      <div className="mb-1 font-semibold text-ink">{label ? shortDate(label) : ""}</div>
      {r.actual !== undefined && (
        <div className="flex items-center justify-between gap-6 text-ink2">
          <span className="flex items-center gap-1.5">
            <i className="inline-block h-2 w-2 rounded-full bg-ink" /> Actual
          </span>
          <b className="text-ink">{num(r.actual, 0)}</b>
        </div>
      )}
      {models.map(
        (m) =>
          typeof r[m] === "number" && (
            <div key={m} className="flex items-center justify-between gap-6 text-ink2">
              <span className="flex items-center gap-1.5">
                <i className="inline-block h-2 w-2 rounded-full" style={{ background: modelColor(m) }} />
                {modelLabel(m)}
              </span>
              <b className="text-ink">{num(r[m] as number, 1)}</b>
            </div>
          ),
      )}
      {r.band && intervalModel && (
        <div className="mt-1 border-t border-line pt-1 text-muted">
          80% interval ({modelLabel(intervalModel)}): {num(r.band[0], 1)} – {num(r.band[1], 1)}
        </div>
      )}
    </div>
  );
}

export function ForecastChart({
  fc,
  models,
  intervalModel,
}: {
  fc: ForecastResponse;
  models: string[];
  intervalModel: string | null;
}) {
  const ordered = useMemo(() => sortModels(models), [models]);
  const data = useMemo(() => buildRows(fc, ordered, intervalModel), [fc, ordered, intervalModel]);
  const hasBand = intervalModel !== null && data.some((d) => d.band);

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-x-5 gap-y-1">
        <SeriesKey color="var(--ink)" label="Actual demand" />
        {ordered.map((m) => (
          <SeriesKey key={m} color={modelColor(m)} label={modelLabel(m)} dashed={false} />
        ))}
        {hasBand && intervalModel && (
          <span className="inline-flex items-center gap-1.5 text-[12.5px] text-ink2">
            <span aria-hidden className="series-key inline-block h-3 w-5 rounded-sm" style={{ background: "var(--wash)" }} />
            80% interval ({modelLabel(intervalModel)})
          </span>
        )}
      </div>
      <div role="img" aria-label={`Actual and forecast demand for ${fc.item_id} in ${fc.store_id}`} className="h-[380px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="date"
              tickFormatter={shortDate}
              minTickGap={32}
              tickLine={false}
              axisLine={{ stroke: "var(--axis)" }}
            />
            <YAxis tickLine={false} axisLine={false} width={40} allowDecimals={false} />
            <Tooltip
              content={<ChartTooltip models={ordered} intervalModel={intervalModel} />}
              cursor={{ stroke: "var(--axis)" }}
            />
            <ReferenceLine
              x={fc.horizon_start}
              stroke="var(--axis)"
              label={{ value: "forecast start", position: "insideTopLeft", fill: "var(--muted)", fontSize: 12 }}
            />
            {hasBand && (
              <Area
                dataKey="band"
                stroke="none"
                fill="var(--wash)"
                isAnimationActive={false}
                activeDot={false}
                legendType="none"
              />
            )}
            <Line
              dataKey="actual"
              stroke="var(--ink)"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              dot={false}
              activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2 }}
              isAnimationActive={false}
              connectNulls
            />
            {ordered.map((m) => (
              <Line
                key={m}
                dataKey={m}
                stroke={modelColor(m)}
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
                dot={false}
                activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2 }}
                isAnimationActive={false}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
