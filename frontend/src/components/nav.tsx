"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useApi } from "@/lib/api";
import { longDate, modelLabel } from "@/lib/format";
import type { MetricsResponse } from "@/lib/types";

const TABS = [
  { href: "/", label: "Forecast Explorer" },
  { href: "/leaderboard", label: "Model Leaderboard" },
  { href: "/errors", label: "Error Explorer" },
  { href: "/simulator", label: "Stock Simulator" },
];

function RunBadge() {
  const res = useApi<MetricsResponse>("/metrics?segment_type=overall&fold=0&model=naive");
  if (res.error) return <span className="text-[12.5px] text-crit">API unreachable</span>;
  if (!res.data) return <span className="text-[12.5px] text-muted">Loading run…</span>;
  const r = res.data.run;
  return (
    <span className="text-[12.5px] text-muted">
      Run #{r.run_id} · {r.n_series} series · data through {longDate(r.last_date)} · best model{" "}
      <strong className="font-semibold text-ink2">{r.best_model ? modelLabel(r.best_model) : "n/a"}</strong>
    </span>
  );
}

export function Nav() {
  const path = usePathname();
  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto flex max-w-[1280px] flex-wrap items-center justify-between gap-x-6 gap-y-2 px-6 pt-4">
        <div>
          <div className="text-[17px] font-semibold tracking-tight text-ink">Demand Forecasting &amp; Stockout Simulation</div>
          <RunBadge />
        </div>
        <a
          href={`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/docs`}
          target="_blank"
          rel="noreferrer"
          className="text-[13px] text-ink2 underline-offset-2 hover:underline"
        >
          API docs ↗
        </a>
      </div>
      <nav aria-label="Views" className="mx-auto mt-3 flex max-w-[1280px] gap-1 px-6">
        {TABS.map((t) => {
          const active = path === t.href;
          return (
            <Link
              key={t.href}
              href={t.href}
              aria-current={active ? "page" : undefined}
              className={`-mb-px border-b-2 px-3 py-2.5 text-[13.5px] font-medium transition-colors ${
                active ? "border-accent text-ink" : "border-transparent text-ink2 hover:text-ink"
              }`}
            >
              {t.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
