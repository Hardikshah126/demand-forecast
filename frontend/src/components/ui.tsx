"use client";

import type { ReactNode } from "react";
import type { Resource } from "@/lib/api";

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-line bg-surface p-5 ${className}`}>
      {(title || actions) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-[15px] font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 max-w-3xl text-[13px] text-ink2">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatTile({
  label,
  value,
  sub,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "neutral" | "good" | "bad";
}) {
  const color = tone === "good" ? "text-good" : tone === "bad" ? "text-crit" : "text-ink";
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3.5">
      <div className="text-[12.5px] text-ink2">{label}</div>
      <div className={`mt-1 text-[26px] font-semibold leading-8 ${color}`}>{value}</div>
      {sub && <div className="mt-1 text-[12.5px] text-muted">{sub}</div>}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-2.5 py-10 text-ink2">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-axis border-t-accent" />
      {label}…
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-lg border border-crit/40 bg-crit/5 p-4">
      <div className="flex items-start gap-2.5">
        <span aria-hidden className="mt-0.5 font-bold text-crit">
          ⚠
        </span>
        <div>
          <div className="font-medium text-crit">Something went wrong</div>
          <p className="mt-0.5 text-ink2">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-2.5 rounded-md border border-line px-3 py-1.5 text-[13px] font-medium hover:bg-[var(--hover)]"
            >
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-axis p-8 text-center">
      <div className="font-medium text-ink">{title}</div>
      {children && <p className="mx-auto mt-1 max-w-md text-ink2">{children}</p>}
    </div>
  );
}

/** Renders loading / error / empty / content for a Resource. */
export function Async<T>({
  res,
  label,
  isEmpty,
  empty,
  children,
}: {
  res: Resource<T>;
  label?: string;
  isEmpty?: (d: T) => boolean;
  empty?: ReactNode;
  children: (d: T) => ReactNode;
}) {
  if (res.error) return <ErrorState message={res.error} onRetry={res.reload} />;
  if (res.loading && !res.data) return <Spinner label={label} />;
  if (!res.data) return null;
  if (isEmpty?.(res.data)) return <>{empty}</>;
  return (
    <div className={res.loading ? "opacity-60 transition-opacity" : "transition-opacity"}>{children(res.data)}</div>
  );
}

export function Select({
  label,
  value,
  onChange,
  options,
  disabled,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  disabled?: boolean;
}) {
  return (
    <label className="flex min-w-[9rem] flex-col gap-1">
      <span className="text-[12.5px] font-medium text-ink2">{label}</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="h-9 rounded-md border border-axis bg-surface px-2.5 text-[13.5px] text-ink disabled:opacity-50"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Segmented({
  value,
  onChange,
  options,
  ariaLabel,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  ariaLabel: string;
}) {
  return (
    <div role="group" aria-label={ariaLabel} className="inline-flex rounded-lg border border-line bg-page p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors ${
            value === o.value ? "bg-surface text-ink shadow-sm ring-1 ring-line" : "text-ink2 hover:text-ink"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  display,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  display: string;
  hint?: string;
}) {
  const id = `slider-${label.replace(/\W+/g, "-").toLowerCase()}`;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-[12.5px] font-medium text-ink2">
          {label}
        </label>
        <span className="text-[13.5px] font-semibold text-ink">{display}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1.5 w-full"
      />
      {hint && <div className="flex justify-between text-[11.5px] text-muted">{hint}</div>}
    </div>
  );
}

export function SeriesKey({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[12.5px] text-ink2">
      <span
        aria-hidden
        className="series-key inline-block h-[3px] w-5 rounded-full"
        style={{
          background: dashed ? "transparent" : color,
          borderTop: dashed ? `2px dashed ${color}` : undefined,
        }}
      />
      {label}
    </span>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "good" | "warn" | "bad" }) {
  const cls =
    tone === "good"
      ? "text-good border-good/40 bg-good/10"
      : tone === "warn"
        ? "text-warn border-warn/40 bg-warn/10"
        : tone === "bad"
          ? "text-crit border-crit/40 bg-crit/10"
          : "text-ink2 border-line bg-page";
  return <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[12px] font-medium ${cls}`}>{children}</span>;
}
