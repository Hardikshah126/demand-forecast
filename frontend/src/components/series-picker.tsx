"use client";

import { useCallback, useMemo, useState } from "react";
import { useApi } from "@/lib/api";
import type { SeriesItem, SeriesListResponse } from "@/lib/types";
import { Select } from "./ui";

const KEY = "dfs.series";

export interface SeriesSelection {
  items: SeriesItem[];
  loading: boolean;
  error: string | null;
  reload: () => void;
  current: SeriesItem | null;
  select: (item: string, store: string) => void;
}

function readStored(): { item: string; store: string } | null {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/** Loads all series once and keeps one shared selection (default: highest-volume series). */
export function useSeriesSelection(): SeriesSelection {
  const res = useApi<SeriesListResponse>("/series?limit=500");
  const items = useMemo(() => res.data?.items ?? [], [res.data]);
  const [chosen, setChosen] = useState<{ item: string; store: string } | null>(null);

  // Derived, not synced through an effect: explicit pick > remembered pick > highest-volume series.
  // `items` only exists after the client-side fetch, so reading localStorage here is hydration-safe.
  const current = useMemo(() => {
    if (!items.length) return null;
    const find = (k: { item: string; store: string } | null) =>
      k ? items.find((s) => s.item_id === k.item && s.store_id === k.store) : undefined;
    const best = [...items].sort((a, b) => (b.avg_daily_sales_28d ?? 0) - (a.avg_daily_sales_28d ?? 0))[0];
    return find(chosen) ?? find(readStored()) ?? best;
  }, [items, chosen]);

  const select = useCallback((item: string, store: string) => {
    setChosen({ item, store });
    try {
      window.localStorage.setItem(KEY, JSON.stringify({ item, store }));
    } catch {
      /* storage unavailable: selection still works for this session */
    }
  }, []);

  return { items, loading: res.loading, error: res.error, reload: res.reload, current, select };
}

export function SeriesPicker({ sel }: { sel: SeriesSelection }) {
  const stores = useMemo(() => [...new Set(sel.items.map((s) => s.store_id))].sort(), [sel.items]);
  const store = sel.current?.store_id ?? stores[0] ?? "";
  const itemsInStore = useMemo(
    () => sel.items.filter((s) => s.store_id === store).sort((a, b) => a.item_id.localeCompare(b.item_id)),
    [sel.items, store],
  );

  return (
    <div className="flex flex-wrap items-end gap-3">
      <Select
        label="Store"
        value={store}
        disabled={!stores.length}
        onChange={(st) => {
          const same = sel.items.find((s) => s.store_id === st && s.item_id === sel.current?.item_id);
          const first = same ?? sel.items.find((s) => s.store_id === st);
          if (first) sel.select(first.item_id, st);
        }}
        options={stores.map((s) => ({ value: s, label: s }))}
      />
      <Select
        label="Item"
        value={sel.current?.item_id ?? ""}
        disabled={!itemsInStore.length}
        onChange={(it) => sel.select(it, store)}
        options={itemsInStore.map((s) => ({ value: s.item_id, label: s.item_id }))}
      />
      {sel.current && (
        <div className="pb-2 text-[12.5px] text-muted">
          {sel.current.cat_id} · {sel.current.dept_id} · avg {(sel.current.avg_daily_sales_28d ?? 0).toFixed(1)} units/day (last 28 d)
        </div>
      )}
    </div>
  );
}
