"use client";

import { useCallback, useEffect, useState } from "react";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(`Cannot reach the API at ${API_URL}. Is the backend running?`, 0);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail))
        detail = body.detail.map((d: { loc?: unknown[]; msg: string }) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ");
    } catch {
      /* keep default message */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export interface Resource<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

interface Settled<T> {
  key: string;
  data: T | null;
  error: string | null;
}

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const aborted = (e: unknown): boolean => e instanceof DOMException && e.name === "AbortError";

/** Derive the public Resource from the last settled request: loading = the settled key is stale. */
function toResource<T>(state: Settled<T> | null, key: string | null, reload: () => void): Resource<T> {
  const fresh = state !== null && state.key === key;
  return {
    data: state?.data ?? null, // stale-while-revalidate: keep the previous result visible
    error: fresh ? state.error : null,
    loading: key !== null && !fresh,
    reload,
  };
}

/** GET a path (or nothing when `path` is null). Aborts stale requests when the path changes. */
export function useApi<T>(path: string | null): Resource<T> {
  const [state, setState] = useState<Settled<T> | null>(null);
  const [tick, setTick] = useState(0);
  const key = path === null ? null : `${path}#${tick}`;

  useEffect(() => {
    if (path === null || key === null) return;
    const ctl = new AbortController();
    request<T>(path, { signal: ctl.signal })
      .then((data) => setState({ key, data, error: null }))
      .catch((e: unknown) => {
        if (!aborted(e)) setState({ key, data: null, error: message(e) });
      });
    return () => ctl.abort();
  }, [path, key]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return toResource(state, key, reload);
}

/** POST a body; debounced so dragging a slider does not fire a request per pixel. */
export function useDebouncedPost<TReq, TRes>(path: string, body: TReq | null, delayMs = 350): Resource<TRes> {
  const [state, setState] = useState<Settled<TRes> | null>(null);
  const [tick, setTick] = useState(0);
  const payload = body === null ? null : JSON.stringify(body);
  const key = payload === null ? null : `${payload}#${tick}`;

  useEffect(() => {
    if (payload === null || key === null) return;
    const ctl = new AbortController();
    const timer = setTimeout(() => {
      request<TRes>(path, { method: "POST", body: payload, signal: ctl.signal })
        .then((data) => setState({ key, data, error: null }))
        .catch((e: unknown) => {
          if (!aborted(e)) setState((s) => ({ key, data: s?.data ?? null, error: message(e) }));
        });
    }, delayMs);
    return () => {
      clearTimeout(timer);
      ctl.abort();
    };
  }, [payload, key, path, delayMs]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return toResource(state, key, reload);
}
