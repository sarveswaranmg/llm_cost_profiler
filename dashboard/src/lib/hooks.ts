import { useCallback, useEffect, useRef, useState } from "react";
import { apiUrl, getJSON, liveWsUrl } from "./api";
import type { TraceFilters, TracePage, TraceSummary } from "./types";

export interface Fetched<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Fetch JSON whenever `url` changes; null url = fetch nothing. */
export function useFetch<T>(url: string | null): Fetched<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(url !== null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (url === null) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    getJSON<T>(url)
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setError(null);
        }
      })
      .catch((e: Error) => {
        if (!cancelled) {
          setData(null);
          setError(e.message);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [url, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, reload };
}

/** Measured width of a container element (ResizeObserver). */
export function useContainerWidth<T extends HTMLElement>(): [
  React.RefObject<T>,
  number,
] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setWidth(entry.contentRect.width);
    });
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

/** Re-render every `ms` so relative timestamps stay fresh. */
export function useNow(ms: number = 15_000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), ms);
    return () => window.clearInterval(id);
  }, [ms]);
  return now;
}

const LIST_LIMIT = 100;
const LIST_CAP = 250;

function summaryMatches(t: TraceSummary, f: TraceFilters): boolean {
  if (f.user_id && t.user_id !== f.user_id) return false;
  if (f.feature_tag && t.feature_tag !== f.feature_tag) return false;
  if (f.min_cost !== undefined && t.total_cost_usd < f.min_cost) return false;
  // `model` can't be checked client-side (summaries don't list models);
  // the caller refetches instead when a model filter is active.
  return true;
}

export interface LiveTraces {
  traces: TraceSummary[];
  connected: boolean;
  loading: boolean;
  error: string | null;
}

/** The sidebar feed: an initial (filtered) fetch merged with /ws/live
 * pushes, newest first. Reconnects with a small backoff. */
export function useLiveTraces(filters: TraceFilters): LiveTraces {
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const filtersRef = useRef(filters);
  filtersRef.current = filters;

  const listUrl = apiUrl("/api/traces", {
    limit: LIST_LIMIT,
    user_id: filters.user_id,
    feature_tag: filters.feature_tag,
    model: filters.model,
    min_cost: filters.min_cost,
  });

  const refetch = useCallback((url: string) => {
    setLoading(true);
    getJSON<TracePage>(url)
      .then((page) => {
        setTraces(page.traces);
        setError(null);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refetch(listUrl);
  }, [listUrl, refetch]);

  // One WebSocket for the lifetime of the component.
  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let retryId: number | undefined;

    const connect = () => {
      ws = new WebSocket(liveWsUrl());
      ws.onopen = () => setConnected(true);
      ws.onmessage = (event: MessageEvent<string>) => {
        const summary = JSON.parse(event.data) as TraceSummary;
        const f = filtersRef.current;
        if (f.model) {
          // Server-side knowledge needed — refresh the filtered list.
          refetch(
            apiUrl("/api/traces", {
              limit: LIST_LIMIT,
              user_id: f.user_id,
              feature_tag: f.feature_tag,
              model: f.model,
              min_cost: f.min_cost,
            }),
          );
          return;
        }
        if (!summaryMatches(summary, f)) return;
        setTraces((prev) =>
          [summary, ...prev.filter((t) => t.trace_id !== summary.trace_id)].slice(
            0,
            LIST_CAP,
          ),
        );
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retryId = window.setTimeout(connect, 2_000);
      };
      ws.onerror = () => ws?.close();
    };
    connect();

    return () => {
      closed = true;
      window.clearTimeout(retryId);
      ws?.close();
    };
  }, [refetch]);

  return { traces, connected, loading, error };
}
