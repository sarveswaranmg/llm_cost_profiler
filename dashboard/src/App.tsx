import { useCallback, useEffect, useState } from "react";
import { Attribution } from "./components/Attribution";
import { DetailDrawer } from "./components/DetailDrawer";
import { EmptyState } from "./components/EmptyState";
import { FlameGraph } from "./components/FlameGraph";
import { Sidebar } from "./components/Sidebar";
import { Timeline } from "./components/Timeline";
import { apiUrl } from "./lib/api";
import { useFetch, useLiveTraces } from "./lib/hooks";
import type { SpanNode, TraceFilters } from "./lib/types";

type Tab = "flame" | "timeline" | "attribution";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "flame", label: "Flamegraph" },
  { id: "timeline", label: "Timeline" },
  { id: "attribution", label: "Attribution" },
];

function initialUrlState(): { tab: Tab; trace: string | null } {
  const params = new URLSearchParams(window.location.search);
  const t = params.get("tab");
  return {
    tab: t === "timeline" || t === "attribution" ? t : "flame",
    trace: params.get("trace"),
  };
}

export default function App() {
  const [filters, setFilters] = useState<TraceFilters>({});
  const [tab, setTab] = useState<Tab>(() => initialUrlState().tab);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(
    () => initialUrlState().trace,
  );
  const [selectedSpan, setSelectedSpan] = useState<SpanNode | null>(null);

  const live = useLiveTraces(filters);

  // Keep tab + trace in the URL so views are shareable/deep-linkable.
  useEffect(() => {
    const params = new URLSearchParams();
    if (tab !== "flame") params.set("tab", tab);
    if (selectedTraceId) params.set("trace", selectedTraceId);
    const query = params.toString();
    window.history.replaceState(
      null,
      "",
      query ? `?${query}` : window.location.pathname,
    );
  }, [tab, selectedTraceId]);

  // Auto-select the newest trace when nothing is selected (or the selection
  // fell out of the filtered list).
  useEffect(() => {
    if (live.traces.length === 0) return;
    if (!selectedTraceId || !live.traces.some((t) => t.trace_id === selectedTraceId)) {
      setSelectedTraceId(live.traces[0].trace_id);
    }
  }, [live.traces, selectedTraceId]);

  const tree = useFetch<SpanNode>(
    selectedTraceId ? apiUrl(`/api/traces/${selectedTraceId}`) : null,
  );

  const selectTrace = useCallback((id: string) => {
    setSelectedTraceId(id);
    setSelectedSpan(null);
  }, []);

  const needsTrace = tab !== "attribution";

  return (
    <div className="flex h-full overflow-hidden bg-page">
      <Sidebar
        traces={live.traces}
        connected={live.connected}
        loading={live.loading}
        error={live.error}
        selectedId={selectedTraceId}
        onSelect={selectTrace}
        filters={filters}
        onFilters={setFilters}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <nav className="flex items-center gap-1 border-b border-hairline bg-surface px-3">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`relative px-3 py-2.5 text-[12px] transition-colors ${
                tab === t.id
                  ? "font-medium text-ink after:absolute after:inset-x-2 after:bottom-0 after:h-0.5 after:rounded-full after:bg-accent"
                  : "text-muted hover:text-ink-2"
              }`}
            >
              {t.label}
            </button>
          ))}
          {needsTrace && tree.data && (
            <span className="num ml-auto truncate pr-2 text-[11px] text-muted">
              {tree.data.name} · {selectedTraceId?.slice(0, 8)}
            </span>
          )}
        </nav>

        <div className="flex min-h-0 flex-1">
          <section className="min-w-0 flex-1">
            {tab === "attribution" ? (
              <Attribution />
            ) : tree.error ? (
              <EmptyState title="Couldn't load this trace" detail={tree.error} />
            ) : !selectedTraceId ? (
              <EmptyState
                title="No trace selected"
                detail="Pick a trace from the sidebar — or generate some traffic first:"
                hint="python examples/demo_langgraph_agent.py"
              />
            ) : tree.data ? (
              tab === "flame" ? (
                <FlameGraph
                  root={tree.data}
                  selectedId={selectedSpan?.span_id ?? null}
                  onSelect={setSelectedSpan}
                />
              ) : (
                <Timeline
                  root={tree.data}
                  selectedId={selectedSpan?.span_id ?? null}
                  onSelect={setSelectedSpan}
                />
              )
            ) : (
              <div className="flex h-full items-center justify-center text-muted">
                loading…
              </div>
            )}
          </section>

          {needsTrace && selectedSpan && (
            <DetailDrawer span={selectedSpan} onClose={() => setSelectedSpan(null)} />
          )}
        </div>
      </main>
    </div>
  );
}
