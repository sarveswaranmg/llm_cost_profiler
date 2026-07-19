import { useEffect, useState } from "react";
import { TIER_COLORS } from "../lib/colors";
import { costTier, fmtTokens, fmtUSD, relTime } from "../lib/format";
import { useNow } from "../lib/hooks";
import type { TraceFilters, TraceSummary } from "../lib/types";
import { EmptyState } from "./EmptyState";

export function Sidebar({
  traces,
  connected,
  loading,
  error,
  selectedId,
  onSelect,
  filters,
  onFilters,
}: {
  traces: TraceSummary[];
  connected: boolean;
  loading: boolean;
  error: string | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  filters: TraceFilters;
  onFilters: (f: TraceFilters) => void;
}) {
  const now = useNow();
  const hasFilters =
    !!filters.user_id || !!filters.feature_tag || !!filters.model || filters.min_cost !== undefined;

  return (
    <aside className="flex w-80 shrink-0 flex-col border-r border-hairline bg-surface">
      <header className="flex items-center gap-2 border-b border-hairline px-4 py-3">
        <span className="flex items-end gap-0.5" aria-hidden>
          <span className="h-3.5 w-1.5 rounded-[2px] bg-llm" />
          <span className="h-2.5 w-1.5 rounded-[2px] bg-retriever" />
          <span className="h-1.5 w-1.5 rounded-[2px] bg-retry" />
        </span>
        <h1 className="num text-[14px] font-semibold tracking-tight">tokenlens</h1>
        <div className="ml-auto flex items-center gap-1.5 text-[11px] text-muted">
          <span
            className={`h-2 w-2 rounded-full ${connected ? "live-dot bg-good" : "bg-muted"}`}
            title={connected ? "live — traces stream in as they finish" : "live feed disconnected"}
          />
          {connected ? "live" : "offline"}
        </div>
      </header>

      <FilterBar filters={filters} onFilters={onFilters} />

      <div className="flex-1 overflow-y-auto">
        {error ? (
          <EmptyState
            title="Can't reach the tokenlens API"
            detail={error}
            hint="tokenlens server --port 8321"
          />
        ) : traces.length === 0 && !loading ? (
          hasFilters ? (
            <EmptyState title="No traces match these filters" />
          ) : (
            <EmptyState
              title="No traces yet"
              detail="Instrument your app with tokenlens, or generate sample traffic with the demo:"
              hint="python examples/demo_langgraph_agent.py"
            />
          )
        ) : (
          <ul>
            {traces.map((t) => (
              <TraceRow
                key={t.trace_id}
                trace={t}
                now={now}
                selected={t.trace_id === selectedId}
                onSelect={() => onSelect(t.trace_id)}
              />
            ))}
          </ul>
        )}
      </div>

      <footer className="border-t border-hairline px-4 py-2 text-[11px] text-muted">
        <span className="num">{traces.length}</span> trace
        {traces.length === 1 ? "" : "s"}
        {hasFilters ? " · filtered" : ""}
      </footer>
    </aside>
  );
}

function TraceRow({
  trace,
  now,
  selected,
  onSelect,
}: {
  trace: TraceSummary;
  now: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const tier = costTier(trace.total_cost_usd);
  return (
    <li>
      <button
        onClick={onSelect}
        className={`block w-full border-b border-hairline px-4 py-2.5 text-left transition-colors ${
          selected
            ? "border-l-2 border-l-accent bg-raised pl-[14px]"
            : "border-l-2 border-l-transparent pl-[14px] hover:bg-hover"
        }`}
      >
        <div className="flex items-baseline gap-2">
          <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium text-ink">
            {trace.root_name}
          </span>
          <span className="num shrink-0 text-[12px]" style={{ color: TIER_COLORS[tier] }}>
            {fmtUSD(trace.total_cost_usd)}
          </span>
        </div>
        <div className="mt-1 flex items-center gap-1.5 text-[11px] text-muted">
          <span className="shrink-0">{relTime(trace.started_at, now)}</span>
          <span aria-hidden>·</span>
          <span className="num shrink-0">{fmtTokens(trace.total_tokens)} tok</span>
          {trace.has_error && (
            <span className="shrink-0 rounded-[3px] bg-retry/20 px-1 py-px text-[10px] font-semibold tracking-wide text-error-ring">
              ERR
            </span>
          )}
          <span className="min-w-0 flex-1" />
          {trace.user_id && <Chip>{trace.user_id}</Chip>}
          {trace.feature_tag && <Chip>{trace.feature_tag}</Chip>}
        </div>
      </button>
    </li>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="max-w-[88px] truncate rounded-[3px] border border-hairline bg-raised px-1 py-px text-[10px] text-ink-2">
      {children}
    </span>
  );
}

function FilterBar({
  filters,
  onFilters,
}: {
  filters: TraceFilters;
  onFilters: (f: TraceFilters) => void;
}) {
  const [user, setUser] = useState(filters.user_id ?? "");
  const [feature, setFeature] = useState(filters.feature_tag ?? "");
  const [model, setModel] = useState(filters.model ?? "");
  const [minCost, setMinCost] = useState(
    filters.min_cost !== undefined ? String(filters.min_cost) : "",
  );

  // Debounce keystrokes into a filters object.
  useEffect(() => {
    const id = window.setTimeout(() => {
      const parsed = Number(minCost);
      onFilters({
        user_id: user.trim() || undefined,
        feature_tag: feature.trim() || undefined,
        model: model.trim() || undefined,
        min_cost: minCost.trim() !== "" && Number.isFinite(parsed) ? parsed : undefined,
      });
    }, 300);
    return () => window.clearTimeout(id);
  }, [user, feature, model, minCost, onFilters]);

  return (
    <div className="grid grid-cols-2 gap-1.5 border-b border-hairline px-3 py-2.5">
      <FilterInput placeholder="user" value={user} onChange={setUser} />
      <FilterInput placeholder="feature" value={feature} onChange={setFeature} />
      <FilterInput placeholder="model" value={model} onChange={setModel} />
      <FilterInput
        placeholder="min $"
        value={minCost}
        onChange={setMinCost}
        inputMode="decimal"
      />
    </div>
  );
}

function FilterInput({
  placeholder,
  value,
  onChange,
  inputMode,
}: {
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  inputMode?: "decimal";
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      inputMode={inputMode}
      spellCheck={false}
      className="w-full rounded-md border border-hairline bg-page px-2 py-1 text-[12px] text-ink placeholder:text-muted focus:border-accent/60 focus:outline-none"
    />
  );
}
