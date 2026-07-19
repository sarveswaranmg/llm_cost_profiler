import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiUrl } from "../lib/api";
import { COST_BAR_COLOR } from "../lib/colors";
import { fmtPercent, fmtTokens, fmtUSD } from "../lib/format";
import { useFetch } from "../lib/hooks";
import type { AggregateEntry, Overview } from "../lib/types";
import { EmptyState } from "./EmptyState";

const TOP_N = 8;

/** Aggregate view over ALL stored traces (independent of the selection):
 * headline cards, cost-by-dimension bars, and the per-node cost table. */
export function Attribution() {
  const overview = useFetch<Overview>(apiUrl("/api/stats/overview"));
  const byFeature = useFetch<AggregateEntry[]>(
    apiUrl("/api/aggregate", { group_by: "feature_tag" }),
  );
  const byUser = useFetch<AggregateEntry[]>(
    apiUrl("/api/aggregate", { group_by: "user_id" }),
  );
  const byModel = useFetch<AggregateEntry[]>(
    apiUrl("/api/aggregate", { group_by: "model" }),
  );
  const byNode = useFetch<AggregateEntry[]>(
    apiUrl("/api/aggregate", { group_by: "node" }),
  );

  if (overview.data && overview.data.trace_count === 0) {
    return (
      <EmptyState
        title="Nothing to attribute yet"
        detail="Once traces are stored, this tab shows who and what is spending the budget."
        hint="python examples/demo_langgraph_agent.py"
      />
    );
  }

  const o = overview.data;
  const totalCost = o ? Number(o.total_cost_usd) : 0;
  const retryWaste = o ? Number(o.retry_waste_usd) : 0;

  return (
    <div className="h-full space-y-4 overflow-y-auto p-4">
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <StatCard label="total spend" value={o ? fmtUSD(o.total_cost_usd) : "…"} sub={o ? `${fmtTokens(o.total_tokens)} tokens · ${o.trace_count} traces` : ""} />
        <StatCard
          label="retry waste"
          value={o ? fmtUSD(o.retry_waste_usd) : "…"}
          sub={o && totalCost > 0 ? `${fmtPercent(retryWaste / totalCost)} of spend re-doing failed calls` : ""}
          alarm={retryWaste > 0}
        />
        <StatCard
          label="error rate"
          value={o ? fmtPercent(o.error_rate) : "…"}
          sub="traces with ≥1 failed span"
          alarm={(o?.error_rate ?? 0) > 0.05}
        />
        <StatCard
          label="avg cost / trace"
          value={o && o.trace_count > 0 ? fmtUSD(totalCost / o.trace_count) : "…"}
          sub={o ? `across ${o.trace_count} traces` : ""}
        />
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <CostBarCard title="cost by feature" entries={byFeature.data} />
        <CostBarCard title="cost by user" entries={byUser.data} />
        <CostBarCard title="cost by model" entries={byModel.data} />
      </div>

      <NodeTable entries={byNode.data} totalCost={totalCost} />
    </div>
  );
}

function StatCard({
  label,
  value,
  sub,
  alarm,
}: {
  label: string;
  value: string;
  sub?: string;
  alarm?: boolean;
}) {
  return (
    <div className="rounded-lg border border-hairline bg-surface px-4 py-3">
      <div className="text-[10.5px] font-semibold uppercase tracking-[0.08em] text-muted">
        {label}
      </div>
      <div className={`num mt-1 text-[22px] leading-tight ${alarm ? "text-error-ring" : "text-ink"}`}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

interface BarDatum {
  key: string;
  cost: number;
  calls: number;
  tokens: number;
}

function foldTop(entries: AggregateEntry[]): BarDatum[] {
  const data = entries.map((e) => ({
    key: e.key,
    cost: Number(e.cost_usd),
    calls: e.call_count,
    tokens: e.total_tokens,
  }));
  if (data.length <= TOP_N) return data;
  const rest = data.slice(TOP_N);
  return [
    ...data.slice(0, TOP_N),
    {
      key: `other (${rest.length})`,
      cost: rest.reduce((s, d) => s + d.cost, 0),
      calls: rest.reduce((s, d) => s + d.calls, 0),
      tokens: rest.reduce((s, d) => s + d.tokens, 0),
    },
  ];
}

function CostBarCard({
  title,
  entries,
}: {
  title: string;
  entries: AggregateEntry[] | null;
}) {
  const data = useMemo(() => (entries ? foldTop(entries) : []), [entries]);
  return (
    <div className="rounded-lg border border-hairline bg-surface px-4 py-3">
      <h3 className="mb-2 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-muted">
        {title}
      </h3>
      {data.length === 0 ? (
        <div className="flex h-24 items-center justify-center text-[11.5px] text-muted">
          no attributed spans
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={data.length * 26 + 30}>
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid horizontal={false} stroke="var(--color-grid)" />
            <XAxis
              type="number"
              tickFormatter={(v: number) => fmtUSD(v)}
              tick={{ fill: "var(--color-muted)", fontSize: 10 }}
              axisLine={{ stroke: "var(--color-grid)" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="key"
              width={104}
              tickFormatter={(v: string) => (v.length > 15 ? `${v.slice(0, 14)}…` : v)}
              tick={{ fill: "var(--color-ink-2)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <RechartsTooltip
              cursor={{ fill: "rgba(255,255,255,0.04)" }}
              content={<BarTooltip />}
            />
            <Bar dataKey="cost" barSize={14} radius={[0, 3, 3, 0]} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.key} fill={COST_BAR_COLOR} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

interface BarTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: BarDatum }>;
}

function BarTooltip({ active, payload }: BarTooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-md border border-hairline bg-raised px-3 py-2 text-[11.5px] shadow-lg">
      <div className="mb-0.5 font-medium text-ink">{d.key}</div>
      <div className="num text-ink-2">{fmtUSD(d.cost)}</div>
      <div className="num text-muted">
        {d.calls} call{d.calls === 1 ? "" : "s"} · {fmtTokens(d.tokens)} tok
      </div>
    </div>
  );
}

type SortKey = "key" | "call_count" | "total_tokens" | "avg" | "cost";

function NodeTable({
  entries,
  totalCost,
}: {
  entries: AggregateEntry[] | null;
  totalCost: number;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("cost");
  const [desc, setDesc] = useState(true);

  const rows = useMemo(() => {
    if (!entries) return [];
    const value = (e: AggregateEntry): number | string => {
      switch (sortKey) {
        case "key":
          return e.key.toLowerCase();
        case "call_count":
          return e.call_count;
        case "total_tokens":
          return e.total_tokens;
        case "avg":
          return Number(e.avg_cost_per_call);
        case "cost":
          return Number(e.cost_usd);
      }
    };
    return [...entries].sort((a, b) => {
      const va = value(a);
      const vb = value(b);
      const cmp = va < vb ? -1 : va > vb ? 1 : 0;
      return desc ? -cmp : cmp;
    });
  }, [entries, sortKey, desc]);

  const maxCost = rows.reduce((m, e) => Math.max(m, Number(e.cost_usd)), 0);

  const header = (key: SortKey, label: string, right = true) => (
    <th
      onClick={() => {
        if (sortKey === key) setDesc((d) => !d);
        else {
          setSortKey(key);
          setDesc(key !== "key");
        }
      }}
      className={`cursor-pointer select-none whitespace-nowrap px-3 py-2 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-muted hover:text-ink-2 ${
        right ? "text-right" : "text-left"
      }`}
      aria-sort={sortKey === key ? (desc ? "descending" : "ascending") : "none"}
    >
      {label}
      {sortKey === key && <span className="ml-1">{desc ? "▾" : "▴"}</span>}
    </th>
  );

  return (
    <div className="rounded-lg border border-hairline bg-surface">
      <div className="flex items-baseline gap-2 px-4 pb-1 pt-3">
        <h3 className="text-[10.5px] font-semibold uppercase tracking-[0.08em] text-muted">
          cost by node
        </h3>
        <span className="text-[11px] text-muted">
          — which step of the pipeline is eating the budget
        </span>
      </div>
      {rows.length === 0 ? (
        <div className="flex h-24 items-center justify-center text-[11.5px] text-muted">
          no spans recorded
        </div>
      ) : (
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-hairline">
              {header("key", "node", false)}
              {header("call_count", "calls")}
              {header("total_tokens", "tokens")}
              {header("avg", "avg $/call")}
              {header("cost", "total $")}
              <th className="w-40 px-3 py-2 text-left text-[10.5px] font-semibold uppercase tracking-[0.08em] text-muted">
                share
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => {
              const cost = Number(e.cost_usd);
              return (
                <tr
                  key={e.key}
                  className="border-b border-hairline/50 last:border-b-0 hover:bg-hover/60"
                >
                  <td className="max-w-64 truncate px-3 py-1.5 text-[12px] text-ink">
                    {e.key}
                  </td>
                  <td className="num px-3 py-1.5 text-right text-[12px] text-ink-2">
                    {e.call_count}
                  </td>
                  <td className="num px-3 py-1.5 text-right text-[12px] text-ink-2">
                    {fmtTokens(e.total_tokens)}
                  </td>
                  <td className="num px-3 py-1.5 text-right text-[12px] text-ink-2">
                    {fmtUSD(e.avg_cost_per_call)}
                  </td>
                  <td className="num px-3 py-1.5 text-right text-[12px] text-ink">
                    {fmtUSD(cost)}
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-grid">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${maxCost > 0 ? (cost / maxCost) * 100 : 0}%`,
                            background: COST_BAR_COLOR,
                          }}
                        />
                      </div>
                      <span className="num w-11 shrink-0 text-right text-[10.5px] text-muted">
                        {totalCost > 0 ? fmtPercent(cost / totalCost) : "—"}
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
