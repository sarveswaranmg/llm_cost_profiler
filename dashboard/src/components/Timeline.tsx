import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import { ERROR_RING, KIND_COLORS } from "../lib/colors";
import { fmtDuration } from "../lib/format";
import { useContainerWidth } from "../lib/hooks";
import type { SpanNode } from "../lib/types";
import { SpanTooltip, type HoverState } from "./Tooltip";

const ROW_H = 26;
const LABEL_W = 208;
const INDENT = 12;

interface FlatSpan {
  span: SpanNode;
  depth: number;
  startMs: number;
  endMs: number;
}

/** Gantt-style waterfall: x = wall-clock time. Rows in pre-order (a parent
 * directly above its children), which reveals both sequencing and
 * parallelism at a glance. */
export function Timeline({
  root,
  selectedId,
  onSelect,
}: {
  root: SpanNode;
  selectedId: string | null;
  onSelect: (span: SpanNode) => void;
}) {
  const [hover, setHover] = useState<HoverState | null>(null);
  const [containerRef, width] = useContainerWidth<HTMLDivElement>();

  const { rows, domainMs } = useMemo(() => {
    const t0 = new Date(root.start_time).getTime();
    const rows: FlatSpan[] = [];
    let maxEnd = 1;
    const walk = (span: SpanNode, depth: number) => {
      const start = new Date(span.start_time).getTime() - t0;
      const end = span.end_time ? new Date(span.end_time).getTime() - t0 : start;
      maxEnd = Math.max(maxEnd, end);
      rows.push({ span, depth, startMs: start, endMs: end });
      span.children.forEach((c) => walk(c, depth + 1));
    };
    walk(root, 0);
    return { rows, domainMs: maxEnd };
  }, [root]);

  const plotW = Math.max(width - LABEL_W - 16, 100);
  const x = useMemo(
    () => scaleLinear().domain([0, domainMs]).range([0, plotW]),
    [domainMs, plotW],
  );
  const ticks = x.ticks(Math.max(3, Math.min(8, Math.floor(plotW / 110))));

  return (
    <div className="flex h-full flex-col">
      <div ref={containerRef} className="flex-1 overflow-y-auto p-4 pt-2">
        {/* time axis */}
        <div className="sticky top-0 z-10 flex bg-page/95 pb-1 backdrop-blur-sm">
          <div className="w-52 shrink-0" />
          <div className="relative h-5 flex-1">
            {ticks.map((t) => (
              <span
                key={t}
                className="num absolute top-0 -translate-x-1/2 text-[10px] text-muted"
                style={{ left: x(t) }}
              >
                {t === 0 ? "0" : fmtDuration(t)}
              </span>
            ))}
          </div>
        </div>

        <div className="relative">
          {/* gridlines spanning all rows */}
          <div
            className="pointer-events-none absolute inset-y-0"
            style={{ left: LABEL_W, width: plotW }}
          >
            {ticks.map((t) => (
              <span
                key={t}
                className="absolute inset-y-0 w-px bg-grid"
                style={{ left: x(t) }}
              />
            ))}
          </div>

          {rows.map(({ span, depth, startMs, endMs }) => {
            const isRetry = span.kind === "RETRY";
            const isError = span.status === "ERROR";
            const selected = span.span_id === selectedId;
            const barX = x(startMs);
            const barW = Math.max(x(endMs) - x(startMs), 2);
            const showRightLabel = barX + barW + 8 < plotW - 60;
            return (
              <button
                key={span.span_id}
                onClick={() => onSelect(span)}
                onMouseMove={(e) =>
                  setHover({ span, x: e.clientX, y: e.clientY })
                }
                onMouseLeave={() => setHover(null)}
                className={`group flex w-full items-center rounded-sm text-left ${
                  selected ? "bg-raised" : "hover:bg-hover/60"
                }`}
                style={{ height: ROW_H }}
              >
                <div
                  className="flex w-52 shrink-0 items-center gap-1.5 pr-2"
                  style={{ paddingLeft: depth * INDENT }}
                >
                  <span
                    className={`h-2 w-2 shrink-0 rounded-xs ${isRetry ? "retry-stripes" : ""}`}
                    style={isRetry ? undefined : { background: KIND_COLORS[span.kind] }}
                  />
                  <span
                    className={`truncate text-[11.5px] ${
                      selected ? "text-ink" : "text-ink-2"
                    }`}
                  >
                    {span.name}
                  </span>
                  {isError && (
                    <span className="shrink-0 text-[10px] text-error-ring">⚠</span>
                  )}
                </div>
                <div className="relative h-full flex-1">
                  <span
                    className={`absolute top-1/2 h-3.5 -translate-y-1/2 rounded-[3px] ${
                      isRetry ? "retry-stripes" : ""
                    }`}
                    style={{
                      left: barX,
                      width: barW,
                      background: isRetry ? undefined : KIND_COLORS[span.kind],
                      boxShadow: isError
                        ? `0 0 0 1px ${ERROR_RING}`
                        : selected
                          ? "0 0 0 1px #f4f4f0"
                          : undefined,
                    }}
                  />
                  {showRightLabel && (
                    <span
                      className="num absolute top-1/2 -translate-y-1/2 text-[10px] text-muted opacity-0 transition-opacity group-hover:opacity-100"
                      style={{ left: barX + barW + 6 }}
                    >
                      {fmtDuration(endMs - startMs)}
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {hover && <SpanTooltip hover={hover} />}
    </div>
  );
}
