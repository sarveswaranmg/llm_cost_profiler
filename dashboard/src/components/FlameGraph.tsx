import { hierarchy, partition, type HierarchyRectangularNode } from "d3-hierarchy";
import { useMemo, useState } from "react";
import { ERROR_RING, KIND_COLORS, KIND_LABELS } from "../lib/colors";
import { fmtDuration, fmtTokens, fmtUSD } from "../lib/format";
import { useContainerWidth } from "../lib/hooks";
import type { SpanKind, SpanNode } from "../lib/types";
import { SpanTooltip, type HoverState } from "./Tooltip";

export type WeightMode = "cost" | "duration" | "tokens";

const ROW_H = 26;
const ROW_GAP = 2;
const MIN_LABEL_W = 36;
const CHAR_W = 6.2;

/** Self-weight per node: the partition layout sums these down the tree, so
 * each must be the node's own contribution only. Durations are wall-clock
 * (a parent covers its children), so self-duration is the residual — and
 * parallel children that overflow the parent still get their full width. */
function selfWeight(span: SpanNode, mode: WeightMode): number {
  switch (mode) {
    case "cost":
      return span.cost_usd ?? 0;
    case "tokens":
      return span.total_tokens ?? 0;
    case "duration": {
      const own = span.duration_ms ?? 0;
      const children = span.children.reduce((s, c) => s + (c.duration_ms ?? 0), 0);
      return Math.max(0, own - children);
    }
  }
}

function totalWeight(span: SpanNode, mode: WeightMode): number {
  return (
    selfWeight(span, mode) + span.children.reduce((s, c) => s + totalWeight(c, mode), 0)
  );
}

export function FlameGraph({
  root,
  selectedId,
  onSelect,
}: {
  root: SpanNode;
  selectedId: string | null;
  onSelect: (span: SpanNode) => void;
}) {
  const [mode, setMode] = useState<WeightMode>("cost");
  const [hover, setHover] = useState<HoverState | null>(null);
  const [containerRef, width] = useContainerWidth<HTMLDivElement>();

  const layout = useMemo(() => {
    if (width === 0) return null;
    const total = totalWeight(root, mode);
    // A sliver floor so zero-weight structural spans stay visible/clickable.
    const epsilon = total > 0 ? total / 400 : 1;
    const h = hierarchy<SpanNode>(root, (d) => d.children).sum(
      (d) => selfWeight(d, mode) + epsilon,
    );
    const height = (h.height + 1) * ROW_H;
    return partition<SpanNode>().size([width, height])(h);
  }, [root, mode, width]);

  const kindsPresent = useMemo(() => {
    const present = new Set<SpanKind>();
    const walk = (s: SpanNode) => {
      present.add(s.kind);
      s.children.forEach(walk);
    };
    walk(root);
    return (Object.keys(KIND_COLORS) as SpanKind[]).filter((k) => present.has(k));
  }, [root]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-hairline px-4 py-2">
        <WeightToggle mode={mode} onMode={setMode} />
        <span className="text-[11px] text-muted">block width ∝ {mode}</span>
        <div className="ml-auto flex flex-wrap items-center gap-2.5">
          {kindsPresent.map((k) => (
            <span key={k} className="flex items-center gap-1 text-[10.5px] text-ink-2">
              <span
                className={`h-2 w-2 rounded-[2px] ${k === "RETRY" ? "retry-stripes" : ""}`}
                style={k === "RETRY" ? undefined : { background: KIND_COLORS[k] }}
              />
              {KIND_LABELS[k]}
            </span>
          ))}
        </div>
      </div>

      <div ref={containerRef} className="flex-1 overflow-y-auto p-4">
        {layout && (
          <svg
            width={width}
            height={(layout.height + 1) * ROW_H}
            className="block"
            role="img"
            aria-label={`Flamegraph of ${root.name}, weighted by ${mode}`}
          >
            <defs>
              <pattern
                id="flame-retry-stripes"
                patternUnits="userSpaceOnUse"
                width="6"
                height="6"
                patternTransform="rotate(45)"
              >
                <rect width="6" height="6" fill={KIND_COLORS.RETRY} />
                <rect width="2" height="6" fill="#932a2a" />
              </pattern>
            </defs>
            {layout.descendants().map((node) => (
              <FlameBlock
                key={node.data.span_id}
                node={node}
                mode={mode}
                selected={node.data.span_id === selectedId}
                onSelect={onSelect}
                onHover={setHover}
              />
            ))}
          </svg>
        )}
      </div>

      {hover && <SpanTooltip hover={hover} />}
    </div>
  );
}

function FlameBlock({
  node,
  mode,
  selected,
  onSelect,
  onHover,
}: {
  node: HierarchyRectangularNode<SpanNode>;
  mode: WeightMode;
  selected: boolean;
  onSelect: (span: SpanNode) => void;
  onHover: (h: HoverState | null) => void;
}) {
  const span = node.data;
  const x = node.x0;
  const w = Math.max(node.x1 - node.x0, 1);
  const y = node.depth * ROW_H;
  const h = ROW_H - ROW_GAP;
  const isRetry = span.kind === "RETRY";
  const isError = span.status === "ERROR";
  const structural =
    span.kind === "CHAIN" || span.kind === "GRAPH_NODE" || span.kind === "CUSTOM";

  const label = useMemo(() => {
    if (w < MIN_LABEL_W) return null;
    // Inclusive (subtree) value — a frame's width covers its children, so
    // its label should too. The tooltip shows the span's own numbers.
    const inclusive = totalWeight(span, mode);
    const value =
      mode === "cost"
        ? fmtUSD(inclusive)
        : mode === "tokens"
          ? `${fmtTokens(inclusive)} tok`
          : fmtDuration(span.duration_ms);
    const full = w >= 110 ? `${span.name} · ${value}` : span.name;
    const maxChars = Math.floor((w - 10) / CHAR_W);
    return full.length > maxChars ? `${full.slice(0, Math.max(1, maxChars - 1))}…` : full;
  }, [w, mode, span]);

  return (
    <g
      onMouseMove={(e) => onHover({ span, x: e.clientX, y: e.clientY })}
      onMouseLeave={() => onHover(null)}
      onClick={() => onSelect(span)}
      className="cursor-pointer"
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={2}
        fill={isRetry ? "url(#flame-retry-stripes)" : KIND_COLORS[span.kind]}
        stroke={
          selected ? "#f4f4f0" : isError ? ERROR_RING : "var(--color-surface)"
        }
        strokeWidth={selected ? 1.5 : 1}
        opacity={selected ? 1 : 0.96}
      />
      {label && (
        <text
          x={x + 5}
          y={y + h / 2 + 0.5}
          dominantBaseline="central"
          fontSize={10.5}
          className="pointer-events-none select-none"
          fill={structural ? "var(--color-ink-2)" : "#ffffff"}
        >
          {isError && !isRetry ? "⚠ " : ""}
          {label}
        </text>
      )}
    </g>
  );
}

function WeightToggle({
  mode,
  onMode,
}: {
  mode: WeightMode;
  onMode: (m: WeightMode) => void;
}) {
  const options: WeightMode[] = ["cost", "duration", "tokens"];
  return (
    <div
      role="radiogroup"
      aria-label="Block width weighting"
      className="flex overflow-hidden rounded-md border border-hairline"
    >
      {options.map((opt) => (
        <button
          key={opt}
          role="radio"
          aria-checked={mode === opt}
          onClick={() => onMode(opt)}
          className={`px-2.5 py-1 text-[11px] transition-colors ${
            mode === opt
              ? "bg-accent/20 font-medium text-ink"
              : "text-muted hover:bg-hover hover:text-ink-2"
          }`}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}
