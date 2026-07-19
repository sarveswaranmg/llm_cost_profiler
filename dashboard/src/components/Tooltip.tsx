import { KIND_COLORS, KIND_LABELS } from "../lib/colors";
import { fmtDuration, fmtTokens, fmtUSD } from "../lib/format";
import type { SpanNode } from "../lib/types";

export interface HoverState {
  span: SpanNode;
  x: number;
  y: number;
}

const WIDTH = 264;

/** Shared hover card for flamegraph + timeline blocks. */
export function SpanTooltip({ hover }: { hover: HoverState }) {
  const { span } = hover;
  const left = Math.min(hover.x + 14, window.innerWidth - WIDTH - 12);
  const top = Math.min(hover.y + 14, window.innerHeight - 190);
  return (
    <div
      className="pointer-events-none fixed z-50 rounded-md border border-hairline bg-raised px-3 py-2.5 shadow-[0_8px_24px_rgba(0,0,0,0.55)]"
      style={{ left, top, width: WIDTH }}
    >
      <div className="mb-1.5 flex items-center gap-2">
        <span
          className="inline-block h-2.5 w-2.5 shrink-0 rounded-[3px]"
          style={{ background: KIND_COLORS[span.kind] }}
        />
        <span className="truncate font-medium text-ink">{span.name}</span>
      </div>
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11.5px]">
        <Cell label="kind" value={KIND_LABELS[span.kind]} />
        {span.model_name && <Cell label="model" value={span.model_name} mono />}
        <Cell label="cost" value={fmtUSD(span.cost_usd)} mono />
        {(span.input_tokens !== null || span.output_tokens !== null) && (
          <Cell
            label="tokens"
            value={`${fmtTokens(span.input_tokens)} in · ${fmtTokens(span.output_tokens)} out`}
            mono
          />
        )}
        {(span.cache_read_tokens ?? 0) > 0 && (
          <Cell label="cache" value={`${fmtTokens(span.cache_read_tokens)} read`} mono />
        )}
        <Cell label="duration" value={fmtDuration(span.duration_ms)} mono />
        {span.retry_index > 0 && (
          <Cell label="retry" value={`attempt #${span.retry_index + 1}`} alarm />
        )}
        {span.status === "ERROR" && (
          <Cell label="error" value={span.error_message ?? "failed"} alarm />
        )}
      </div>
    </div>
  );
}

function Cell({
  label,
  value,
  mono,
  alarm,
}: {
  label: string;
  value: string;
  mono?: boolean;
  alarm?: boolean;
}) {
  return (
    <>
      <span className="text-muted">{label}</span>
      <span
        className={`truncate ${mono ? "num" : ""} ${alarm ? "text-error-ring" : "text-ink-2"}`}
      >
        {value}
      </span>
    </>
  );
}
