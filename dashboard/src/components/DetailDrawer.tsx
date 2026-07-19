import { useEffect, useState } from "react";
import { KIND_COLORS, KIND_LABELS } from "../lib/colors";
import { fmtClock, fmtDuration, fmtTokens, fmtUSD } from "../lib/format";
import type { SpanNode } from "../lib/types";

/** Right-hand span inspector: full metadata, prompt preview, copyable JSON. */
export function DetailDrawer({
  span,
  onClose,
}: {
  span: SpanNode;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => setCopied(false), [span.span_id]);

  const copyJson = () => {
    const { children: _children, ...bare } = span;
    void navigator.clipboard
      .writeText(JSON.stringify(bare, null, 2))
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1_500);
      });
  };

  const hasTokens =
    span.input_tokens !== null ||
    span.output_tokens !== null ||
    span.cache_read_tokens !== null ||
    span.cache_write_tokens !== null;

  return (
    <aside className="flex w-[400px] shrink-0 flex-col border-l border-hairline bg-surface">
      <header className="flex items-center gap-2.5 border-b border-hairline px-4 py-3">
        <span
          className="h-3 w-3 shrink-0 rounded-[3px]"
          style={{ background: KIND_COLORS[span.kind] }}
        />
        <div className="min-w-0 flex-1">
          <div className="truncate font-medium">{span.name}</div>
          <div className="text-[11px] text-muted">
            {KIND_LABELS[span.kind]}
            {span.retry_index > 0 && (
              <span className="text-retry"> · retry #{span.retry_index}</span>
            )}
          </div>
        </div>
        <button
          onClick={copyJson}
          className="rounded-md border border-hairline px-2 py-1 text-[11px] text-ink-2 hover:bg-hover"
        >
          {copied ? "copied ✓" : "copy JSON"}
        </button>
        <button
          onClick={onClose}
          aria-label="Close"
          className="rounded-md px-2 py-1 text-muted hover:bg-hover hover:text-ink"
        >
          ✕
        </button>
      </header>

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-3">
        {span.status === "ERROR" && (
          <section className="rounded-md border border-retry/50 bg-retry/10 px-3 py-2">
            <div className="text-[11px] font-medium tracking-wide text-error-ring">
              ERROR
            </div>
            <div className="mt-0.5 text-[12px] text-ink-2">
              {span.error_message ?? "span failed"}
            </div>
          </section>
        )}

        <Section title="call">
          <Row label="model" value={span.model_name ?? "—"} mono />
          <Row label="provider" value={span.provider ?? "—"} />
          <Row label="cost" value={fmtUSD(span.cost_usd)} mono />
          <Row label="duration" value={fmtDuration(span.duration_ms)} mono />
          <Row label="started" value={fmtClock(span.start_time)} mono />
        </Section>

        {hasTokens && (
          <Section title="tokens">
            <div className="grid grid-cols-2 gap-1.5">
              <TokenTile label="input" value={span.input_tokens} />
              <TokenTile label="output" value={span.output_tokens} />
              <TokenTile label="cache read" value={span.cache_read_tokens} />
              <TokenTile label="cache write" value={span.cache_write_tokens} />
            </div>
          </Section>
        )}

        <Section title="attribution">
          <Row label="user" value={span.user_id ?? "—"} />
          <Row label="feature" value={span.feature_tag ?? "—"} />
          <Row label="session" value={span.session_id ?? "—"} mono />
          <Row label="span id" value={span.span_id} mono />
        </Section>

        {span.prompt_preview && (
          <Section title="prompt preview">
            <pre className="num overflow-x-auto whitespace-pre-wrap rounded-md border border-hairline bg-page px-3 py-2 text-[11.5px] leading-relaxed text-ink-2">
              {span.prompt_preview}
            </pre>
          </Section>
        )}

        {Object.keys(span.metadata).length > 0 && (
          <Section title="metadata">
            <pre className="num overflow-x-auto rounded-md border border-hairline bg-page px-3 py-2 text-[11.5px] leading-relaxed text-ink-2">
              {JSON.stringify(span.metadata, null, 2)}
            </pre>
          </Section>
        )}
      </div>
    </aside>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="mb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-muted">
        {title}
      </h3>
      {children}
    </section>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5">
      <span className="shrink-0 text-[12px] text-muted">{label}</span>
      <span className={`truncate text-[12px] text-ink-2 ${mono ? "num" : ""}`}>
        {value}
      </span>
    </div>
  );
}

function TokenTile({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="rounded-md border border-hairline bg-raised px-2.5 py-1.5">
      <div className="text-[10.5px] text-muted">{label}</div>
      <div className="num text-[13px] text-ink">
        {value === null ? "—" : fmtTokens(value)}
      </div>
    </div>
  );
}
