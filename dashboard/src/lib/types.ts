/** Mirrors the Pydantic response models of the tokenlens API (Part 5).
 *
 * Money fields typed `string` are exact decimal strings (e.g. "0.000125")
 * — parse with Number() for display only, never for arithmetic you care
 * about.
 */

export type SpanKind =
  | "LLM_CALL"
  | "CHAIN"
  | "GRAPH_NODE"
  | "TOOL"
  | "RETRIEVER"
  | "EMBEDDING"
  | "RETRY"
  | "CUSTOM";

export type SpanStatus = "OK" | "ERROR";

export interface TraceSummary {
  trace_id: string;
  root_name: string;
  started_at: string;
  total_cost_usd: number;
  total_tokens: number;
  user_id: string | null;
  feature_tag: string | null;
  session_id: string | null;
  has_error: boolean;
}

export interface TracePage {
  traces: TraceSummary[];
  count: number;
  limit: number;
  offset: number;
}

export interface SpanNode {
  span_id: string;
  trace_id: string;
  parent_span_id: string | null;
  name: string;
  kind: SpanKind;
  start_time: string;
  end_time: string | null;
  status: SpanStatus;
  error_message: string | null;
  metadata: Record<string, unknown>;
  model_name: string | null;
  provider: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  prompt_preview: string | null;
  retry_index: number;
  cost_usd: number | null;
  user_id: string | null;
  feature_tag: string | null;
  session_id: string | null;
  duration_ms: number | null;
  total_tokens: number | null;
  children: SpanNode[];
}

export interface AggregateEntry {
  key: string;
  cost_usd: string;
  total_tokens: number;
  call_count: number;
  avg_cost_per_call: string;
}

export interface Overview {
  total_cost_usd: string;
  total_tokens: number;
  trace_count: number;
  error_rate: number;
  retry_waste_usd: string;
  top_traces: TraceSummary[];
}

export interface TraceFilters {
  user_id?: string;
  feature_tag?: string;
  model?: string;
  min_cost?: number;
}
