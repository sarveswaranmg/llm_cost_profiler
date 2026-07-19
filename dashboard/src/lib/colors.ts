import type { SpanKind } from "./types";
import type { CostTier } from "./format";

/** Span-kind fills. The four chromatic hues (LLM/tool/retriever/retry) are
 * a validated all-pairs set on the dark surface; structural kinds are
 * neutral grays so they recede; embedding is a light step of the retriever
 * hue (same family). Retries are additionally striped (never color alone).
 */
export const KIND_COLORS: Record<SpanKind, string> = {
  LLM_CALL: "#256abf",
  TOOL: "#9085e9",
  RETRIEVER: "#1baf7a",
  EMBEDDING: "#4cc39a",
  RETRY: "#d03b3b",
  CHAIN: "#454541",
  GRAPH_NODE: "#605f58",
  CUSTOM: "#77766e",
};

export const KIND_LABELS: Record<SpanKind, string> = {
  LLM_CALL: "llm",
  TOOL: "tool",
  RETRIEVER: "retriever",
  EMBEDDING: "embedding",
  RETRY: "retry",
  CHAIN: "chain",
  GRAPH_NODE: "node",
  CUSTOM: "custom",
};

export const TIER_COLORS: Record<CostTier, string> = {
  good: "#0ca30c",
  warn: "#fab219",
  crit: "#d03b3b",
};

/** Single hue for all cost bar charts — same measure, same color. */
export const COST_BAR_COLOR = "#3987e5";

export const ERROR_RING = "#ff9a9a";
