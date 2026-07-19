"""Shared builders for the storage backend tests (SQLite + Postgres)."""

from datetime import timedelta

from tokenlens.core.span import Span, SpanKind, SpanStatus, Trace
from tokenlens.storage.base import TraceFilters


def build_pipeline_trace(
    trace_id,
    started_at,
    *,
    user_id="u-1",
    feature_tag="chat",
    session_id="sess-1",
    model_a="gpt-4o",
    model_b="claude-sonnet-5",
    cost_a=0.1,
    cost_b=0.2,
    retry_cost=0.000015,
):
    """A realistic 6-span tree: chain → (retriever, llm, retry, chain → llm).

    cost_a/cost_b default to 0.1/0.2 deliberately — their float sum is
    0.30000000000000004, so any backend summing raw floats instead of exact
    micro-dollars diverges from the Decimal-based attribution math.
    """
    attribution = {"user_id": user_id, "feature_tag": feature_tag, "session_id": session_id}
    at = started_at

    root = Span(
        trace_id=trace_id,
        name="pipeline",
        kind=SpanKind.CHAIN,
        start_time=at,
        end_time=at + timedelta(seconds=5),
        metadata={"env": "test", "nested": {"tags": ["a", "b"]}},
        **attribution,
    )
    retriever = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="retrieve_docs",
        kind=SpanKind.RETRIEVER,
        start_time=at + timedelta(milliseconds=100),
        end_time=at + timedelta(milliseconds=350),
        metadata={"k": 4},
        **attribution,
    )
    llm_a = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="draft_answer",
        kind=SpanKind.LLM_CALL,
        start_time=at + timedelta(milliseconds=400),
        end_time=at + timedelta(seconds=2),
        model_name=model_a,
        provider="openai",
        input_tokens=1200,
        output_tokens=300,
        cost_usd=cost_a,
        prompt_preview="Summarize the retrieved documents…",
        **attribution,
    )
    retry = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="draft_answer",
        kind=SpanKind.RETRY,
        start_time=at + timedelta(seconds=2, milliseconds=100),
        end_time=at + timedelta(seconds=2, milliseconds=600),
        status=SpanStatus.ERROR,
        error_message="rate limited",
        model_name=model_a,
        provider="openai",
        input_tokens=1200,
        output_tokens=0,
        retry_index=1,
        cost_usd=retry_cost,
        **attribution,
    )
    refine = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="refine",
        kind=SpanKind.CHAIN,
        start_time=at + timedelta(seconds=3),
        end_time=at + timedelta(seconds=5),
        **attribution,
    )
    llm_b = Span(
        trace_id=trace_id,
        parent_span_id=refine.span_id,
        name="refine_answer",
        kind=SpanKind.LLM_CALL,
        start_time=at + timedelta(seconds=3, milliseconds=200),
        end_time=at + timedelta(seconds=4, milliseconds=800),
        model_name=model_b,
        provider="anthropic",
        input_tokens=800,
        output_tokens=150,
        cache_read_tokens=200,
        cost_usd=cost_b,
        **attribution,
    )
    return Trace.from_spans(trace_id, [root, retriever, llm_a, retry, refine, llm_b])


def matches_filters(trace: Trace, filters: TraceFilters) -> bool:
    """Pure-Python mirror of the SQL trace filters, for agreement tests."""
    root = trace.root
    if filters.user_id is not None and root.user_id != filters.user_id:
        return False
    if filters.feature_tag is not None and root.feature_tag != filters.feature_tag:
        return False
    if filters.model is not None and all(s.model_name != filters.model for s in trace.spans):
        return False
    if filters.since is not None and root.start_time < filters.since:
        return False
    if filters.until is not None and root.start_time > filters.until:
        return False
    return filters.min_cost is None or trace.total_cost() >= filters.min_cost
