"""Programmatic trace builders with hand-computed expected costs.

sample_trace() builds the canonical 6-span tree used across the suite:

    root CHAIN "handle_request"
    ├── RETRIEVER "vector_search"          (no tokens, no cost)
    ├── LLM_CALL  "draft_answer"           gpt-4o-mini      1000 in / 500 out
    ├── LLM_CALL  "draft_answer" (retry 1) gpt-4o-mini      1000 in /   0 out
    └── TOOL      "web_search"
        └── LLM_CALL "summarize_results"   claude-haiku-4-5 2000 in / 400 out
                                                            + 1000 cache-read

Expected costs are hand-computed below from the bundled pricing table
(USD per 1M tokens) so assertions never depend on the code under test:

    gpt-4o-mini       input 0.15 / output 0.60
    claude-haiku-4-5  input 1.00 / output 5.00 / cache-read 0.10

    draft:     1000*0.15/1e6 + 500*0.60/1e6            = 0.000450
    retry:     1000*0.15/1e6                           = 0.000150
    summarize: 2000*1.00/1e6 + 400*5.00/1e6
               + 1000*0.10/1e6                         = 0.004100
    total                                              = 0.004700
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tokenlens.core.span import Span, SpanKind, Trace

SAMPLE_BASE_TIME = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)

EXPECTED_DRAFT_COST = Decimal("0.000450")
EXPECTED_RETRY_COST = Decimal("0.000150")
EXPECTED_SUMMARIZE_COST = Decimal("0.004100")
SAMPLE_TRACE_COST = EXPECTED_DRAFT_COST + EXPECTED_RETRY_COST + EXPECTED_SUMMARIZE_COST
SAMPLE_TRACE_TOKENS = 1500 + 1000 + 3400  # draft + retry + summarize
SAMPLE_TRACE_SPAN_COUNT = 6


def sample_trace(
    trace_id: str = "sample-trace",
    started_at: datetime = SAMPLE_BASE_TIME,
    *,
    user_id: str | None = "u-42",
    feature_tag: str | None = "chat",
    session_id: str | None = "sess-1",
    with_costs: bool = True,
) -> Trace:
    """Build the canonical sample trace.

    with_costs=True stamps the hand-computed costs onto the LLM spans (what
    the pricing enricher would produce); with_costs=False leaves cost_usd
    None so tests can exercise enrichment itself.
    """
    attribution = {"user_id": user_id, "feature_tag": feature_tag, "session_id": session_id}
    at = started_at

    root = Span(
        trace_id=trace_id,
        name="handle_request",
        kind=SpanKind.CHAIN,
        start_time=at,
        end_time=at + timedelta(seconds=4),
        metadata={"env": "test"},
        **attribution,
    )
    retriever = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="vector_search",
        kind=SpanKind.RETRIEVER,
        start_time=at + timedelta(milliseconds=10),
        end_time=at + timedelta(milliseconds=210),
        metadata={"k": 4},
        **attribution,
    )
    draft = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="draft_answer",
        kind=SpanKind.LLM_CALL,
        start_time=at + timedelta(milliseconds=250),
        end_time=at + timedelta(milliseconds=1250),
        model_name="gpt-4o-mini",
        provider="openai",
        input_tokens=1000,
        output_tokens=500,
        cost_usd=float(EXPECTED_DRAFT_COST) if with_costs else None,
        prompt_preview="Draft an answer from the retrieved context…",
        **attribution,
    )
    retry = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="draft_answer",
        kind=SpanKind.LLM_CALL,
        start_time=at + timedelta(milliseconds=1300),
        end_time=at + timedelta(milliseconds=1500),
        model_name="gpt-4o-mini",
        provider="openai",
        input_tokens=1000,
        output_tokens=0,
        retry_index=1,
        cost_usd=float(EXPECTED_RETRY_COST) if with_costs else None,
        **attribution,
    )
    tool = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="web_search",
        kind=SpanKind.TOOL,
        start_time=at + timedelta(milliseconds=1600),
        end_time=at + timedelta(seconds=3, milliseconds=900),
        metadata={"query": "tokenlens"},
        **attribution,
    )
    summarize = Span(
        trace_id=trace_id,
        parent_span_id=tool.span_id,
        name="summarize_results",
        kind=SpanKind.LLM_CALL,
        start_time=at + timedelta(seconds=2),
        end_time=at + timedelta(seconds=3, milliseconds=800),
        model_name="claude-haiku-4-5",
        provider="anthropic",
        input_tokens=2000,
        output_tokens=400,
        cache_read_tokens=1000,
        cost_usd=float(EXPECTED_SUMMARIZE_COST) if with_costs else None,
        **attribution,
    )
    return Trace.from_spans(trace_id, [root, retriever, draft, retry, tool, summarize])
