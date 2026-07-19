"""The sample_trace fixture's hand-computed costs, verified against the
real pricing engine and the attribution/flame code paths.

This is the tie between the arithmetic written down in tests/factories.py
and what tokenlens actually computes — if the bundled pricing table or the
cost math changes, this file fails first.
"""

from decimal import Decimal

from tests.factories import (
    EXPECTED_DRAFT_COST,
    EXPECTED_RETRY_COST,
    EXPECTED_SUMMARIZE_COST,
    SAMPLE_TRACE_COST,
    SAMPLE_TRACE_SPAN_COUNT,
    SAMPLE_TRACE_TOKENS,
)

from tokenlens.core.span import SpanKind
from tokenlens.cost.attribution import cost_by, retry_waste, trace_flame_data
from tokenlens.cost.pricing import enrich_span_cost


def test_pricing_enricher_reproduces_hand_computed_costs(sample_trace) -> None:
    trace = sample_trace(with_costs=False)

    for span in trace.spans:
        enrich_span_cost(span)

    by_name = {(s.name, s.retry_index): s for s in trace.spans}
    assert Decimal(str(by_name[("draft_answer", 0)].cost_usd)) == EXPECTED_DRAFT_COST
    assert Decimal(str(by_name[("draft_answer", 1)].cost_usd)) == EXPECTED_RETRY_COST
    assert Decimal(str(by_name[("summarize_results", 0)].cost_usd)) == EXPECTED_SUMMARIZE_COST
    # Non-LLM spans stay unpriced.
    assert by_name[("vector_search", 0)].cost_usd is None
    assert by_name[("web_search", 0)].cost_usd is None


def test_sample_trace_totals_match_hand_computed_constants(sample_trace) -> None:
    trace = sample_trace()

    assert len(trace.spans) == SAMPLE_TRACE_SPAN_COUNT
    assert trace.total_tokens() == SAMPLE_TRACE_TOKENS
    total = sum(
        (Decimal(str(s.cost_usd)) for s in trace.spans if s.cost_usd is not None), Decimal(0)
    )
    assert total == SAMPLE_TRACE_COST


def test_retry_waste_counts_only_the_retry_span(sample_trace) -> None:
    assert retry_waste([sample_trace()]) == EXPECTED_RETRY_COST

    # A trace with no retry_index > 0 spans wastes exactly Decimal 0, even
    # though its spans carry real (non-retry) costs.
    no_retries = sample_trace(trace_id="clean")
    for span in no_retries.spans:
        span.retry_index = 0
    assert retry_waste([no_retries]) == Decimal(0)


def test_cost_by_model_matches_hand_computed_split(sample_trace) -> None:
    entries = {e.key: e for e in cost_by([sample_trace()], "model_name")}

    assert set(entries) == {"gpt-4o-mini", "claude-haiku-4-5"}
    assert entries["gpt-4o-mini"].cost_usd == EXPECTED_DRAFT_COST + EXPECTED_RETRY_COST
    assert entries["gpt-4o-mini"].call_count == 2
    assert entries["gpt-4o-mini"].total_tokens == 2500
    assert entries["claude-haiku-4-5"].cost_usd == EXPECTED_SUMMARIZE_COST
    assert entries["claude-haiku-4-5"].call_count == 1
    assert entries["claude-haiku-4-5"].total_tokens == 3400


def test_flame_data_root_is_total_cost_in_micro_dollars(sample_trace) -> None:
    flame = trace_flame_data(sample_trace())

    assert flame["name"] == "handle_request"
    assert flame["value"] == int(SAMPLE_TRACE_COST * 1_000_000)  # 4700 µ$

    tool_node = next(c for c in flame["children"] if c["name"] == "web_search")
    # The TOOL span itself costs nothing; its value is its LLM child's.
    assert tool_node["value"] == int(EXPECTED_SUMMARIZE_COST * 1_000_000)
    assert tool_node["children"][0]["name"] == "summarize_results"
    retry_node = next(c for c in flame["children"] if c["data"]["retry_index"] == 1)
    assert retry_node["value"] == int(EXPECTED_RETRY_COST * 1_000_000)
    assert retry_node["data"]["kind"] == SpanKind.LLM_CALL.value
