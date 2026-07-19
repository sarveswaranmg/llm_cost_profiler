from decimal import Decimal

import pytest

from tokenlens.core.span import Span, SpanKind, Trace
from tokenlens.cost.attribution import cost_by, retry_waste, trace_flame_data


def _trace() -> Trace:
    """root chain → llm_a (gpt-4o, user-1) + llm_b retry pair (haiku, user-2)."""
    root = Span(name="pipeline", kind=SpanKind.CHAIN, trace_id="t1", span_id="root")
    llm_a = Span(
        name="answer",
        kind=SpanKind.LLM_CALL,
        trace_id="t1",
        span_id="a",
        parent_span_id="root",
        model_name="gpt-4o",
        user_id="user-1",
        input_tokens=1000,
        output_tokens=500,
        cost_usd=0.0075,
    )
    llm_b_fail = Span(
        name="condense",
        kind=SpanKind.LLM_CALL,
        trace_id="t1",
        span_id="b0",
        parent_span_id="root",
        model_name="claude-haiku-4-5",
        user_id="user-2",
        input_tokens=100,
        output_tokens=0,
        retry_index=0,
        cost_usd=0.0001,
    )
    llm_b_retry = Span(
        name="condense",
        kind=SpanKind.LLM_CALL,
        trace_id="t1",
        span_id="b1",
        parent_span_id="root",
        model_name="claude-haiku-4-5",
        user_id="user-2",
        input_tokens=100,
        output_tokens=40,
        retry_index=1,
        cost_usd=0.0003,
    )
    return Trace.from_spans("t1", [root, llm_a, llm_b_fail, llm_b_retry])


def test_cost_by_model_name_sorted_desc_with_avg() -> None:
    entries = cost_by([_trace()], "model_name")
    assert [e.key for e in entries] == ["gpt-4o", "claude-haiku-4-5"]

    gpt = entries[0]
    assert gpt.cost_usd == Decimal("0.0075")
    assert gpt.total_tokens == 1500
    assert gpt.call_count == 1
    assert gpt.avg_cost_per_call == Decimal("0.007500")

    haiku = entries[1]
    assert haiku.cost_usd == Decimal("0.0004")
    assert haiku.call_count == 2
    assert haiku.avg_cost_per_call == Decimal("0.000200")


def test_cost_by_user_id_skips_unattributed_spans() -> None:
    entries = cost_by([_trace()], "user_id")
    assert {e.key for e in entries} == {"user-1", "user-2"}  # root chain span skipped


def test_cost_by_kind_and_node_name() -> None:
    by_kind = {e.key: e for e in cost_by([_trace()], "kind")}
    assert by_kind["LLM_CALL"].call_count == 3
    assert by_kind["CHAIN"].cost_usd == Decimal(0)

    by_node = {e.key: e for e in cost_by([_trace()], "node_name")}
    assert by_node["condense"].call_count == 2


def test_cost_by_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown cost_by key"):
        cost_by([_trace()], "nonsense")


def test_retry_waste_counts_only_retry_spans() -> None:
    # Only the retry_index=1 span counts — not the failed first attempt.
    assert retry_waste([_trace()]) == Decimal("0.0003")


def test_retry_waste_empty() -> None:
    assert retry_waste([]) == Decimal(0)


def test_trace_flame_data_shape_and_micro_dollar_values() -> None:
    flame = trace_flame_data(_trace())

    assert flame["name"] == "pipeline"
    assert [c["name"] for c in flame["children"]] == ["answer", "condense", "condense"]

    answer = flame["children"][0]
    assert answer["value"] == 7500  # 0.0075 USD in micro-dollars
    assert answer["children"] == []
    assert answer["data"]["model"] == "gpt-4o"
    assert answer["data"]["tokens"] == 1500
    assert answer["data"]["kind"] == "LLM_CALL"

    # Parent value is inclusive: 7500 + 100 + 300 (root itself has no cost).
    assert flame["value"] == 7900
    assert flame["data"]["kind"] == "CHAIN"

    retry = flame["children"][2]
    assert retry["data"]["retry_index"] == 1
    assert retry["value"] == 300
