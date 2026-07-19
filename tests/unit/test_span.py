from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from tokenlens.core.span import PROMPT_PREVIEW_LIMIT, Span, SpanKind, Trace


def test_duration_ms_is_none_while_open() -> None:
    s = Span(name="call")
    assert s.duration_ms is None


def test_duration_ms_computed_once_closed() -> None:
    start = datetime.now(UTC)
    s = Span(name="call", start_time=start)
    s.end_time = start + timedelta(milliseconds=250)
    assert s.duration_ms == pytest.approx(250)


def test_total_tokens_sums_present_fields() -> None:
    s = Span(name="call", input_tokens=10, output_tokens=5)
    assert s.total_tokens == 15


def test_total_tokens_none_when_nothing_set() -> None:
    s = Span(name="call")
    assert s.total_tokens is None


def test_prompt_preview_is_truncated() -> None:
    s = Span(name="call", prompt_preview="x" * 500)
    assert s.prompt_preview is not None
    assert len(s.prompt_preview) == PROMPT_PREVIEW_LIMIT + 1  # + ellipsis marker
    assert s.prompt_preview.endswith("…")


def test_prompt_preview_short_text_untouched() -> None:
    s = Span(name="call", prompt_preview="hello")
    assert s.prompt_preview == "hello"


def test_trace_to_tree_and_aggregation() -> None:
    root = Span(name="root", kind=SpanKind.CHAIN, trace_id="t1", cost_usd=0.01)
    child = Span(
        name="child",
        kind=SpanKind.LLM_CALL,
        trace_id="t1",
        parent_span_id=root.span_id,
        cost_usd=0.02,
        input_tokens=10,
        output_tokens=5,
    )
    trace = Trace.from_spans("t1", [root, child])

    tree = trace.to_tree()
    assert tree["name"] == "root"
    assert len(tree["children"]) == 1
    assert tree["children"][0]["name"] == "child"
    assert tree["children"][0]["children"] == []

    assert trace.total_cost() == pytest.approx(0.03)
    assert trace.total_tokens() == 15


def test_trace_requires_exactly_one_root() -> None:
    a = Span(name="a", trace_id="t1")
    b = Span(name="b", trace_id="t1")
    with pytest.raises(ValueError):
        Trace.from_spans("t1", [a, b])


def test_span_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Span(name="call", not_a_real_field="oops")  # type: ignore[call-arg]
