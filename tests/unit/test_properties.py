"""Property-based tests (hypothesis) over randomly generated span trees.

Trees are generated as a parent-index list (span i>0 attaches to a random
earlier span), which yields every possible tree shape without recursive
strategies. Costs are drawn in integer micro-dollars and converted to the
same float representation the pricing enricher produces.
"""

from decimal import Decimal
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tokenlens.core.span import Span, SpanKind, Trace
from tokenlens.cost.attribution import trace_flame_data

_settings = settings(
    max_examples=50,
    deadline=None,
    # The autouse context-isolation fixture is function-scoped; it resets
    # global state that these pure-data tests never touch, so sharing it
    # across hypothesis examples is safe.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_maybe_tokens = st.one_of(st.none(), st.integers(min_value=0, max_value=1_000_000))
_maybe_micro_cost = st.one_of(st.none(), st.integers(min_value=0, max_value=10_000_000))


@st.composite
def span_trees(draw: Any) -> Trace:
    n = draw(st.integers(min_value=1, max_value=25))
    parent_idx = [None] + [draw(st.integers(0, i - 1)) for i in range(1, n)]

    spans: list[Span] = []
    for i in range(n):
        micro = draw(_maybe_micro_cost)
        spans.append(
            Span(
                trace_id="prop-trace",
                parent_span_id=None if parent_idx[i] is None else spans[parent_idx[i]].span_id,
                name=f"span-{i}",
                kind=draw(st.sampled_from(list(SpanKind))),
                input_tokens=draw(_maybe_tokens),
                output_tokens=draw(_maybe_tokens),
                cache_read_tokens=draw(_maybe_tokens),
                # float(Decimal(micro) / 1e6) is exactly what enrich_span_cost
                # stores, so Decimal(str(cost_usd)) round-trips losslessly.
                cost_usd=None if micro is None else float(Decimal(micro) / Decimal(1_000_000)),
                retry_index=draw(st.integers(0, 2)),
            )
        )
    return Trace.from_spans("prop-trace", spans)


@_settings
@given(trace=span_trees())
def test_total_cost_and_tokens_equal_sum_over_all_spans(trace: Trace) -> None:
    # Summed in span order, exactly as Trace.total_cost does — equality is
    # exact, not approximate.
    assert trace.total_cost() == sum(s.cost_usd or 0.0 for s in trace.spans)
    assert trace.total_tokens() == sum(s.total_tokens or 0 for s in trace.spans)


@_settings
@given(trace=span_trees())
def test_to_tree_round_trips_every_span_without_loss(trace: Trace) -> None:
    tree = trace.to_tree()

    flattened: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        flattened.append(node)
        for child in node["children"]:
            assert child["parent_span_id"] == node["span_id"]
            walk(child)

    walk(tree)

    by_id = {s.span_id: s for s in trace.spans}
    assert len(flattened) == len(trace.spans)
    assert {n["span_id"] for n in flattened} == set(by_id)
    for node in flattened:
        original = by_id[node["span_id"]].model_dump(mode="json")
        node_without_children = {k: v for k, v in node.items() if k != "children"}
        assert node_without_children == original


@_settings
@given(trace=span_trees())
def test_flame_data_parent_value_covers_children_and_is_integer_micros(trace: Trace) -> None:
    flame = trace_flame_data(trace)

    total_micro = 0

    def walk(node: dict[str, Any]) -> None:
        nonlocal total_micro
        assert isinstance(node["value"], int)
        assert node["value"] >= 0
        child_sum = sum(child["value"] for child in node["children"])
        assert node["value"] >= child_sum  # inclusive: a frame covers its children
        total_micro += node["value"] - child_sum  # self weight
        for child in node["children"]:
            walk(child)

    walk(flame)

    expected_micro = sum(
        int(Decimal(str(s.cost_usd)) * 1_000_000) for s in trace.spans if s.cost_usd is not None
    )
    assert flame["value"] == expected_micro
    assert total_micro == expected_micro
