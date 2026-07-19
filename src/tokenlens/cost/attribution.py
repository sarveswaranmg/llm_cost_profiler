"""Cost attribution and aggregation over collected traces.

Answers the questions the raw span tree can't at a glance: who spent the
money (cost_by), how much of it was burned on retries (retry_waste), and
what does one trace look like as a flamegraph (trace_flame_data).

Aggregation math uses Decimal (spans store cost as a float already
quantized to 6 decimal places, so Decimal(str(...)) round-trips exactly).
"""

from collections import defaultdict
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from tokenlens.core.span import Span, Trace

_USD_QUANTUM = Decimal("0.000001")
_MICRO_USD = Decimal(1_000_000)

# key name (as exposed to callers) → Span attribute it groups by.
_GROUP_KEYS = {
    "user_id": "user_id",
    "feature_tag": "feature_tag",
    "model_name": "model_name",
    "node_name": "name",
    "kind": "kind",
}


@dataclass(frozen=True)
class CostBreakdownEntry:
    key: str
    cost_usd: Decimal
    total_tokens: int
    call_count: int
    avg_cost_per_call: Decimal


def _span_cost(span: Span) -> Decimal:
    return Decimal(str(span.cost_usd)) if span.cost_usd is not None else Decimal(0)


def cost_by(traces: list[Trace], key: str) -> list[CostBreakdownEntry]:
    """Break down cost/tokens/call counts across all spans, grouped by `key`.

    key ∈ {user_id, feature_tag, model_name, node_name, kind}. Spans whose
    group value is unset (None) are skipped — e.g. chain spans when grouping
    by model_name. Sorted by cost descending (ties broken by key).
    """
    attr = _GROUP_KEYS.get(key)
    if attr is None:
        raise ValueError(f"unknown cost_by key {key!r}; expected one of {sorted(_GROUP_KEYS)}")

    cost: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
    tokens: dict[str, int] = defaultdict(int)
    calls: dict[str, int] = defaultdict(int)

    for trace in traces:
        for span in trace.spans:
            value = getattr(span, attr)
            if value is None:
                continue
            group = str(value)
            cost[group] += _span_cost(span)
            tokens[group] += span.total_tokens or 0
            calls[group] += 1

    entries = [
        CostBreakdownEntry(
            key=group,
            cost_usd=cost[group],
            total_tokens=tokens[group],
            call_count=calls[group],
            avg_cost_per_call=(cost[group] / calls[group]).quantize(
                _USD_QUANTUM, rounding=ROUND_HALF_UP
            ),
        )
        for group in cost
    ]
    entries.sort(key=lambda e: (-e.cost_usd, e.key))
    return entries


def retry_waste(traces: list[Trace]) -> Decimal:
    """Total cost of retry attempts (spans with retry_index > 0).

    This is money spent re-doing work that already failed at least once —
    the headline number for "how much are flaky calls costing us".
    """
    return sum(
        (_span_cost(span) for trace in traces for span in trace.spans if span.retry_index > 0),
        Decimal(0),
    )


def trace_flame_data(trace: Trace) -> dict[str, Any]:
    """Convert a trace into d3-flamegraph input.

    Each node is {name, value, children, data}: `value` is the *inclusive*
    cost of the subtree in integer micro-dollars (d3-flamegraph sizes a frame
    by its own value, so a parent must cover its children), and `data`
    carries tokens/model/duration for tooltips.
    """
    children_by_parent: dict[str, list[Span]] = defaultdict(list)
    for span in trace.spans:
        if span.parent_span_id is not None:
            children_by_parent[span.parent_span_id].append(span)

    def build(span: Span) -> dict[str, Any]:
        children = [build(child) for child in children_by_parent.get(span.span_id, [])]
        self_micro = int((_span_cost(span) * _MICRO_USD).to_integral_value(rounding=ROUND_HALF_UP))
        return {
            "name": span.name,
            "value": self_micro + sum(child["value"] for child in children),
            "children": children,
            "data": {
                "kind": span.kind.value,
                "status": span.status.value,
                "model": span.model_name,
                "tokens": span.total_tokens,
                "duration_ms": span.duration_ms,
                "retry_index": span.retry_index,
                "self_cost_usd": span.cost_usd,
            },
        }

    return build(trace.root)
