"""instrument_graph() on a real compiled LangGraph graph mixing sync and
async nodes: one GRAPH_NODE span per execution, graph output unchanged,
and the caller's input state never mutated."""

import copy
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import SpanKind


class _State(TypedDict):
    steps: Annotated[list[str], operator.add]
    payload: dict[str, Any]


def _build_graph() -> Any:
    def load(state: _State) -> dict[str, Any]:
        return {"steps": ["load"]}

    async def enrich(state: _State) -> dict[str, Any]:
        return {"steps": ["enrich"]}

    def summarize(state: _State) -> dict[str, Any]:
        return {"steps": ["summarize"]}

    graph = StateGraph(_State)
    graph.add_node("load", load)
    graph.add_node("enrich", enrich)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "load")
    graph.add_edge("load", "enrich")
    graph.add_edge("enrich", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


async def test_mixed_sync_async_nodes_each_get_one_graph_node_span(
    collector: TraceCollector,
) -> None:
    from tokenlens.instrument.langgraph import instrument_graph

    compiled = instrument_graph(_build_graph())
    input_state = {"steps": ["start"], "payload": {"nested": {"k": [1, 2]}}}

    with tokenlens.span("graph_run", kind=SpanKind.CHAIN):
        result = await compiled.ainvoke(input_state)

    assert result["steps"] == ["start", "load", "enrich", "summarize"]

    (trace,) = collector.flush()
    node_spans = [s for s in trace.spans if s.kind == SpanKind.GRAPH_NODE]
    assert sorted(s.name for s in node_spans) == ["enrich", "load", "summarize"]
    assert all(s.parent_span_id == trace.root.span_id for s in node_spans)
    assert all(s.end_time is not None for s in node_spans)


async def test_instrumentation_observes_state_without_mutating_it(
    collector: TraceCollector,
) -> None:
    from tokenlens.instrument.langgraph import instrument_graph

    input_state = {"steps": ["start"], "payload": {"nested": {"k": [1, 2]}}}
    snapshot = copy.deepcopy(input_state)

    # Same graph, uninstrumented: the reference output.
    baseline = await _build_graph().ainvoke(copy.deepcopy(input_state))

    compiled = instrument_graph(_build_graph())
    with tokenlens.span("graph_run", kind=SpanKind.CHAIN):
        result = await compiled.ainvoke(input_state)

    assert result == baseline  # instrumentation changed nothing about the output
    assert input_state == snapshot  # ...and never mutated the caller's state
    collector.flush()
