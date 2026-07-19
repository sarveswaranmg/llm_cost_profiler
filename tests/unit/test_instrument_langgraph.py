import asyncio
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import SpanKind
from tokenlens.instrument.langgraph import instrument_graph, traced_node


class _State(TypedDict):
    x: Annotated[list[int], operator.add]


def test_sync_nodes_nest_under_enclosing_span(collector: TraceCollector) -> None:
    def node_a(state: _State) -> dict[str, list[int]]:
        return {"x": [state["x"][-1] + 1]}

    def node_b(state: _State) -> dict[str, list[int]]:
        return {"x": [state["x"][-1] + 10]}

    graph = StateGraph(_State)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    compiled = instrument_graph(graph.compile())

    with tokenlens.span("graph_run", kind=SpanKind.CHAIN):
        result = compiled.invoke({"x": [1]})

    assert result["x"][-1] == 12

    traces = collector.flush()
    assert len(traces) == 1
    tree = traces[0].to_tree()
    assert tree["name"] == "graph_run"
    child_names = [c["name"] for c in tree["children"]]
    assert child_names == ["a", "b"]
    assert all(c["kind"] == "GRAPH_NODE" for c in tree["children"])


async def test_async_parallel_nodes_get_isolated_spans(collector: TraceCollector) -> None:
    async def node_a(state: _State) -> dict[str, list[int]]:
        await asyncio.sleep(0.001)
        return {"x": [1]}

    async def node_b(state: _State) -> dict[str, list[int]]:
        await asyncio.sleep(0.001)
        return {"x": [2]}

    graph = StateGraph(_State)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_edge("a", END)
    graph.add_edge("b", END)
    compiled = instrument_graph(graph.compile())

    with tokenlens.span("graph_run", kind=SpanKind.CHAIN):
        result = await compiled.ainvoke({"x": []})

    assert sorted(result["x"]) == [1, 2]

    traces = collector.flush()
    assert len(traces) == 1
    tree = traces[0].to_tree()
    assert {c["name"] for c in tree["children"]} == {"a", "b"}
    assert len(traces[0].spans) == 3  # graph_run + 2 nodes


def test_instrument_graph_is_idempotent(collector: TraceCollector) -> None:
    def node_a(state: _State) -> dict[str, list[int]]:
        return {"x": [state["x"][-1] + 1]}

    graph = StateGraph(_State)
    graph.add_node("a", node_a)
    graph.add_edge(START, "a")
    graph.add_edge("a", END)
    compiled = graph.compile()

    instrument_graph(compiled)
    instrument_graph(compiled)  # calling twice must not double-wrap

    with tokenlens.span("graph_run", kind=SpanKind.CHAIN):
        compiled.invoke({"x": [1]})

    traces = collector.flush()
    tree = traces[0].to_tree()
    assert len(tree["children"]) == 1
    assert tree["children"][0]["name"] == "a"


def test_instrument_graph_skips_start_end_pseudo_nodes(collector: TraceCollector) -> None:
    def node_a(state: _State) -> dict[str, list[int]]:
        return {"x": [state["x"][-1] + 1]}

    graph = StateGraph(_State)
    graph.add_node("a", node_a)
    graph.add_edge(START, "a")
    graph.add_edge("a", END)
    compiled = instrument_graph(graph.compile())

    with tokenlens.span("graph_run", kind=SpanKind.CHAIN):
        compiled.invoke({"x": [1]})

    traces = collector.flush()
    names = {s.name for s in traces[0].spans}
    assert "__start__" not in names
    assert "__end__" not in names


def test_instrument_graph_rejects_non_graph_object() -> None:
    with pytest.raises(TypeError):
        instrument_graph(object())


def test_traced_node_decorator_sync(collector: TraceCollector) -> None:
    @traced_node("my_node")
    def step(x: int) -> int:
        return x + 1

    with tokenlens.span("root", kind=SpanKind.CHAIN):
        assert step(1) == 2

    traces = collector.flush()
    tree = traces[0].to_tree()
    assert tree["children"][0]["name"] == "my_node"
    assert tree["children"][0]["kind"] == "GRAPH_NODE"


async def test_traced_node_decorator_async(collector: TraceCollector) -> None:
    @traced_node("my_async_node")
    async def step(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    with tokenlens.span("root", kind=SpanKind.CHAIN):
        assert await step(1) == 2

    traces = collector.flush()
    tree = traces[0].to_tree()
    assert tree["children"][0]["name"] == "my_async_node"
    assert tree["children"][0]["kind"] == "GRAPH_NODE"
