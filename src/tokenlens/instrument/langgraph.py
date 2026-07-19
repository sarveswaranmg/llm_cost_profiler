"""LangGraph instrumentation.

Unlike the LangChain callback handler, this uses tokenlens' own contextvars
stack (`tokenlens.core.context.span`) directly: a compiled graph's nodes
execute in the same call stack (sync) or context-propagating asyncio tasks
(async) as whatever invoked them, so nesting works correctly whether nodes
run sequentially or fan out in parallel — see the module tests for both
cases.

`instrument_graph` only wraps node functions; it does not open an
enclosing span for the whole graph run. Wrap the top-level
`compiled_graph.invoke(...)`/`.ainvoke(...)` call in `with tokenlens.span(...):`
(or call it from inside an already-open span) so node spans land in one
coherent trace — otherwise each node becomes its own single-node trace.
"""

import functools
import inspect
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from tokenlens.core.context import span
from tokenlens.core.span import SpanKind

_T = TypeVar("_T")
_WRAPPED_MARKER = "_tokenlens_wrapped"


def traced_node(name: str) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator: wrap a node function in a GRAPH_NODE span named `name`.

    Works for both sync and async functions. Useful for manually-built
    graphs, or any node-shaped callable, that don't go through
    `instrument_graph`.
    """

    def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
        if inspect.iscoroutinefunction(func):
            async_func = func

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(name, kind=SpanKind.GRAPH_NODE):
                    return await async_func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(name, kind=SpanKind.GRAPH_NODE):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


def instrument_graph(compiled_graph: Any) -> Any:
    """Wrap every node in a compiled LangGraph graph with a GRAPH_NODE span.

    Patches each node's underlying sync/async function in place — state
    passed between nodes is only observed, never mutated — and returns the
    same graph object. Safe to call more than once (already-wrapped nodes
    are skipped) and safe to call on graphs with only a sync or only an
    async implementation per node.
    """
    nodes = getattr(compiled_graph, "nodes", None)
    if nodes is None:
        raise TypeError(
            "instrument_graph() expects a compiled LangGraph graph (an object "
            f"with a `.nodes` mapping); got {type(compiled_graph).__name__}"
        )

    for name, node in nodes.items():
        if name.startswith("__"):
            continue  # skip LangGraph's internal __start__/__end__ pseudo-nodes
        _instrument_node(name, node)

    return compiled_graph


def _instrument_node(name: str, node: Any) -> None:
    runnable = getattr(node, "bound", node)
    if getattr(runnable, _WRAPPED_MARKER, False):
        return

    func: Callable[..., Any] | None = getattr(runnable, "func", None)
    afunc: Callable[..., Coroutine[Any, Any, Any]] | None = getattr(runnable, "afunc", None)
    if func is None and afunc is None:
        return  # not a shape we recognize how to wrap; leave the node alone

    if func is not None:
        runnable.func = traced_node(name)(func)
    if afunc is not None:
        runnable.afunc = traced_node(name)(afunc)
    runnable._tokenlens_wrapped = True
