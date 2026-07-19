"""Context propagation across threads.

contextvars gives every new thread an EMPTY context, so propagation into a
ThreadPoolExecutor is explicit: snapshot with contextvars.copy_context()
and submit ctx.run(fn). These tests pin down both sides of that contract —
propagated workers nest under the parent span, unpropagated workers start
their own traces (and never corrupt anyone else's context).
"""

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import SpanKind


def test_threadpool_children_share_parent_when_context_is_propagated(
    collector: TraceCollector,
) -> None:
    started = threading.Barrier(3)

    def work(i: int) -> None:
        started.wait()  # force all three spans to be open concurrently
        with tokenlens.span(f"child-{i}"):
            pass

    with (
        tokenlens.span("root", kind=SpanKind.CHAIN) as root,
        ThreadPoolExecutor(max_workers=3) as pool,
    ):
        # One copy_context() snapshot per task: a single Context object
        # cannot be entered by two threads at once.
        futures = [pool.submit(contextvars.copy_context().run, work, i) for i in range(3)]
        for f in futures:
            f.result()

    traces = collector.flush()
    assert len(traces) == 1
    children = [s for s in traces[0].spans if s.parent_span_id is not None]
    assert len(children) == 3
    assert all(c.parent_span_id == root.span_id for c in children)
    assert all(c.trace_id == root.trace_id for c in children)
    assert len({c.span_id for c in children}) == 3  # no cross-thread corruption
    assert sorted(c.name for c in children) == ["child-0", "child-1", "child-2"]


def test_threadpool_attribution_propagates_with_context(collector: TraceCollector) -> None:
    tokenlens.set_user("u-threaded")
    tokenlens.set_feature("fanout")

    def work() -> None:
        with tokenlens.span("worker"):
            pass

    with tokenlens.span("root", kind=SpanKind.CHAIN), ThreadPoolExecutor(max_workers=2) as pool:
        for f in [pool.submit(contextvars.copy_context().run, work) for _ in range(2)]:
            f.result()

    (trace,) = collector.flush()
    workers = [s for s in trace.spans if s.name == "worker"]
    assert len(workers) == 2
    assert all(s.user_id == "u-threaded" for s in workers)
    assert all(s.feature_tag == "fanout" for s in workers)


def test_unpropagated_threads_start_fresh_traces_without_corrupting_parent(
    collector: TraceCollector,
) -> None:
    # Submitted WITHOUT copy_context(): the worker thread's context is empty,
    # so its span becomes the root of its own trace — by design.
    def work(i: int) -> None:
        with tokenlens.span(f"detached-{i}"):
            pass

    with tokenlens.span("root", kind=SpanKind.CHAIN) as root:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in [pool.submit(work, i) for i in range(3)]:
                f.result()
        # The main thread's context must be untouched by the workers.
        with tokenlens.span("attached-child"):
            pass

    traces = {t.root.name: t for t in collector.flush()}
    assert set(traces) == {"root", "detached-0", "detached-1", "detached-2"}
    root_trace = traces["root"]
    assert [s.name for s in root_trace.spans if s.parent_span_id] == ["attached-child"]
    assert all(t.root.parent_span_id is None for t in traces.values())
    assert all(t.trace_id != root.trace_id for name, t in traces.items() if name != "root")
