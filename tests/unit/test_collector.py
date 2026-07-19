"""TraceCollector unit tests: buffering, flush grouping, the background
flush thread, atexit flush (via subprocess), and thread-safety of record()."""

import os
import sqlite3
import subprocess
import sys
import threading
import time

from tokenlens.core.collector import InMemoryExporter, TraceCollector
from tokenlens.core.span import Span, SpanKind


def _root(trace_id: str, name: str = "root") -> Span:
    return Span(trace_id=trace_id, name=name, kind=SpanKind.CHAIN)


def _child(trace_id: str, parent: Span, name: str = "child") -> Span:
    return Span(trace_id=trace_id, parent_span_id=parent.span_id, name=name)


def test_flush_groups_buffered_spans_by_trace_id() -> None:
    c = TraceCollector(flush_interval_seconds=0, enrichers=[])
    root_a, root_b = _root("trace-a"), _root("trace-b")
    # Interleave the two traces' spans to prove grouping is by trace_id,
    # not by arrival order.
    c.record(_child("trace-a", root_a))
    c.record(_child("trace-b", root_b))
    c.record(_child("trace-a", root_a, name="child-2"))
    c.record(root_a)
    c.record(root_b)

    traces = {t.trace_id: t for t in c.flush()}

    assert set(traces) == {"trace-a", "trace-b"}
    assert len(traces["trace-a"].spans) == 3
    assert len(traces["trace-b"].spans) == 2
    assert all(s.trace_id == "trace-a" for s in traces["trace-a"].spans)


def test_incomplete_trace_is_not_flushed_until_root_arrives() -> None:
    c = TraceCollector(flush_interval_seconds=0, enrichers=[])
    root = _root("trace-x")
    c.record(_child("trace-x", root))

    assert c.flush() == []  # only a child so far — trace still open

    c.record(root)
    traces = c.flush()
    assert [t.trace_id for t in traces] == ["trace-x"]
    assert len(traces[0].spans) == 2
    assert c.flush() == []  # nothing left behind


def test_flush_hands_traces_to_every_exporter() -> None:
    first, second = InMemoryExporter(), InMemoryExporter()
    c = TraceCollector(flush_interval_seconds=0, exporters=[first, second], enrichers=[])
    c.record(_root("trace-1"))

    c.flush()

    assert [t.trace_id for t in first.traces] == ["trace-1"]
    assert [t.trace_id for t in second.traces] == ["trace-1"]


def test_add_and_remove_exporter() -> None:
    base, extra = InMemoryExporter(), InMemoryExporter()
    c = TraceCollector(flush_interval_seconds=0, exporters=[base], enrichers=[])

    c.add_exporter(extra)
    c.record(_root("trace-1"))
    c.flush()

    c.remove_exporter(extra)
    c.record(_root("trace-2"))
    c.flush()

    assert [t.trace_id for t in base.traces] == ["trace-1", "trace-2"]
    assert [t.trace_id for t in extra.traces] == ["trace-1"]


def test_background_flush_thread_fires_without_manual_flush() -> None:
    exporter = InMemoryExporter()
    c = TraceCollector(flush_interval_seconds=0.05, exporters=[exporter], enrichers=[])
    try:
        c.record(_root("bg-trace"))

        deadline = time.monotonic() + 2.0
        while not exporter.traces and time.monotonic() < deadline:
            time.sleep(0.01)

        assert [t.trace_id for t in exporter.traces] == ["bg-trace"]
    finally:
        c.shutdown()


def test_shutdown_stops_thread_and_flushes_remainder() -> None:
    exporter = InMemoryExporter()
    # Interval far beyond the test's lifetime: only shutdown() can flush.
    c = TraceCollector(flush_interval_seconds=3600, exporters=[exporter], enrichers=[])
    c.record(_root("late-trace"))

    c.shutdown()

    assert [t.trace_id for t in exporter.traces] == ["late-trace"]
    assert c._thread is not None and not c._thread.is_alive()


def test_atexit_flush_persists_traces_on_interpreter_exit(tmp_path) -> None:
    # A real interpreter run that records a span and exits WITHOUT flushing:
    # the atexit hook registered by TraceCollector must flush to storage.
    db_path = tmp_path / "atexit.db"
    code = (
        "import tokenlens\n"
        "with tokenlens.span('root'):\n"
        "    with tokenlens.span('child'):\n"
        "        pass\n"
        "# no flush() on purpose — atexit must handle it\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TOKENLENS_DB_PATH": str(db_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    con = sqlite3.connect(db_path)
    try:
        (trace_count,) = con.execute("SELECT COUNT(*) FROM traces").fetchone()
        (span_count,) = con.execute("SELECT COUNT(*) FROM spans").fetchone()
    finally:
        con.close()
    assert trace_count == 1
    assert span_count == 2


def test_concurrent_recording_from_ten_threads_loses_nothing() -> None:
    c = TraceCollector(flush_interval_seconds=0, enrichers=[])
    threads_n, traces_per_thread = 10, 30
    barrier = threading.Barrier(threads_n)

    def worker(thread_idx: int) -> None:
        barrier.wait()  # maximize interleaving
        for i in range(traces_per_thread):
            trace_id = f"t{thread_idx}-{i}"
            root = _root(trace_id)
            c.record(_child(trace_id, root))
            c.record(root)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    traces = c.flush()

    expected_ids = {f"t{i}-{j}" for i in range(threads_n) for j in range(traces_per_thread)}
    assert {t.trace_id for t in traces} == expected_ids
    assert all(len(t.spans) == 2 for t in traces)
    assert c.flush() == []  # nothing dropped, nothing duplicated
