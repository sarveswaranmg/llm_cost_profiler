"""In-process trace collector.

TraceCollector buffers finished spans in memory, grouped by trace id. A
trace is considered complete once its root span (parent_span_id is None) has
been recorded — which relies on children being closed before their parent's
`with tokenlens.span(...)` block exits, exactly as nested or awaited spans
naturally do. Complete traces are handed to pluggable exporter callbacks on
flush(), which runs periodically on a background daemon thread and once
more at interpreter exit.
"""

import atexit
import logging
import threading
from collections import defaultdict
from collections.abc import Callable

from tokenlens.core.span import Span, Trace

logger = logging.getLogger("tokenlens.core")

Exporter = Callable[[Trace], None]
Enricher = Callable[[Span], None]


def _cost_enricher(span: Span) -> None:
    # Imported lazily: tokenlens.cost depends on tokenlens.core, so a
    # top-level import here would be circular during package init.
    from tokenlens.cost.pricing import enrich_span_cost

    enrich_span_cost(span)


class InMemoryExporter:
    """Exporter that keeps completed traces in memory.

    The default for directly constructed TraceCollectors (handy in tests).
    The process-wide collector from get_collector() persists to the
    configured storage backend instead — see _default_exporter().
    """

    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def __call__(self, trace: Trace) -> None:
        self.traces.append(trace)


class TraceCollector:
    def __init__(
        self,
        flush_interval_seconds: float = 5.0,
        exporters: list[Exporter] | None = None,
        enrichers: list[Enricher] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._spans_by_trace: dict[str, list[Span]] = defaultdict(list)
        self._completed_trace_ids: set[str] = set()
        self.exporters: list[Exporter] = (
            exporters if exporters is not None else [InMemoryExporter()]
        )
        self.enrichers: list[Enricher] = enrichers if enrichers is not None else [_cost_enricher]
        self._flush_interval = flush_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        if flush_interval_seconds > 0:
            self._start_background_flush()
        atexit.register(self.flush)

    def add_exporter(self, exporter: Exporter) -> None:
        with self._lock:
            self.exporters.append(exporter)

    def set_exporters(self, exporters: list[Exporter]) -> None:
        """Replace all exporters. Used by tokenlens.init() to rewire storage."""
        with self._lock:
            self.exporters = list(exporters)

    def remove_exporter(self, exporter: Exporter) -> None:
        """Detach an exporter added with add_exporter; no-op if absent."""
        with self._lock:
            self.exporters = [e for e in self.exporters if e is not exporter]

    def record(self, span: Span) -> None:
        for enricher in self.enrichers:
            try:
                enricher(span)
            except Exception:
                # Enrichment (e.g. cost computation) must never break tracing,
                # let alone the user's app — record the span as-is.
                logger.exception("tokenlens: span enricher %r failed", enricher)
        with self._lock:
            self._spans_by_trace[span.trace_id].append(span)
            if span.parent_span_id is None:
                self._completed_trace_ids.add(span.trace_id)

    def flush(self) -> list[Trace]:
        with self._lock:
            ready_ids = list(self._completed_trace_ids)
            traces: list[Trace] = []
            for trace_id in ready_ids:
                spans = self._spans_by_trace.pop(trace_id, [])
                self._completed_trace_ids.discard(trace_id)
                if spans:
                    traces.append(Trace.from_spans(trace_id, spans))

        # Exporters run outside the lock so a slow or reentrant exporter
        # (e.g. one that calls back into record()) can't deadlock the
        # collector or block concurrent span recording.
        for trace in traces:
            for exporter in self.exporters:
                exporter(trace)
        return traces

    def _start_background_flush(self) -> None:
        def _loop() -> None:
            while not self._stop_event.wait(self._flush_interval):
                self.flush()

        self._thread = threading.Thread(target=_loop, name="tokenlens-flush", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the background flush thread and flush any remaining traces."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.flush()


_collector: TraceCollector | None = None
_collector_lock = threading.Lock()


def _default_exporter() -> Exporter:
    # Imported lazily: tokenlens.storage depends on tokenlens.core, so a
    # top-level import here would be circular during package init.
    try:
        from tokenlens.storage import StorageExporter

        return StorageExporter()
    except Exception:
        logger.exception(
            "tokenlens: could not set up the default storage backend; "
            "traces will be kept in memory only"
        )
        return InMemoryExporter()


def get_collector() -> TraceCollector:
    """Return the process-wide TraceCollector, creating it on first use.

    Unless tokenlens.init() configured something else, completed traces are
    persisted to the default SQLite backend (~/.tokenlens/traces.db).
    """
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = TraceCollector(exporters=[_default_exporter()])
    return _collector


def set_collector(collector: TraceCollector) -> None:
    """Replace the global collector. Mainly useful for tests."""
    global _collector
    with _collector_lock:
        _collector = collector
