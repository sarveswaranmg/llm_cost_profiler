"""Live trace pub/sub for the /ws/live WebSocket.

LiveBroadcaster sits between the collector's flush thread and any number of
WebSocket clients. Publishing is fire-and-forget: the flush thread only
schedules a callback on the server's event loop (call_soon_threadsafe is
non-blocking), and a subscriber whose bounded queue is full simply loses
that message. A slow dashboard tab can never back-pressure the collector.
"""

import asyncio
import contextlib
import logging
import threading
from typing import Any

from tokenlens.core.span import SpanStatus, Trace
from tokenlens.server.models import TraceSummaryModel

logger = logging.getLogger("tokenlens.server")

QUEUE_SIZE = 100


class LiveBroadcaster:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach to the server's event loop (called at app startup)."""
        self._loop = loop

    def subscribe(self) -> "asyncio.Queue[dict[str, Any]]":
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=QUEUE_SIZE)
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[dict[str, Any]]") -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def publish(self, trace: Trace) -> None:
        """Hand a completed trace to all subscribers. Thread-safe, non-blocking.

        This is the collector-exporter entry point, called from the flush
        thread. Serialization happens here (off the event loop is fine — the
        payload is small) and fan-out happens on the loop.
        """
        loop = self._loop
        with self._lock:
            if loop is None or not self._subscribers:
                return
        payload = _summary_payload(trace)
        loop.call_soon_threadsafe(self._fan_out, payload)

    def _fan_out(self, payload: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            # Slow client: drop this message for them, never block.
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)


def _summary_payload(trace: Trace) -> dict[str, Any]:
    root = trace.root
    summary = TraceSummaryModel(
        trace_id=trace.trace_id,
        root_name=root.name,
        started_at=root.start_time,
        total_cost_usd=trace.total_cost(),
        total_tokens=trace.total_tokens(),
        user_id=root.user_id,
        feature_tag=root.feature_tag,
        session_id=root.session_id,
        has_error=any(s.status == SpanStatus.ERROR for s in trace.spans),
    )
    return summary.model_dump(mode="json")


class LiveExporter:
    """Collector exporter that forwards flushed traces to a broadcaster.

    Swallow-all, mirroring StorageExporter: a broadcast problem must never
    break the flush thread.
    """

    def __init__(self, broadcaster: LiveBroadcaster) -> None:
        self.broadcaster = broadcaster

    def __call__(self, trace: Trace) -> None:
        try:
            self.broadcaster.publish(trace)
        except Exception:
            logger.exception("tokenlens: failed to broadcast trace %s", trace.trace_id)
