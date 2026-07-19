"""LiveBroadcaster under pressure: the collector's flush thread must never
block or crash on account of slow, absent, or departed WebSocket clients."""

import asyncio
import time

import pytest

from tokenlens.server.live import QUEUE_SIZE, LiveBroadcaster, LiveExporter


async def test_slow_subscriber_drops_messages_but_never_blocks_the_flush_thread(
    sample_trace,
) -> None:
    broadcaster = LiveBroadcaster()
    broadcaster.bind(asyncio.get_running_loop())
    queue = broadcaster.subscribe()
    trace = sample_trace()
    exporter = LiveExporter(broadcaster)
    burst = QUEUE_SIZE * 5

    def flush_thread() -> float:
        start = time.monotonic()
        for _ in range(burst):
            exporter(trace)
        return time.monotonic() - start

    # The publish burst runs on a real worker thread (like the collector's
    # flush thread) while this client reads nothing at all.
    elapsed = await asyncio.wait_for(asyncio.to_thread(flush_thread), timeout=5.0)
    assert elapsed < 2.0  # fire-and-forget: no waiting on the slow client

    # Let the scheduled fan-out callbacks drain onto the loop.
    for _ in range(10):
        await asyncio.sleep(0)

    # The slow client lost messages instead of buffering unboundedly.
    assert queue.qsize() <= QUEUE_SIZE
    payload = queue.get_nowait()
    assert payload["trace_id"] == trace.trace_id
    assert payload["has_error"] is False


async def test_unsubscribed_client_stops_receiving(sample_trace) -> None:
    broadcaster = LiveBroadcaster()
    broadcaster.bind(asyncio.get_running_loop())
    queue = broadcaster.subscribe()

    broadcaster.publish(sample_trace())
    await asyncio.sleep(0)
    assert queue.qsize() == 1

    broadcaster.unsubscribe(queue)
    broadcaster.publish(sample_trace(trace_id="after-unsubscribe"))
    await asyncio.sleep(0)
    assert queue.qsize() == 1  # nothing new arrived


async def test_publish_without_subscribers_is_a_cheap_noop(sample_trace) -> None:
    broadcaster = LiveBroadcaster()
    broadcaster.bind(asyncio.get_running_loop())
    # No subscribers: publish must return immediately without serializing.
    broadcaster.publish(sample_trace())


async def test_live_exporter_swallows_broadcast_failures(
    sample_trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    broadcaster = LiveBroadcaster()

    def explode(trace: object) -> None:
        raise RuntimeError("loop is gone")

    monkeypatch.setattr(broadcaster, "publish", explode)
    LiveExporter(broadcaster)(sample_trace())  # must not raise
