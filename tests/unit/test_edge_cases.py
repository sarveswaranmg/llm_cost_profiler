"""Failure-mode and edge-case tests.

The most important invariant in this file: a broken storage backend must
never surface an exception to the user's application. tokenlens is a
profiler — it observes, it never takes the app down.
"""

import logging

import pytest

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import PROMPT_PREVIEW_LIMIT, Span, SpanKind, Trace
from tokenlens.cost.attribution import cost_by, retry_waste
from tokenlens.storage import StorageExporter
from tokenlens.storage.base import StatsOverview, StorageBackend, TraceFilters

# -- 34: zero-token, tool-only traces ---------------------------------------


def _tool_only_trace(trace_id: str = "tool-only") -> Trace:
    root = Span(trace_id=trace_id, name="job", kind=SpanKind.CHAIN, user_id="u-1")
    tool = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="shell",
        kind=SpanKind.TOOL,
        user_id="u-1",
    )
    return Trace.from_spans(trace_id, [root, tool])


def test_tool_only_trace_costs_zero_without_division_errors() -> None:
    trace = _tool_only_trace()

    assert trace.total_cost() == 0.0
    assert trace.total_tokens() == 0
    assert retry_waste([trace]) == 0

    entries = cost_by([trace], "kind")
    by_kind = {e.key: e for e in entries}
    assert by_kind["TOOL"].cost_usd == 0
    assert by_kind["TOOL"].avg_cost_per_call == 0  # no ZeroDivisionError
    assert by_kind["TOOL"].call_count == 1


async def test_overview_of_empty_backend_has_zero_rates(tmp_sqlite_storage) -> None:
    overview = await tmp_sqlite_storage.overview()

    assert overview.trace_count == 0
    assert overview.total_cost_usd == 0
    assert overview.error_rate == 0.0  # no division by zero traces
    assert overview.retry_waste_usd == 0
    assert overview.top_traces == []


async def test_zero_cost_trace_round_trips_through_storage(tmp_sqlite_storage) -> None:
    await tmp_sqlite_storage.save_traces([_tool_only_trace()])

    overview = await tmp_sqlite_storage.overview()
    assert overview.trace_count == 1
    assert overview.total_cost_usd == 0
    assert overview.error_rate == 0.0


# -- 35: unicode + oversized prompts ----------------------------------------


def test_long_multibyte_prompt_truncates_safely() -> None:
    # 10k characters of multi-byte text: emoji (4-byte UTF-8) + CJK.
    prompt = ("🐍測試🔥" * 2500)[:10_000]
    span = Span(trace_id="t", name="llm", prompt_preview=prompt)

    preview = span.prompt_preview
    assert preview is not None
    assert len(preview) == PROMPT_PREVIEW_LIMIT + 1  # 200 chars + ellipsis
    assert preview.endswith("…")
    assert preview[:-1] == prompt[:PROMPT_PREVIEW_LIMIT]  # cut between chars, never inside
    preview.encode("utf-8")  # still valid text — no broken surrogates


def test_prompt_exactly_at_limit_is_untouched() -> None:
    prompt = "字" * PROMPT_PREVIEW_LIMIT
    span = Span(trace_id="t", name="llm", prompt_preview=prompt)
    assert span.prompt_preview == prompt


# -- 36: storage failure must never reach the user's app ---------------------


class ExplodingBackend:
    """Satisfies the StorageBackend protocol; every operation raises."""

    async def save_traces(self, traces: list[Trace]) -> None:
        raise RuntimeError("disk on fire")

    async def get_trace(self, trace_id: str) -> Trace | None:
        raise RuntimeError("disk on fire")

    async def list_traces(
        self, filters: TraceFilters | None = None, *, limit: int = 50, offset: int = 0
    ):
        raise RuntimeError("disk on fire")

    async def aggregate(self, group_by: str, filters: TraceFilters | None = None):
        raise RuntimeError("disk on fire")

    async def overview(self, filters: TraceFilters | None = None) -> StatsOverview:
        raise RuntimeError("disk on fire")

    async def prune(self, older_than_days: int = 30) -> int:
        raise RuntimeError("disk on fire")

    async def close(self) -> None:
        raise RuntimeError("disk on fire")


def test_storage_write_failure_is_logged_and_swallowed(
    collector: TraceCollector, caplog: pytest.LogCaptureFixture
) -> None:
    backend = ExplodingBackend()
    assert isinstance(backend, StorageBackend)  # protocol check init() relies on
    collector.set_exporters([StorageExporter(backend)])

    # The user's app: traced code plus a flush. Neither may raise.
    with (
        tokenlens.span("user-request", kind=SpanKind.CHAIN),
        tokenlens.span("llm", kind=SpanKind.LLM_CALL),
    ):
        pass

    with caplog.at_level(logging.ERROR, logger="tokenlens.storage"):
        flushed = collector.flush()  # must not propagate RuntimeError

    assert len(flushed) == 1  # the trace was still assembled and handed over
    assert any("failed to persist" in r.message for r in caplog.records)


def test_storage_failure_does_not_starve_other_exporters(
    collector: TraceCollector, caplog: pytest.LogCaptureFixture
) -> None:
    from tokenlens.core.collector import InMemoryExporter

    memory = InMemoryExporter()
    collector.set_exporters([StorageExporter(ExplodingBackend()), memory])

    with tokenlens.span("root"):
        pass
    with caplog.at_level(logging.ERROR, logger="tokenlens.storage"):
        collector.flush()

    assert [t.root.name for t in memory.traces] == ["root"]


# -- 37: clock sanity ---------------------------------------------------------


def test_duration_is_zero_not_negative_when_clock_does_not_advance(
    collector: TraceCollector, frozen_clock
) -> None:
    with tokenlens.span("instant") as s:
        pass

    assert s.end_time is not None
    assert s.end_time >= s.start_time
    assert s.duration_ms == 0.0


def test_duration_matches_ticked_clock_exactly(collector: TraceCollector, frozen_clock) -> None:
    with tokenlens.span("outer") as outer:
        frozen_clock.tick(ms=25)
        with tokenlens.span("inner") as inner:
            frozen_clock.tick(ms=5)

    assert inner.duration_ms == 5.0
    assert outer.duration_ms == 30.0
    assert outer.end_time is not None and inner.end_time is not None
    assert outer.end_time >= inner.end_time >= inner.start_time >= outer.start_time


def test_error_span_still_gets_valid_end_time(collector: TraceCollector, frozen_clock) -> None:
    with pytest.raises(ValueError, match="boom"), tokenlens.span("failing") as s:
        frozen_clock.tick(ms=3)
        raise ValueError("boom")

    assert s.status.value == "ERROR"
    assert s.error_message == "boom"
    assert s.end_time is not None
    assert s.duration_ms == 3.0
    assert s.end_time >= s.start_time
