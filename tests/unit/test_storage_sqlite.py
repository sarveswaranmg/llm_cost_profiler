"""SQLite backend tests: round-trips, filters, and SQL-vs-Python attribution
agreement (aggregate() must reproduce cost_by() exactly)."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from tests.storage_utils import build_pipeline_trace, matches_filters

import tokenlens
import tokenlens.storage as storage
from tokenlens.core.collector import TraceCollector, set_collector
from tokenlens.core.span import SpanKind, Trace
from tokenlens.cost.attribution import cost_by
from tokenlens.storage.base import GROUP_BY_COLUMNS, TraceFilters
from tokenlens.storage.sqlite import SqliteBackend

BASE = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
async def backend(tmp_path) -> AsyncIterator[SqliteBackend]:
    b = SqliteBackend(tmp_path / "traces.db")
    yield b
    await b.close()


def sample_traces() -> list[Trace]:
    return [
        build_pipeline_trace("trace-1", BASE),
        build_pipeline_trace(
            "trace-2",
            BASE + timedelta(hours=1),
            user_id="u-2",
            feature_tag="search",
            model_b="claude-haiku-4-5",
            cost_a=0.123456,
            cost_b=0.000001,
        ),
        build_pipeline_trace(
            "trace-3",
            BASE + timedelta(hours=2),
            user_id="u-1",
            feature_tag="search",
            model_a="gpt-4o-mini",
            cost_a=0.007,
            cost_b=1.25,
        ),
    ]


async def test_round_trip_trace_tree(backend: SqliteBackend) -> None:
    original = build_pipeline_trace("trace-rt", BASE)
    await backend.save_traces([original])

    loaded = await backend.get_trace("trace-rt")
    assert loaded is not None
    assert loaded.trace_id == original.trace_id
    assert loaded.root == original.root

    assert sorted(loaded.spans, key=lambda s: s.span_id) == sorted(
        original.spans, key=lambda s: s.span_id
    )
    # The reassembled tree (parent/child structure) survives too.
    assert loaded.to_tree() == original.to_tree()


async def test_get_trace_unknown_returns_none(backend: SqliteBackend) -> None:
    assert await backend.get_trace("nope") is None


async def test_save_is_idempotent(backend: SqliteBackend) -> None:
    trace = build_pipeline_trace("trace-dup", BASE)
    await backend.save_traces([trace])
    await backend.save_traces([trace])

    summaries = await backend.list_traces()
    assert [s.trace_id for s in summaries] == ["trace-dup"]
    loaded = await backend.get_trace("trace-dup")
    assert loaded is not None
    assert len(loaded.spans) == len(trace.spans)


async def test_list_traces_summary_fields_and_order(backend: SqliteBackend) -> None:
    traces = sample_traces()
    await backend.save_traces(traces)

    summaries = await backend.list_traces()
    # Newest first.
    assert [s.trace_id for s in summaries] == ["trace-3", "trace-2", "trace-1"]

    first = summaries[0]
    trace3 = traces[2]
    assert first.root_name == "pipeline"
    assert first.started_at == trace3.root.start_time
    assert first.total_cost_usd == pytest.approx(trace3.total_cost())
    assert first.total_tokens == trace3.total_tokens()
    assert first.user_id == "u-1"
    assert first.feature_tag == "search"
    assert first.session_id == "sess-1"
    # Every pipeline trace contains an ERROR retry span.
    assert first.has_error is True


async def test_list_traces_limit_offset(backend: SqliteBackend) -> None:
    await backend.save_traces(sample_traces())
    page1 = await backend.list_traces(limit=2)
    page2 = await backend.list_traces(limit=2, offset=2)
    assert [s.trace_id for s in page1] == ["trace-3", "trace-2"]
    assert [s.trace_id for s in page2] == ["trace-1"]


@pytest.mark.parametrize(
    "filters",
    [
        TraceFilters(user_id="u-1"),
        TraceFilters(feature_tag="search"),
        TraceFilters(model="gpt-4o"),
        TraceFilters(since=BASE + timedelta(minutes=30)),
        TraceFilters(until=BASE + timedelta(minutes=30)),
        TraceFilters(since=BASE + timedelta(minutes=30), until=BASE + timedelta(minutes=90)),
        TraceFilters(min_cost=0.3),
        TraceFilters(user_id="u-1", feature_tag="search", model="claude-sonnet-5"),
        TraceFilters(user_id="nobody"),
    ],
)
async def test_list_traces_filters(backend: SqliteBackend, filters: TraceFilters) -> None:
    traces = sample_traces()
    await backend.save_traces(traces)

    expected = sorted(
        (t.trace_id for t in traces if matches_filters(t, filters)),
        key=lambda tid: next(t.root.start_time for t in traces if t.trace_id == tid),
        reverse=True,
    )
    got = [s.trace_id for s in await backend.list_traces(filters=filters)]
    assert got == expected


@pytest.mark.parametrize("group_by", sorted(GROUP_BY_COLUMNS))
async def test_aggregate_matches_pure_python_attribution(
    backend: SqliteBackend, group_by: str
) -> None:
    traces = sample_traces()
    await backend.save_traces(traces)

    assert await backend.aggregate(group_by) == cost_by(traces, group_by)


@pytest.mark.parametrize(
    "filters",
    [
        TraceFilters(user_id="u-1"),
        TraceFilters(feature_tag="search", min_cost=0.5),
        TraceFilters(model="gpt-4o-mini"),
        TraceFilters(since=BASE + timedelta(minutes=30)),
    ],
)
async def test_aggregate_with_filters_matches_filtered_cost_by(
    backend: SqliteBackend, filters: TraceFilters
) -> None:
    traces = sample_traces()
    await backend.save_traces(traces)

    kept = [t for t in traces if matches_filters(t, filters)]
    for group_by in GROUP_BY_COLUMNS:
        assert await backend.aggregate(group_by, filters=filters) == cost_by(kept, group_by)


async def test_aggregate_rejects_unknown_key(backend: SqliteBackend) -> None:
    with pytest.raises(ValueError, match="group_by"):
        await backend.aggregate("cost_usd; DROP TABLE spans")


async def test_prune_deletes_old_traces_and_spans(backend: SqliteBackend) -> None:
    now = datetime.now(UTC)
    old = build_pipeline_trace("trace-old", now - timedelta(days=40))
    recent = build_pipeline_trace("trace-recent", now - timedelta(days=1))
    await backend.save_traces([old, recent])

    deleted = await backend.prune(older_than_days=30)
    assert deleted == 1
    assert await backend.get_trace("trace-old") is None
    assert [s.trace_id for s in await backend.list_traces()] == ["trace-recent"]


async def test_schema_migration_is_idempotent(tmp_path) -> None:
    path = tmp_path / "traces.db"
    first = SqliteBackend(path)
    await first.save_traces([build_pipeline_trace("trace-1", BASE)])
    await first.close()

    # Reopening runs the migration again against an existing file.
    second = SqliteBackend(path)
    assert [s.trace_id for s in await second.list_traces()] == ["trace-1"]
    await second.close()


@pytest.fixture
def _global_state_reset():
    yield
    set_collector(TraceCollector(flush_interval_seconds=0))
    storage._backend = None


def test_init_wires_collector_flush_to_backend(tmp_path, _global_state_reset) -> None:
    set_collector(TraceCollector(flush_interval_seconds=0))
    db_path = tmp_path / "wired.db"
    backend = tokenlens.init(storage="sqlite", db_path=db_path)
    assert isinstance(backend, SqliteBackend)

    tokenlens.set_user("u-42")
    with (
        tokenlens.span("pipeline", kind=SpanKind.CHAIN),
        tokenlens.span(
            "call", kind=SpanKind.LLM_CALL, model_name="gpt-4o", input_tokens=10, output_tokens=5
        ),
    ):
        pass
    tokenlens.get_collector().flush()

    # StorageExporter persists synchronously during flush(), so a fresh
    # backend on the same file must see the trace immediately.
    reader = SqliteBackend(db_path)
    summaries = storage._runner.run(reader.list_traces())
    assert len(summaries) == 1
    assert summaries[0].root_name == "pipeline"
    assert summaries[0].user_id == "u-42"
    trace = storage._runner.run(reader.get_trace(summaries[0].trace_id))
    assert trace is not None and len(trace.spans) == 2
    storage._runner.run(reader.close())


def test_init_accepts_backend_instance_and_prune_helper(tmp_path, _global_state_reset) -> None:
    set_collector(TraceCollector(flush_interval_seconds=0))
    backend = SqliteBackend(tmp_path / "prune.db")
    assert tokenlens.init(storage=backend) is backend

    now = datetime.now(UTC)
    exporter = storage.StorageExporter(backend)
    exporter(build_pipeline_trace("trace-old", now - timedelta(days=45)))
    exporter(build_pipeline_trace("trace-new", now))

    assert storage.prune(older_than_days=30) == 1
    remaining = storage._runner.run(backend.list_traces())
    assert [s.trace_id for s in remaining] == ["trace-new"]


def test_init_rejects_unknown_storage(_global_state_reset) -> None:
    with pytest.raises(ValueError, match="unknown storage"):
        tokenlens.init(storage="mongodb")
