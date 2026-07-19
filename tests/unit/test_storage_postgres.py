"""Postgres backend tests.

Run only when TOKENLENS_PG_DSN points at a database we may write to (tables
are created and truncated); skipped gracefully otherwise, e.g.:

    TOKENLENS_PG_DSN=postgresql://localhost/tokenlens_test pytest tests/test_storage_postgres.py
"""

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from tests.storage_utils import build_pipeline_trace, matches_filters

from tokenlens.cost.attribution import cost_by
from tokenlens.storage.base import GROUP_BY_COLUMNS, TraceFilters

pytestmark = pytest.mark.skipif(
    not os.environ.get("TOKENLENS_PG_DSN"),
    reason="TOKENLENS_PG_DSN not set; skipping Postgres storage tests",
)

BASE = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
async def backend() -> AsyncIterator["PostgresBackend"]:  # noqa: F821
    from tokenlens.storage.postgres import PostgresBackend

    b = PostgresBackend()
    pool = await b._get_pool()
    await pool.execute("TRUNCATE traces, spans")
    yield b
    await b.close()


def sample_traces():
    return [
        build_pipeline_trace("trace-1", BASE),
        build_pipeline_trace(
            "trace-2",
            BASE + timedelta(hours=1),
            user_id="u-2",
            feature_tag="search",
            cost_a=0.123456,
            cost_b=0.000001,
        ),
    ]


async def test_round_trip_trace_tree(backend) -> None:
    original = build_pipeline_trace("trace-rt", BASE)
    await backend.save_traces([original])
    await backend.save_traces([original])  # idempotent

    loaded = await backend.get_trace("trace-rt")
    assert loaded is not None
    assert sorted(loaded.spans, key=lambda s: s.span_id) == sorted(
        original.spans, key=lambda s: s.span_id
    )
    assert loaded.to_tree() == original.to_tree()
    assert await backend.get_trace("nope") is None


async def test_list_traces_and_filters(backend) -> None:
    traces = sample_traces()
    await backend.save_traces(traces)

    assert [s.trace_id for s in await backend.list_traces()] == ["trace-2", "trace-1"]

    for filters in (
        TraceFilters(user_id="u-2"),
        TraceFilters(feature_tag="chat"),
        TraceFilters(model="gpt-4o"),
        TraceFilters(since=BASE + timedelta(minutes=30)),
        TraceFilters(min_cost=0.3),
    ):
        expected = {t.trace_id for t in traces if matches_filters(t, filters)}
        got = {s.trace_id for s in await backend.list_traces(filters=filters)}
        assert got == expected, filters


async def test_aggregate_matches_pure_python_attribution(backend) -> None:
    traces = sample_traces()
    await backend.save_traces(traces)

    for group_by in GROUP_BY_COLUMNS:
        assert await backend.aggregate(group_by) == cost_by(traces, group_by)

    filters = TraceFilters(feature_tag="search")
    kept = [t for t in traces if matches_filters(t, filters)]
    for group_by in GROUP_BY_COLUMNS:
        assert await backend.aggregate(group_by, filters=filters) == cost_by(kept, group_by)


async def test_prune(backend) -> None:
    now = datetime.now(UTC)
    await backend.save_traces(
        [
            build_pipeline_trace("trace-old", now - timedelta(days=40)),
            build_pipeline_trace("trace-recent", now - timedelta(days=1)),
        ]
    )
    assert await backend.prune(older_than_days=30) == 1
    assert [s.trace_id for s in await backend.list_traces()] == ["trace-recent"]
