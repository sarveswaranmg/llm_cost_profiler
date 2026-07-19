"""Integration fixtures: a SQLite backend seeded with 10 varied traces and
an httpx client over the real FastAPI app.

The seed mix (users u-0..u-4, features chat/search/summarize/ops/batch,
OpenAI + Anthropic models, 3 traces with errors, 7 with retry spans) is
deliberately uneven so filters, aggregation, and stats all have something
to disagree about if the SQL drifts from the Python attribution math.
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from tests.factories import sample_trace
from tests.storage_utils import build_pipeline_trace

from tokenlens.core.span import Span, SpanKind, Trace
from tokenlens.server.app import create_app
from tokenlens.storage.sqlite import SqliteBackend

SEED_BASE = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)


def _tool_only_trace(trace_id: str, started_at: datetime) -> Trace:
    root = Span(
        trace_id=trace_id,
        name="nightly_export",
        kind=SpanKind.CHAIN,
        start_time=started_at,
        end_time=started_at + timedelta(seconds=2),
        user_id="u-2",
        feature_tag="ops",
    )
    tool = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="dump_csv",
        kind=SpanKind.TOOL,
        start_time=started_at + timedelta(milliseconds=100),
        end_time=started_at + timedelta(seconds=1),
        user_id="u-2",
        feature_tag="ops",
    )
    return Trace.from_spans(trace_id, [root, tool])


def _expensive_trace(trace_id: str, started_at: datetime) -> Trace:
    root = Span(
        trace_id=trace_id,
        name="batch_summarize",
        kind=SpanKind.CHAIN,
        start_time=started_at,
        end_time=started_at + timedelta(seconds=30),
        user_id="u-3",
        feature_tag="batch",
    )
    call = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="summarize_corpus",
        kind=SpanKind.LLM_CALL,
        start_time=started_at + timedelta(seconds=1),
        end_time=started_at + timedelta(seconds=29),
        model_name="claude-sonnet-5",
        provider="anthropic",
        input_tokens=500_000,
        output_tokens=60_000,
        cost_usd=2.5,
        user_id="u-3",
        feature_tag="batch",
    )
    return Trace.from_spans(trace_id, [root, call])


def build_seed_traces() -> list[Trace]:
    h = timedelta(hours=1)
    return [
        sample_trace("seed-0", SEED_BASE, user_id="u-0", feature_tag="chat"),
        sample_trace("seed-1", SEED_BASE + 1 * h, user_id="u-1", feature_tag="chat"),
        sample_trace("seed-2", SEED_BASE + 2 * h, user_id="u-2", feature_tag="search"),
        sample_trace("seed-3", SEED_BASE + 3 * h, user_id="u-3", feature_tag="search"),
        build_pipeline_trace("seed-4", SEED_BASE + 4 * h, user_id="u-4", feature_tag="summarize"),
        build_pipeline_trace(
            "seed-5",
            SEED_BASE + 5 * h,
            user_id="u-0",
            feature_tag="summarize",
            model_a="gpt-4.1",
            cost_a=0.05,
        ),
        build_pipeline_trace(
            "seed-6", SEED_BASE + 6 * h, user_id="u-1", feature_tag="chat", cost_b=0.000001
        ),
        _tool_only_trace("seed-7", SEED_BASE + 7 * h),
        _expensive_trace("seed-8", SEED_BASE + 8 * h),
        sample_trace("seed-9", SEED_BASE + 9 * h, user_id="u-4", feature_tag="chat"),
    ]


@pytest.fixture
def seed_traces() -> list[Trace]:
    return build_seed_traces()


@pytest.fixture
async def seeded_backend(seed_traces: list[Trace], tmp_path) -> AsyncIterator[SqliteBackend]:
    backend = SqliteBackend(tmp_path / "seeded.db")
    await backend.save_traces(seed_traces)
    yield backend
    await backend.close()


@pytest.fixture
async def seeded_server(
    seeded_backend: SqliteBackend, tmp_path
) -> AsyncIterator[httpx.AsyncClient]:
    """httpx AsyncClient over the real app, backed by the seeded SQLite DB."""
    app = create_app(seeded_backend, dashboard_dist=tmp_path / "no-dist")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
