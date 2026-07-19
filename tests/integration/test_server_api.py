"""API server tests: httpx AsyncClient against a seeded SQLite backend,
plus the /ws/live WebSocket via Starlette's TestClient."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest
from starlette.testclient import TestClient
from tests.storage_utils import build_pipeline_trace

from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import Span, SpanKind, Trace
from tokenlens.cost.attribution import cost_by, retry_waste
from tokenlens.server.app import create_app
from tokenlens.storage.sqlite import SqliteBackend

BASE = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


def build_simple_trace(trace_id: str, started_at: datetime) -> Trace:
    """A clean two-span trace: no errors, no retries."""
    root = Span(
        trace_id=trace_id,
        name="summarize",
        kind=SpanKind.CHAIN,
        start_time=started_at,
        end_time=started_at + timedelta(seconds=1),
        user_id="u-1",
        feature_tag="search",
        session_id="sess-9",
    )
    call = Span(
        trace_id=trace_id,
        parent_span_id=root.span_id,
        name="summarize_call",
        kind=SpanKind.LLM_CALL,
        start_time=started_at + timedelta(milliseconds=50),
        end_time=started_at + timedelta(milliseconds=900),
        model_name="gpt-4o-mini",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
        cost_usd=1.25,
        user_id="u-1",
        feature_tag="search",
        session_id="sess-9",
    )
    return Trace.from_spans(trace_id, [root, call])


def seed_traces() -> list[Trace]:
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
        build_simple_trace("trace-3", BASE + timedelta(hours=2)),
    ]


@pytest.fixture
async def seeded_backend(tmp_path) -> AsyncIterator[SqliteBackend]:
    backend = SqliteBackend(tmp_path / "api.db")
    await backend.save_traces(seed_traces())
    yield backend
    await backend.close()


@pytest.fixture
async def client(seeded_backend: SqliteBackend, tmp_path) -> AsyncIterator[httpx.AsyncClient]:
    # Point dashboard_dist at a nonexistent dir so tests exercise the
    # placeholder branch regardless of whether dashboard/dist is built.
    app = create_app(seeded_backend, dashboard_dist=tmp_path / "no-dist")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_serves_built_dashboard_when_dist_exists(
    seeded_backend: SqliteBackend, tmp_path
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>tokenlens</title>built!")
    app = create_app(seeded_backend, dashboard_dist=dist)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/")
        assert resp.status_code == 200
        assert "built!" in resp.text
        # API routes still win over the static mount.
        assert (await c.get("/api/pricing")).status_code == 200


async def test_list_traces_page(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/traces")
    assert resp.status_code == 200
    page = resp.json()
    assert page["count"] == 3 and page["limit"] == 50 and page["offset"] == 0
    assert [t["trace_id"] for t in page["traces"]] == ["trace-3", "trace-2", "trace-1"]
    first = page["traces"][0]
    assert first["root_name"] == "summarize"
    assert first["total_cost_usd"] == pytest.approx(1.25)
    assert first["total_tokens"] == 150
    assert first["user_id"] == "u-1" and first["feature_tag"] == "search"
    assert first["has_error"] is False  # trace-3 is clean
    assert page["traces"][2]["has_error"] is True  # trace-1 has an ERROR retry span


async def test_list_traces_pagination_and_filters(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/traces", params={"limit": 1, "offset": 1})
    assert [t["trace_id"] for t in resp.json()["traces"]] == ["trace-2"]

    resp = await client.get("/api/traces", params={"user_id": "u-2"})
    assert [t["trace_id"] for t in resp.json()["traces"]] == ["trace-2"]

    resp = await client.get("/api/traces", params={"model": "gpt-4o"})
    assert [t["trace_id"] for t in resp.json()["traces"]] == ["trace-2", "trace-1"]

    resp = await client.get(
        "/api/traces",
        params={"since": (BASE + timedelta(minutes=30)).isoformat(), "min_cost": 1.0},
    )
    assert [t["trace_id"] for t in resp.json()["traces"]] == ["trace-3"]

    resp = await client.get("/api/traces", params={"limit": 0})
    assert resp.status_code == 422  # ge=1


async def test_get_trace_tree(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/traces/trace-1")
    assert resp.status_code == 200
    tree = resp.json()
    assert tree["name"] == "pipeline" and tree["parent_span_id"] is None
    children = {c["name"] for c in tree["children"]}
    assert children == {"retrieve_docs", "draft_answer", "refine"}
    refine = next(c for c in tree["children"] if c["name"] == "refine")
    assert [c["name"] for c in refine["children"]] == ["refine_answer"]
    llm = next(c for c in tree["children"] if c["kind"] == "LLM_CALL")
    assert llm["model_name"] == "gpt-4o"
    assert llm["total_tokens"] == 1500 and llm["duration_ms"] == pytest.approx(1600.0)
    assert tree["metadata"] == {"env": "test", "nested": {"tags": ["a", "b"]}}


async def test_get_trace_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/traces/nope")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


async def test_trace_flame(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/traces/trace-1/flame")
    assert resp.status_code == 200
    flame = resp.json()
    assert flame["name"] == "pipeline"
    # Inclusive root value = whole-trace cost in micro-dollars.
    assert flame["value"] == 300015
    assert flame["value"] >= sum(c["value"] for c in flame["children"])
    assert flame["data"]["kind"] == "CHAIN" and flame["data"]["self_cost_usd"] is None
    retry = next(c for c in flame["children"] if c["data"]["retry_index"] == 1)
    assert retry["data"]["status"] == "ERROR"

    assert (await client.get("/api/traces/nope/flame")).status_code == 404


async def test_aggregate_matches_attribution(client: httpx.AsyncClient) -> None:
    for api_key, storage_key in [
        ("model", "model_name"),
        ("node", "node_name"),
        ("user_id", "user_id"),
        ("feature_tag", "feature_tag"),
        ("kind", "kind"),
    ]:
        resp = await client.get("/api/aggregate", params={"group_by": api_key})
        assert resp.status_code == 200
        expected = cost_by(seed_traces(), storage_key)
        got = resp.json()
        assert [e["key"] for e in got] == [e.key for e in expected]
        for g, e in zip(got, expected, strict=True):
            assert Decimal(g["cost_usd"]) == e.cost_usd
            assert g["total_tokens"] == e.total_tokens
            assert g["call_count"] == e.call_count
            assert Decimal(g["avg_cost_per_call"]) == e.avg_cost_per_call


async def test_aggregate_filters_and_validation(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/aggregate", params={"group_by": "model", "user_id": "u-2"})
    assert {e["key"] for e in resp.json()} == {"gpt-4o", "claude-sonnet-5"}

    assert (await client.get("/api/aggregate", params={"group_by": "nope"})).status_code == 422
    assert (await client.get("/api/aggregate")).status_code == 422  # group_by required


async def test_stats_overview(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/stats/overview")
    assert resp.status_code == 200
    stats = resp.json()

    traces = seed_traces()
    expected_cost = sum(
        (Decimal(str(s.cost_usd)) for t in traces for s in t.spans if s.cost_usd is not None),
        Decimal(0),
    )
    assert Decimal(stats["total_cost_usd"]) == expected_cost
    assert stats["total_tokens"] == sum(t.total_tokens() for t in traces)
    assert stats["trace_count"] == 3
    # trace-1 and trace-2 contain an ERROR retry span; trace-3 is clean.
    assert stats["error_rate"] == pytest.approx(2 / 3)
    assert Decimal(stats["retry_waste_usd"]) == retry_waste(traces)
    assert [t["trace_id"] for t in stats["top_traces"]] == ["trace-3", "trace-1", "trace-2"]

    resp = await client.get(
        "/api/stats/overview", params={"since": (BASE + timedelta(minutes=90)).isoformat()}
    )
    stats = resp.json()
    assert stats["trace_count"] == 1 and stats["error_rate"] == 0.0
    assert Decimal(stats["retry_waste_usd"]) == 0


async def test_pricing(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/pricing")
    assert resp.status_code == 200
    table = resp.json()
    assert table["source"] == "<bundled>"
    assert "gpt-4o" in table["models"]
    entry = table["models"]["gpt-4o"]
    assert entry["provider"] == "openai"
    assert Decimal(entry["input_per_mtok"]) > 0
    assert table["aliases"]["gpt-4o-2024-11-20"] == "gpt-4o"


async def test_openapi_and_placeholder(client: httpx.AsyncClient) -> None:
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert {
        "/api/traces",
        "/api/traces/{trace_id}",
        "/api/traces/{trace_id}/flame",
        "/api/aggregate",
        "/api/stats/overview",
        "/api/pricing",
    } <= set(paths)

    resp = await client.get("/")
    assert resp.status_code == 200
    assert "dashboard hasn't been built" in resp.text


async def test_cors_for_localhost_dev(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/pricing", headers={"Origin": "http://localhost:5173"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_ws_live_pushes_flushed_traces(tmp_path, collector: TraceCollector) -> None:
    backend = SqliteBackend(tmp_path / "live.db")
    app = create_app(backend, collector=collector)

    with TestClient(app) as tc, tc.websocket_connect("/ws/live") as websocket:
        trace = build_pipeline_trace("trace-live", datetime.now(UTC))
        for span in trace.spans:
            collector.record(span)
        collector.flush()

        message = websocket.receive_json()
        assert message["trace_id"] == "trace-live"
        assert message["root_name"] == "pipeline"
        assert message["total_tokens"] == trace.total_tokens()
        assert message["total_cost_usd"] == pytest.approx(trace.total_cost())
        assert message["has_error"] is True
