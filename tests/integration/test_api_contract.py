"""API contract tests over the seeded server: every route, every group_by,
pagination stability, 404/422 behavior, and agreement between HTTP
responses and the pure-Python attribution math on the same seed data."""

from decimal import Decimal
from typing import Any

import httpx
import pytest

from tokenlens.core.span import SpanStatus, Trace
from tokenlens.cost.attribution import cost_by, retry_waste

# API group_by value → attribution key.
_GROUP_KEYS = {
    "user_id": "user_id",
    "feature_tag": "feature_tag",
    "model": "model_name",
    "node": "node_name",
    "kind": "kind",
}


async def test_list_traces_returns_all_seeds_newest_first(
    seeded_server: httpx.AsyncClient, seed_traces: list[Trace]
) -> None:
    resp = await seeded_server.get("/api/traces", params={"limit": 50})

    assert resp.status_code == 200
    page = resp.json()
    assert page["count"] == len(seed_traces)
    ids = [t["trace_id"] for t in page["traces"]]
    assert ids[0] == "seed-9"  # newest first
    assert set(ids) == {t.trace_id for t in seed_traces}

    expected_errors = {
        t.trace_id for t in seed_traces if any(s.status == SpanStatus.ERROR for s in t.spans)
    }
    flagged = {t["trace_id"] for t in page["traces"] if t["has_error"]}
    assert flagged == expected_errors == {"seed-4", "seed-5", "seed-6"}


async def test_pagination_is_stable_with_no_duplicates_or_gaps(
    seeded_server: httpx.AsyncClient, seed_traces: list[Trace]
) -> None:
    seen: list[str] = []
    for offset in range(0, 12, 3):
        resp = await seeded_server.get("/api/traces", params={"limit": 3, "offset": offset})
        assert resp.status_code == 200
        seen.extend(t["trace_id"] for t in resp.json()["traces"])

    assert len(seen) == len(set(seen)) == len(seed_traces)  # no dupes, no gaps


@pytest.mark.parametrize(
    ("params", "expected_ids"),
    [
        ({"user_id": "u-0"}, {"seed-0", "seed-5"}),
        ({"feature_tag": "search"}, {"seed-2", "seed-3"}),
        ({"model": "gpt-4.1"}, {"seed-5"}),
        ({"model": "claude-haiku-4-5"}, {"seed-0", "seed-1", "seed-2", "seed-3", "seed-9"}),
        ({"min_cost": 1.0}, {"seed-8"}),
        ({"user_id": "u-1", "feature_tag": "chat"}, {"seed-1", "seed-6"}),  # filters AND together
        ({"since": "2026-07-10T19:30:00Z"}, {"seed-8", "seed-9"}),
        ({"until": "2026-07-10T13:00:00Z"}, {"seed-0", "seed-1"}),
    ],
)
async def test_trace_filters(
    seeded_server: httpx.AsyncClient, params: dict[str, Any], expected_ids: set[str]
) -> None:
    resp = await seeded_server.get("/api/traces", params=params)

    assert resp.status_code == 200
    assert {t["trace_id"] for t in resp.json()["traces"]} == expected_ids


async def test_get_trace_returns_the_exact_nested_tree(
    seeded_server: httpx.AsyncClient, seed_traces: list[Trace]
) -> None:
    resp = await seeded_server.get("/api/traces/seed-0")

    assert resp.status_code == 200
    tree = resp.json()
    assert tree["name"] == "handle_request"
    assert {c["name"] for c in tree["children"]} == {
        "vector_search",
        "draft_answer",
        "web_search",
    }
    tool = next(c for c in tree["children"] if c["name"] == "web_search")
    assert [c["name"] for c in tool["children"]] == ["summarize_results"]

    def count(node: dict[str, Any]) -> int:
        return 1 + sum(count(c) for c in node["children"])

    assert count(tree) == len(seed_traces[0].spans)


async def test_unknown_trace_id_is_a_clean_404(seeded_server: httpx.AsyncClient) -> None:
    for path in ("/api/traces/nope", "/api/traces/nope/flame"):
        resp = await seeded_server.get(path)
        assert resp.status_code == 404
        body = resp.json()
        assert "nope" in body["detail"]  # a clean error body, not a stack trace


async def test_flame_output_is_valid_d3_input(
    seeded_server: httpx.AsyncClient, seed_traces: list[Trace]
) -> None:
    resp = await seeded_server.get("/api/traces/seed-0/flame")

    assert resp.status_code == 200
    flame = resp.json()

    def validate(node: dict[str, Any]) -> None:
        assert set(node) >= {"name", "value", "children"}
        assert isinstance(node["value"], int)  # integer micro-dollars
        assert node["value"] >= sum(c["value"] for c in node["children"])
        for child in node["children"]:
            validate(child)

    validate(flame)
    expected_total = sum(
        int(Decimal(str(s.cost_usd)) * 1_000_000)
        for s in seed_traces[0].spans
        if s.cost_usd is not None
    )
    assert flame["value"] == expected_total


@pytest.mark.parametrize("group_by", sorted(_GROUP_KEYS))
async def test_aggregate_matches_pure_python_attribution(
    seeded_server: httpx.AsyncClient, seed_traces: list[Trace], group_by: str
) -> None:
    resp = await seeded_server.get("/api/aggregate", params={"group_by": group_by})

    assert resp.status_code == 200
    entries = resp.json()
    expected = cost_by(seed_traces, _GROUP_KEYS[group_by])

    assert [e["key"] for e in entries] == [e.key for e in expected]  # same order too
    for got, want in zip(entries, expected, strict=True):
        assert Decimal(got["cost_usd"]) == want.cost_usd
        assert got["total_tokens"] == want.total_tokens
        assert got["call_count"] == want.call_count
        assert Decimal(got["avg_cost_per_call"]) == want.avg_cost_per_call


async def test_overview_totals_match_seed_data(
    seeded_server: httpx.AsyncClient, seed_traces: list[Trace]
) -> None:
    resp = await seeded_server.get("/api/stats/overview")

    assert resp.status_code == 200
    stats = resp.json()

    expected_cost = sum(
        (Decimal(str(s.cost_usd)) for t in seed_traces for s in t.spans if s.cost_usd is not None),
        Decimal(0),
    )
    assert Decimal(stats["total_cost_usd"]) == expected_cost
    assert stats["total_tokens"] == sum(t.total_tokens() for t in seed_traces)
    assert stats["trace_count"] == len(seed_traces)
    assert stats["error_rate"] == pytest.approx(3 / 10)
    assert Decimal(stats["retry_waste_usd"]) == retry_waste(seed_traces)

    top = stats["top_traces"]
    assert len(top) == 5
    assert top[0]["trace_id"] == "seed-8"  # the $2.50 batch job
    costs = [Decimal(t["total_cost_usd"]) for t in top]
    assert costs == sorted(costs, reverse=True)


async def test_overview_since_narrows_the_window(seeded_server: httpx.AsyncClient) -> None:
    resp = await seeded_server.get("/api/stats/overview", params={"since": "2026-07-10T19:30:00Z"})

    assert resp.status_code == 200
    assert resp.json()["trace_count"] == 2  # seed-8 and seed-9 only


async def test_pricing_endpoint_serves_the_active_table(
    seeded_server: httpx.AsyncClient,
) -> None:
    resp = await seeded_server.get("/api/pricing")

    assert resp.status_code == 200
    table = resp.json()
    assert table["version"]
    assert "gpt-4o-mini" in table["models"]
    assert Decimal(table["models"]["gpt-4o-mini"]["input_per_mtok"]) == Decimal("0.15")


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},  # below ge=1
        {"limit": 501},  # above le=500
        {"limit": "many"},
        {"offset": -1},
        {"min_cost": -2},
        {"min_cost": "cheap"},
        {"since": "yesterday-ish"},
    ],
)
async def test_malformed_trace_query_params_are_422_not_500(
    seeded_server: httpx.AsyncClient, params: dict[str, Any]
) -> None:
    resp = await seeded_server.get("/api/traces", params=params)

    assert resp.status_code == 422
    assert "detail" in resp.json()


async def test_malformed_aggregate_params_are_422_not_500(
    seeded_server: httpx.AsyncClient,
) -> None:
    assert (await seeded_server.get("/api/aggregate")).status_code == 422  # missing group_by
    resp = await seeded_server.get("/api/aggregate", params={"group_by": "vibes"})
    assert resp.status_code == 422
    resp = await seeded_server.get("/api/stats/overview", params={"since": "not-a-time"})
    assert resp.status_code == 422
