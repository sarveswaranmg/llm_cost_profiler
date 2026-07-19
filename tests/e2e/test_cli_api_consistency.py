"""The CLI and the HTTP API must report identical numbers for the same DB.

`tokenlens report --json` and GET /api/aggregate both sit on top of
StorageBackend.aggregate — this test catches either surface drifting in
how it serializes or post-processes those results.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest
from tests.storage_utils import build_pipeline_trace
from typer.testing import CliRunner

from tokenlens.cli import app
from tokenlens.server.app import create_app
from tokenlens.storage.sqlite import SqliteBackend

runner = CliRunner()

# CLI --group-by value → API group_by value.
_CLI_TO_API_GROUP = {"feature": "feature_tag", "user": "user_id", "model": "model", "node": "node"}


@pytest.fixture(autouse=True)
def _no_pg(monkeypatch) -> None:
    monkeypatch.delenv("TOKENLENS_PG_DSN", raising=False)


@pytest.fixture
def shared_db(tmp_path) -> str:
    path = str(tmp_path / "shared.db")
    now = datetime.now(UTC)

    async def seed() -> None:
        backend = SqliteBackend(path)
        await backend.save_traces(
            [
                build_pipeline_trace("t-1", now - timedelta(hours=1)),
                build_pipeline_trace(
                    "t-2",
                    now - timedelta(hours=2),
                    user_id="u-2",
                    feature_tag="search",
                    cost_a=0.123456,
                ),
            ]
        )
        await backend.close()

    asyncio.run(seed())
    return path


def _api_get(db_path: str, path: str, params: dict[str, Any]) -> Any:
    async def go() -> Any:
        backend = SqliteBackend(db_path)
        try:
            app_ = create_app(backend)
            transport = httpx.ASGITransport(app=app_)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(path, params=params)
                assert resp.status_code == 200, resp.text
                return resp.json()
        finally:
            await backend.close()

    return asyncio.run(go())


@pytest.mark.parametrize("cli_group", sorted(_CLI_TO_API_GROUP))
def test_report_json_matches_api_aggregate(shared_db: str, cli_group: str) -> None:
    result = runner.invoke(app, ["report", "--db", shared_db, "--json", "--group-by", cli_group])
    assert result.exit_code == 0, result.output
    cli_payload = json.loads(result.output)

    api_entries = _api_get(shared_db, "/api/aggregate", {"group_by": _CLI_TO_API_GROUP[cli_group]})

    assert [e["key"] for e in cli_payload["entries"]] == [e["key"] for e in api_entries]
    for cli_e, api_e in zip(cli_payload["entries"], api_entries, strict=True):
        assert Decimal(cli_e["cost_usd"]) == Decimal(api_e["cost_usd"])
        assert cli_e["total_tokens"] == api_e["total_tokens"]
        assert cli_e["call_count"] == api_e["call_count"]
        assert Decimal(cli_e["avg_cost_per_call"]) == Decimal(api_e["avg_cost_per_call"])


def test_report_json_totals_match_api_overview(shared_db: str) -> None:
    result = runner.invoke(app, ["report", "--db", shared_db, "--json"])
    assert result.exit_code == 0, result.output
    totals = json.loads(result.output)["totals"]

    overview = _api_get(shared_db, "/api/stats/overview", {})

    assert Decimal(totals["cost_usd"]) == Decimal(overview["total_cost_usd"])
    assert totals["total_tokens"] == overview["total_tokens"]
    assert totals["traces"] == overview["trace_count"]
    assert totals["error_rate"] == overview["error_rate"]
    assert Decimal(totals["retry_waste_usd"]) == Decimal(overview["retry_waste_usd"])
