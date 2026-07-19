"""CLI tests via typer's CliRunner against a seeded SQLite database."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from tests.storage_utils import build_pipeline_trace
from typer.testing import CliRunner

from tokenlens.cli import app
from tokenlens.storage.sqlite import SqliteBackend

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_pg(monkeypatch) -> None:
    # CLI storage resolution must not pick up a Postgres DSN from the host env.
    monkeypatch.delenv("TOKENLENS_PG_DSN", raising=False)


@pytest.fixture
def seeded_db(tmp_path) -> str:
    path = str(tmp_path / "cli.db")
    now = datetime.now(UTC)

    async def seed() -> None:
        backend = SqliteBackend(path)
        await backend.save_traces(
            [
                build_pipeline_trace("cli-1", now - timedelta(hours=1)),
                build_pipeline_trace(
                    "cli-2",
                    now - timedelta(days=2),
                    user_id="u-2",
                    feature_tag="search",
                    cost_a=0.4,
                    cost_b=0.2,
                ),
                build_pipeline_trace("cli-old", now - timedelta(days=60)),
            ]
        )
        await backend.close()

    asyncio.run(seed())
    return path


def test_report_table(seeded_db: str) -> None:
    result = runner.invoke(app, ["report", "--db", seeded_db, "--group-by", "model"])
    assert result.exit_code == 0
    assert "cost by model" in result.output
    assert "gpt-4o" in result.output
    assert "retry waste" in result.output


def test_report_since_window(seeded_db: str) -> None:
    result = runner.invoke(app, ["report", "--db", seeded_db, "--since", "1d"])
    assert result.exit_code == 0
    # Only cli-1 is newer than a day.
    assert "1 traces" in result.output


def test_report_json(seeded_db: str) -> None:
    result = runner.invoke(app, ["report", "--db", seeded_db, "--group-by", "user", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["group_by"] == "user"
    assert payload["totals"]["traces"] == 3
    keys = {e["key"] for e in payload["entries"]}
    assert keys == {"u-1", "u-2"}
    # Money is exact decimal strings.
    assert payload["totals"]["retry_waste_usd"] == "0.000045"


def test_report_empty_db(tmp_path) -> None:
    result = runner.invoke(app, ["report", "--db", str(tmp_path / "empty.db")])
    assert result.exit_code == 1
    assert "No traces yet" in result.output
    assert "demo_langgraph_agent" in result.output


def test_report_bad_window(seeded_db: str) -> None:
    result = runner.invoke(app, ["report", "--db", seeded_db, "--since", "banana"])
    assert result.exit_code == 2
    assert "Can't parse time window" in result.output


def test_top(seeded_db: str) -> None:
    result = runner.invoke(app, ["top", "--db", seeded_db, "--n", "2"])
    assert result.exit_code == 0
    assert "top 2 traces" in result.output
    assert "cli-2" in result.output  # most expensive
    assert "█" in result.output
    assert "cli-old" not in result.output  # cut off by --n


def test_trace_tree(seeded_db: str) -> None:
    result = runner.invoke(app, ["trace", "cli-1", "--db", seeded_db])
    assert result.exit_code == 0
    for expected in ("pipeline", "draft_answer", "refine_answer", "retry #1", "rate limited"):
        assert expected in result.output


def test_trace_not_found(seeded_db: str) -> None:
    result = runner.invoke(app, ["trace", "nope", "--db", seeded_db])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_prune(seeded_db: str) -> None:
    result = runner.invoke(app, ["prune", "--db", seeded_db, "--older-than", "30d", "--yes"])
    assert result.exit_code == 0
    assert "Pruned 1 trace" in result.output

    result = runner.invoke(app, ["top", "--db", seeded_db, "--n", "10"])
    assert "cli-old" not in result.output


def test_prune_asks_for_confirmation(seeded_db: str) -> None:
    result = runner.invoke(app, ["prune", "--db", seeded_db], input="n\n")
    assert result.exit_code != 0  # aborted

    result = runner.invoke(app, ["top", "--db", seeded_db, "--n", "10"])
    assert "cli-old" in result.output  # nothing deleted


def test_pricing() -> None:
    result = runner.invoke(app, ["pricing"])
    assert result.exit_code == 0
    assert "pricing table" in result.output
    assert "gpt-4o" in result.output
    assert "aliases" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
