"""SQLite storage backend.

Default StorageBackend implementation — zero configuration, single-file
database at ~/.tokenlens/traces.db (override via constructor or the
TOKENLENS_DB_PATH env var), suitable for local development and single-user
usage. Datetimes are stored as UTC ISO-8601 TEXT (lexicographic order ==
chronological order); span metadata is stored as a JSON TEXT column.
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

from tokenlens.core.span import Trace
from tokenlens.cost.attribution import CostBreakdownEntry
from tokenlens.storage.base import (
    SPAN_COLUMNS,
    SUMMARY_SELECT_SQL,
    TRACE_COLUMNS,
    StatsOverview,
    TraceFilters,
    TraceSummary,
    breakdown_from_rows,
    group_by_column,
    micros_to_usd,
    span_from_row,
    span_to_row,
    summary_from_row,
    trace_to_row,
    trace_where,
)

SCHEMA_VERSION = 1

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id       TEXT PRIMARY KEY,
    root_name      TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    total_tokens   INTEGER NOT NULL DEFAULT 0,
    user_id        TEXT,
    feature_tag    TEXT,
    session_id     TEXT
);

CREATE TABLE IF NOT EXISTS spans (
    span_id            TEXT PRIMARY KEY,
    trace_id           TEXT NOT NULL,
    parent_span_id     TEXT,
    name               TEXT NOT NULL,
    kind               TEXT NOT NULL,
    start_time         TEXT NOT NULL,
    end_time           TEXT,
    status             TEXT NOT NULL,
    error_message      TEXT,
    metadata           TEXT NOT NULL DEFAULT '{}',
    model_name         TEXT,
    provider           TEXT,
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    cache_read_tokens  INTEGER,
    cache_write_tokens INTEGER,
    prompt_preview     TEXT,
    retry_index        INTEGER NOT NULL DEFAULT 0,
    cost_usd           REAL,
    user_id            TEXT,
    feature_tag        TEXT,
    session_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_traces_started_at ON traces(started_at);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent_span_id ON spans(parent_span_id);
CREATE INDEX IF NOT EXISTS idx_spans_model_name ON spans(model_name);
CREATE INDEX IF NOT EXISTS idx_spans_user_id ON spans(user_id);
CREATE INDEX IF NOT EXISTS idx_spans_feature_tag ON spans(feature_tag);
CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time);
"""

# COALESCE-heavy on purpose: totals must match attribution.cost_by, which
# treats missing cost/tokens as 0. Cost is summed as integer micro-dollars
# so SQLite float addition can't drift from the pure-Python Decimal sum.
_TOTAL_TOKENS_SQL = (
    "COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)"
    " + COALESCE(cache_read_tokens, 0) + COALESCE(cache_write_tokens, 0)"
)
_COST_MICROS_SQL = "CAST(ROUND(COALESCE(cost_usd, 0) * 1000000) AS INTEGER)"


def default_db_path() -> Path:
    env = os.environ.get("TOKENLENS_DB_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".tokenlens" / "traces.db"


class SqliteBackend:
    """Zero-config StorageBackend backed by a single SQLite file."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path).expanduser() if db_path is not None else default_db_path()
        self._db: aiosqlite.Connection | None = None
        self._connect_lock = asyncio.Lock()

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            async with self._connect_lock:
                if self._db is None:
                    self.db_path.parent.mkdir(parents=True, exist_ok=True)
                    db = await aiosqlite.connect(self.db_path)
                    await db.execute("PRAGMA journal_mode = WAL")
                    await self._migrate(db)
                    self._db = db
        return self._db

    async def _migrate(self, db: aiosqlite.Connection) -> None:
        # The CREATE statements are idempotent, so simply running them brings
        # a fresh or current database to SCHEMA_VERSION. The user_version
        # pragma records what the file is at, so future versions know which
        # ALTERs to apply.
        async with db.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()
        version = row[0] if row else 0
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.db_path} has schema version {version}, newer than this "
                f"tokenlens supports ({SCHEMA_VERSION}); upgrade tokenlens"
            )
        await db.executescript(_CREATE_TABLES)
        await db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await db.commit()

    async def save_traces(self, traces: list[Trace]) -> None:
        if not traces:
            return
        db = await self._conn()
        trace_sql = (
            f"INSERT OR REPLACE INTO traces ({', '.join(TRACE_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(TRACE_COLUMNS))})"
        )
        span_sql = (
            f"INSERT INTO spans ({', '.join(SPAN_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(SPAN_COLUMNS))})"
        )
        for trace in traces:
            await db.execute(trace_sql, trace_to_row(trace, datetimes_as_iso=True))
            # Re-saving a trace replaces its spans wholesale, keeping
            # save_traces idempotent.
            await db.execute("DELETE FROM spans WHERE trace_id = ?", (trace.trace_id,))
            await db.executemany(
                span_sql,
                [span_to_row(span, datetimes_as_iso=True) for span in trace.spans],
            )
        await db.commit()

    async def get_trace(self, trace_id: str) -> Trace | None:
        db = await self._conn()
        sql = (
            f"SELECT {', '.join(SPAN_COLUMNS)} FROM spans "
            "WHERE trace_id = ? ORDER BY start_time, span_id"
        )
        async with db.execute(sql, (trace_id,)) as cursor:
            rows = await cursor.fetchall()
        if not rows:
            return None
        return Trace.from_spans(trace_id, [span_from_row(row) for row in rows])

    async def list_traces(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: TraceFilters | None = None,
    ) -> list[TraceSummary]:
        db = await self._conn()
        conditions, params = trace_where(filters, lambda: "?", datetimes_as_iso=True)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = (
            f"SELECT {SUMMARY_SELECT_SQL} FROM traces {where} "
            "ORDER BY started_at DESC, trace_id LIMIT ? OFFSET ?"
        )
        async with db.execute(sql, (*params, limit, offset)) as cursor:
            rows = await cursor.fetchall()
        return [summary_from_row(row) for row in rows]

    async def aggregate(
        self,
        group_by: str,
        filters: TraceFilters | None = None,
    ) -> list[CostBreakdownEntry]:
        column = group_by_column(group_by)
        db = await self._conn()
        conditions, params = trace_where(filters, lambda: "?", datetimes_as_iso=True)
        span_filter = ""
        if conditions:
            span_filter = (
                " AND spans.trace_id IN "
                f"(SELECT trace_id FROM traces WHERE {' AND '.join(conditions)})"
            )
        sql = (
            f"SELECT spans.{column}, SUM({_COST_MICROS_SQL}), "
            f"SUM({_TOTAL_TOKENS_SQL}), COUNT(*) "
            f"FROM spans WHERE spans.{column} IS NOT NULL{span_filter} "
            f"GROUP BY spans.{column}"
        )
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return breakdown_from_rows([tuple(row) for row in rows])

    async def overview(self, filters: TraceFilters | None = None) -> StatsOverview:
        db = await self._conn()
        conditions, params = trace_where(filters, lambda: "?", datetimes_as_iso=True)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        async with db.execute(
            f"SELECT COUNT(*), COALESCE(SUM(total_tokens), 0) FROM traces {where}", params
        ) as cursor:
            trace_count, total_tokens = await cursor.fetchone()  # type: ignore[misc]

        span_scope = f"trace_id IN (SELECT trace_id FROM traces {where})"
        async with db.execute(
            f"SELECT COALESCE(SUM({_COST_MICROS_SQL}), 0), "
            f"COALESCE(SUM(CASE WHEN retry_index > 0 THEN {_COST_MICROS_SQL} ELSE 0 END), 0), "
            "COUNT(DISTINCT CASE WHEN status = 'ERROR' THEN trace_id END) "
            f"FROM spans WHERE {span_scope}",
            params,
        ) as cursor:
            cost_micros, retry_micros, error_traces = await cursor.fetchone()  # type: ignore[misc]

        async with db.execute(
            f"SELECT {SUMMARY_SELECT_SQL} FROM traces {where} "
            "ORDER BY total_cost_usd DESC, trace_id LIMIT 5",
            params,
        ) as cursor:
            top_rows = await cursor.fetchall()

        return StatsOverview(
            total_cost_usd=micros_to_usd(cost_micros),
            total_tokens=int(total_tokens),
            trace_count=int(trace_count),
            error_rate=(error_traces / trace_count) if trace_count else 0.0,
            retry_waste_usd=micros_to_usd(retry_micros),
            top_traces=[summary_from_row(row) for row in top_rows],
        )

    async def prune(self, older_than_days: int = 30) -> int:
        db = await self._conn()
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        await db.execute(
            "DELETE FROM spans WHERE trace_id IN "
            "(SELECT trace_id FROM traces WHERE started_at < ?)",
            (cutoff,),
        )
        cursor = await db.execute("DELETE FROM traces WHERE started_at < ?", (cutoff,))
        await db.commit()
        return cursor.rowcount

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def __repr__(self) -> str:  # pragma: no cover - debugging nicety
        return f"SqliteBackend(db_path={str(self.db_path)!r})"


__all__ = ["SCHEMA_VERSION", "SqliteBackend", "default_db_path"]
