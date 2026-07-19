"""Postgres storage backend.

StorageBackend implementation using asyncpg, suitable for production
deployments with concurrent writers/readers. The DSN comes from the
constructor or the TOKENLENS_PG_DSN env var. Requires the `postgres` extra:
pip install "tokenlens[postgres]".

Datetimes are stored as TIMESTAMPTZ; span metadata as JSONB (bound/read as
JSON text so no custom codecs are needed).
"""

import os
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    import asyncpg
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgresBackend requires the 'postgres' extra. "
        "Install with: pip install 'tokenlens[postgres]'"
    ) from exc

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

DSN_ENV_VAR = "TOKENLENS_PG_DSN"
SCHEMA_VERSION = 1

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS tokenlens_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id       TEXT PRIMARY KEY,
    root_name      TEXT NOT NULL,
    started_at     TIMESTAMPTZ NOT NULL,
    total_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_tokens   BIGINT NOT NULL DEFAULT 0,
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
    start_time         TIMESTAMPTZ NOT NULL,
    end_time           TIMESTAMPTZ,
    status             TEXT NOT NULL,
    error_message      TEXT,
    metadata           JSONB NOT NULL DEFAULT '{}',
    model_name         TEXT,
    provider           TEXT,
    input_tokens       BIGINT,
    output_tokens      BIGINT,
    cache_read_tokens  BIGINT,
    cache_write_tokens BIGINT,
    prompt_preview     TEXT,
    retry_index        INTEGER NOT NULL DEFAULT 0,
    cost_usd           DOUBLE PRECISION,
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

# Same micro-dollar trick as the SQLite backend: sum cost as exact integers
# so the result matches attribution.cost_by's Decimal math bit-for-bit.
_TOTAL_TOKENS_SQL = (
    "COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)"
    " + COALESCE(cache_read_tokens, 0) + COALESCE(cache_write_tokens, 0)"
)
_COST_MICROS_SQL = "ROUND((COALESCE(cost_usd, 0) * 1000000)::numeric)::bigint"


def _placeholders(start: int = 1) -> Any:
    """Yield "$1", "$2", ... — asyncpg's positional parameter style."""
    n = start - 1

    def next_placeholder() -> str:
        nonlocal n
        n += 1
        return f"${n}"

    return next_placeholder


class PostgresBackend:
    """StorageBackend backed by Postgres via an asyncpg pool."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.environ.get(DSN_ENV_VAR)
        if not self.dsn:
            raise ValueError(
                f"PostgresBackend needs a DSN: pass one or set {DSN_ENV_VAR} "
                "(e.g. postgresql://user:pass@localhost/tokenlens)"
            )
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> "asyncpg.Pool":
        if self._pool is None:
            pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
            async with pool.acquire() as conn:
                await self._migrate(conn)
            self._pool = pool
        return self._pool

    async def _migrate(self, conn: "asyncpg.Connection") -> None:
        # Idempotent CREATEs bring any database to the current schema; the
        # meta row records the version for future ALTER-based migrations.
        await conn.execute(_CREATE_TABLES)
        row = await conn.fetchval("SELECT value FROM tokenlens_meta WHERE key = 'schema_version'")
        if row is not None and int(row) > SCHEMA_VERSION:
            raise RuntimeError(
                f"database has schema version {row}, newer than this tokenlens "
                f"supports ({SCHEMA_VERSION}); upgrade tokenlens"
            )
        await conn.execute(
            "INSERT INTO tokenlens_meta (key, value) VALUES ('schema_version', $1) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            str(SCHEMA_VERSION),
        )

    async def save_traces(self, traces: list[Trace]) -> None:
        if not traces:
            return
        pool = await self._get_pool()
        trace_ph = ", ".join(f"${i}" for i in range(1, len(TRACE_COLUMNS) + 1))
        assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in TRACE_COLUMNS[1:])
        trace_sql = (
            f"INSERT INTO traces ({', '.join(TRACE_COLUMNS)}) VALUES ({trace_ph}) "
            f"ON CONFLICT (trace_id) DO UPDATE SET {assignments}"
        )
        span_values = ", ".join(
            f"${i}::jsonb" if col == "metadata" else f"${i}"
            for i, col in enumerate(SPAN_COLUMNS, start=1)
        )
        span_sql = f"INSERT INTO spans ({', '.join(SPAN_COLUMNS)}) VALUES ({span_values})"
        async with pool.acquire() as conn, conn.transaction():
            for trace in traces:
                await conn.execute(trace_sql, *trace_to_row(trace, datetimes_as_iso=False))
                await conn.execute("DELETE FROM spans WHERE trace_id = $1", trace.trace_id)
                await conn.executemany(
                    span_sql,
                    [span_to_row(span, datetimes_as_iso=False) for span in trace.spans],
                )

    async def get_trace(self, trace_id: str) -> Trace | None:
        pool = await self._get_pool()
        select_cols = ", ".join("metadata::text" if c == "metadata" else c for c in SPAN_COLUMNS)
        rows = await pool.fetch(
            f"SELECT {select_cols} FROM spans WHERE trace_id = $1 ORDER BY start_time, span_id",
            trace_id,
        )
        if not rows:
            return None
        return Trace.from_spans(trace_id, [span_from_row(tuple(row)) for row in rows])

    async def list_traces(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: TraceFilters | None = None,
    ) -> list[TraceSummary]:
        pool = await self._get_pool()
        placeholder = _placeholders()
        conditions, params = trace_where(filters, placeholder, datetimes_as_iso=False)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = (
            f"SELECT {SUMMARY_SELECT_SQL} FROM traces {where} "
            f"ORDER BY started_at DESC, trace_id LIMIT {placeholder()} OFFSET {placeholder()}"
        )
        rows = await pool.fetch(sql, *params, limit, offset)
        return [summary_from_row(tuple(row)) for row in rows]

    async def aggregate(
        self,
        group_by: str,
        filters: TraceFilters | None = None,
    ) -> list[CostBreakdownEntry]:
        column = group_by_column(group_by)
        pool = await self._get_pool()
        conditions, params = trace_where(filters, _placeholders(), datetimes_as_iso=False)
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
        rows = await pool.fetch(sql, *params)
        return breakdown_from_rows([tuple(row) for row in rows])

    async def overview(self, filters: TraceFilters | None = None) -> StatsOverview:
        pool = await self._get_pool()
        conditions, params = trace_where(filters, _placeholders(), datetimes_as_iso=False)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        trace_row = await pool.fetchrow(
            f"SELECT COUNT(*), COALESCE(SUM(total_tokens), 0) FROM traces {where}", *params
        )
        trace_count, total_tokens = trace_row[0], trace_row[1]

        span_scope = f"trace_id IN (SELECT trace_id FROM traces {where})"
        span_row = await pool.fetchrow(
            f"SELECT COALESCE(SUM({_COST_MICROS_SQL}), 0), "
            f"COALESCE(SUM(CASE WHEN retry_index > 0 THEN {_COST_MICROS_SQL} ELSE 0 END), 0), "
            "COUNT(DISTINCT CASE WHEN status = 'ERROR' THEN trace_id END) "
            f"FROM spans WHERE {span_scope}",
            *params,
        )
        cost_micros, retry_micros, error_traces = span_row[0], span_row[1], span_row[2]

        top_rows = await pool.fetch(
            f"SELECT {SUMMARY_SELECT_SQL} FROM traces {where} "
            "ORDER BY total_cost_usd DESC, trace_id LIMIT 5",
            *params,
        )

        return StatsOverview(
            total_cost_usd=micros_to_usd(cost_micros),
            total_tokens=int(total_tokens),
            trace_count=int(trace_count),
            error_rate=(error_traces / trace_count) if trace_count else 0.0,
            retry_waste_usd=micros_to_usd(retry_micros),
            top_traces=[summary_from_row(tuple(row)) for row in top_rows],
        )

    async def prune(self, older_than_days: int = 30) -> int:
        pool = await self._get_pool()
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM spans WHERE trace_id IN "
                "(SELECT trace_id FROM traces WHERE started_at < $1)",
                cutoff,
            )
            result = await conn.execute("DELETE FROM traces WHERE started_at < $1", cutoff)
        # asyncpg returns a status tag like "DELETE 3".
        return int(result.rsplit(" ", 1)[-1])

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:  # pragma: no cover - debugging nicety
        return "PostgresBackend(...)"


__all__ = ["DSN_ENV_VAR", "SCHEMA_VERSION", "PostgresBackend"]
