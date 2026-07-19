"""StorageBackend protocol.

Defines the interface every storage backend (SQLite, Postgres, ...) must
implement: writing completed traces, and querying traces/spans back out for
the server and CLI. Also hosts the shared row <-> model conversion helpers
so both SQL backends serialize spans identically.

Aggregation contract: `aggregate()` must return exactly what
`tokenlens.cost.attribution.cost_by()` would return for the same traces —
same grouping keys, same Decimal math (cost summed in integer micro-dollars,
which is lossless because spans store cost quantized to 6 decimal places),
same (-cost, key) ordering. Filters select *traces*; aggregation then runs
over every span of the matching traces, mirroring "filter the trace list in
Python, then call cost_by on it".
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Protocol, runtime_checkable

from tokenlens.core.span import Span, Trace
from tokenlens.cost.attribution import CostBreakdownEntry

_USD_QUANTUM = Decimal("0.000001")
_MICRO_USD = Decimal(1_000_000)

# group_by key (as exposed to callers) → spans-table column. Mirrors
# attribution._GROUP_KEYS; also the whitelist that keeps group_by out of
# SQL-injection territory.
GROUP_BY_COLUMNS: dict[str, str] = {
    "user_id": "user_id",
    "feature_tag": "feature_tag",
    "model_name": "model_name",
    "node_name": "name",
    "kind": "kind",
}

# Column order shared by both backends' spans tables and by
# span_to_row/span_from_row. Must stay in sync with the CREATE TABLE DDL.
SPAN_COLUMNS: tuple[str, ...] = (
    "span_id",
    "trace_id",
    "parent_span_id",
    "name",
    "kind",
    "start_time",
    "end_time",
    "status",
    "error_message",
    "metadata",
    "model_name",
    "provider",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "prompt_preview",
    "retry_index",
    "cost_usd",
    "user_id",
    "feature_tag",
    "session_id",
)

TRACE_COLUMNS: tuple[str, ...] = (
    "trace_id",
    "root_name",
    "started_at",
    "total_cost_usd",
    "total_tokens",
    "user_id",
    "feature_tag",
    "session_id",
)

# SELECT list for trace summaries: the stored columns plus a computed
# "did any span fail" flag. Valid in both SQLite and Postgres.
SUMMARY_SELECT_SQL = ", ".join(TRACE_COLUMNS) + (
    ", EXISTS (SELECT 1 FROM spans WHERE spans.trace_id = traces.trace_id"
    " AND spans.status = 'ERROR') AS has_error"
)


@dataclass(frozen=True)
class TraceFilters:
    """Trace-level filters shared by list_traces and aggregate.

    Naive datetimes are interpreted as UTC. `model` matches traces that
    contain at least one span with that model_name.
    """

    user_id: str | None = None
    feature_tag: str | None = None
    model: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    min_cost: float | None = None


@dataclass(frozen=True)
class TraceSummary:
    """One row of list_traces — the traces table, essentially."""

    trace_id: str
    root_name: str
    started_at: datetime
    total_cost_usd: float
    total_tokens: int
    user_id: str | None
    feature_tag: str | None
    session_id: str | None
    has_error: bool = False


@dataclass(frozen=True)
class StatsOverview:
    """Headline numbers for a set of traces (the dashboard's top row).

    error_rate is the fraction of traces containing at least one ERROR
    span; retry_waste_usd matches attribution.retry_waste (cost of spans
    with retry_index > 0). Money is exact Decimal, summed in micro-dollars.
    """

    total_cost_usd: Decimal
    total_tokens: int
    trace_count: int
    error_rate: float
    retry_waste_usd: Decimal
    top_traces: list[TraceSummary]


@runtime_checkable
class StorageBackend(Protocol):
    """Async persistence interface for completed traces."""

    async def save_traces(self, traces: list[Trace]) -> None:
        """Persist traces (idempotent: re-saving a trace_id replaces it)."""
        ...

    async def get_trace(self, trace_id: str) -> Trace | None:
        """Load one full trace tree, or None if unknown."""
        ...

    async def list_traces(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: TraceFilters | None = None,
    ) -> list[TraceSummary]:
        """Summaries of matching traces, newest first."""
        ...

    async def aggregate(
        self,
        group_by: str,
        filters: TraceFilters | None = None,
    ) -> list[CostBreakdownEntry]:
        """Cost breakdown over all spans of matching traces, computed in SQL.

        group_by ∈ GROUP_BY_COLUMNS; result matches attribution.cost_by().
        """
        ...

    async def overview(self, filters: TraceFilters | None = None) -> StatsOverview:
        """Headline stats over matching traces, computed in SQL."""
        ...

    async def prune(self, older_than_days: int = 30) -> int:
        """Delete traces (and their spans) older than the cutoff; return count."""
        ...

    async def close(self) -> None:
        """Release connections. Safe to call more than once."""
        ...


def ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _iso(dt: datetime | None) -> str | None:
    # Uniform UTC ISO-8601 so lexicographic comparison in SQLite matches
    # chronological order.
    return None if dt is None else ensure_utc(dt).isoformat()


def _parse_dt(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def span_to_row(span: Span, *, datetimes_as_iso: bool) -> tuple[Any, ...]:
    """Serialize a Span in SPAN_COLUMNS order.

    datetimes_as_iso=True for SQLite (TEXT columns); False for Postgres
    (TIMESTAMPTZ columns take datetime objects). metadata is always a JSON
    string; enum fields become their string values.
    """
    to_dt: Callable[[datetime | None], Any] = _iso if datetimes_as_iso else lambda dt: dt
    return (
        span.span_id,
        span.trace_id,
        span.parent_span_id,
        span.name,
        span.kind.value,
        to_dt(span.start_time),
        to_dt(span.end_time),
        span.status.value,
        span.error_message,
        json.dumps(span.metadata),
        span.model_name,
        span.provider,
        span.input_tokens,
        span.output_tokens,
        span.cache_read_tokens,
        span.cache_write_tokens,
        span.prompt_preview,
        span.retry_index,
        span.cost_usd,
        span.user_id,
        span.feature_tag,
        span.session_id,
    )


def span_from_row(row: Sequence[Any]) -> Span:
    """Inverse of span_to_row; accepts ISO strings or datetime objects."""
    data = dict(zip(SPAN_COLUMNS, row, strict=True))
    data["start_time"] = _parse_dt(data["start_time"])
    data["end_time"] = _parse_dt(data["end_time"])
    if isinstance(data["metadata"], str):
        data["metadata"] = json.loads(data["metadata"])
    return Span(**data)


def trace_to_row(trace: Trace, *, datetimes_as_iso: bool) -> tuple[Any, ...]:
    """Serialize the traces-table summary row in TRACE_COLUMNS order."""
    root = trace.root
    started_at = _iso(root.start_time) if datetimes_as_iso else root.start_time
    return (
        trace.trace_id,
        root.name,
        started_at,
        trace.total_cost(),
        trace.total_tokens(),
        root.user_id,
        root.feature_tag,
        root.session_id,
    )


def summary_from_row(row: Sequence[Any]) -> TraceSummary:
    """Build a TraceSummary from a SUMMARY_SELECT_SQL row."""
    data = dict(zip((*TRACE_COLUMNS, "has_error"), row, strict=True))
    data["started_at"] = _parse_dt(data["started_at"])
    data["has_error"] = bool(data["has_error"])
    return TraceSummary(**data)


def trace_where(
    filters: TraceFilters | None,
    placeholder: Callable[[], str],
    *,
    datetimes_as_iso: bool,
    table: str = "traces",
) -> tuple[list[str], list[Any]]:
    """Build WHERE conditions against the traces table for both backends.

    `placeholder()` yields the next parameter marker ("?" for SQLite, "$1",
    "$2", ... for Postgres). Returns (conditions, params).
    """
    if filters is None:
        return [], []
    to_dt: Callable[[datetime], Any] = (lambda dt: _iso(dt)) if datetimes_as_iso else ensure_utc
    conditions: list[str] = []
    params: list[Any] = []
    if filters.user_id is not None:
        conditions.append(f"{table}.user_id = {placeholder()}")
        params.append(filters.user_id)
    if filters.feature_tag is not None:
        conditions.append(f"{table}.feature_tag = {placeholder()}")
        params.append(filters.feature_tag)
    if filters.model is not None:
        conditions.append(
            "EXISTS (SELECT 1 FROM spans _s WHERE _s.trace_id = "
            f"{table}.trace_id AND _s.model_name = {placeholder()})"
        )
        params.append(filters.model)
    if filters.since is not None:
        conditions.append(f"{table}.started_at >= {placeholder()}")
        params.append(to_dt(filters.since))
    if filters.until is not None:
        conditions.append(f"{table}.started_at <= {placeholder()}")
        params.append(to_dt(filters.until))
    if filters.min_cost is not None:
        conditions.append(f"{table}.total_cost_usd >= {placeholder()}")
        params.append(filters.min_cost)
    return conditions, params


def group_by_column(group_by: str) -> str:
    column = GROUP_BY_COLUMNS.get(group_by)
    if column is None:
        raise ValueError(
            f"unknown aggregate group_by {group_by!r}; expected one of {sorted(GROUP_BY_COLUMNS)}"
        )
    return column


def micros_to_usd(micros: Any) -> Decimal:
    """Exact Decimal dollars from an integer micro-dollar SQL sum."""
    return Decimal(int(micros)) / _MICRO_USD


def breakdown_from_rows(rows: Sequence[Sequence[Any]]) -> list[CostBreakdownEntry]:
    """Convert (group, cost_micro_usd, total_tokens, call_count) rows into
    CostBreakdownEntry objects, with the same Decimal math and ordering as
    attribution.cost_by()."""
    entries = []
    for group, cost_micros, tokens, calls in rows:
        cost = Decimal(int(cost_micros)) / _MICRO_USD
        entries.append(
            CostBreakdownEntry(
                key=str(group),
                cost_usd=cost,
                total_tokens=int(tokens),
                call_count=int(calls),
                avg_cost_per_call=(cost / int(calls)).quantize(
                    _USD_QUANTUM, rounding=ROUND_HALF_UP
                ),
            )
        )
    entries.sort(key=lambda e: (-e.cost_usd, e.key))
    return entries
