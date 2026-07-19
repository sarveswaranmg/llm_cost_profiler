"""CLI entry point for the `tokenlens` command.

Commands: server (API + dashboard), report (cost breakdown), top (most
expensive traces), trace (ASCII span tree), prune (retention), pricing
(active rate table). Requires the `cli` extra:
pip install "tokenlens[cli]".

Storage resolution, same for every command: --db PATH forces that SQLite
file; otherwise TOKENLENS_PG_DSN selects Postgres, then TOKENLENS_DB_PATH /
~/.tokenlens/traces.db select SQLite — identical to the library defaults.
"""

import asyncio
import json as jsonlib
import os
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, NoReturn

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.tree import Tree
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The tokenlens CLI requires the 'cli' extra. Install with: pip install 'tokenlens[cli]'"
    ) from exc

from tokenlens.core.span import Span, SpanKind, SpanStatus, Trace
from tokenlens.storage.base import StorageBackend, TraceFilters, TraceSummary

QUICKSTART_URL = "https://github.com/tokenlens/tokenlens#quickstart"
BAR_WIDTH = 28

app = typer.Typer(
    help="tokenlens — a flamegraph for token spend.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
# soft_wrap: never hard-wrap plain prints when piped (tables manage their
# own layout) — keeps `tokenlens report | grep ...` predictable.
console = Console(soft_wrap=True)

DbOption = Annotated[
    str | None,
    typer.Option(
        "--db",
        help="SQLite database path (default: $TOKENLENS_DB_PATH or ~/.tokenlens/traces.db; "
        "set $TOKENLENS_PG_DSN to use Postgres instead).",
    ),
]

# Span-kind colors, mirroring the dashboard palette.
_KIND_STYLE: dict[SpanKind, str] = {
    SpanKind.LLM_CALL: "blue",
    SpanKind.TOOL: "medium_purple",
    SpanKind.RETRIEVER: "cyan",
    SpanKind.EMBEDDING: "bright_cyan",
    SpanKind.RETRY: "red",
    SpanKind.CHAIN: "grey58",
    SpanKind.GRAPH_NODE: "grey66",
    SpanKind.CUSTOM: "grey74",
}


# ── shared helpers ───────────────────────────────────────────────────────────


def _make_backend(db: str | None) -> StorageBackend:
    if db is None and os.environ.get("TOKENLENS_PG_DSN"):
        from tokenlens.storage.postgres import PostgresBackend

        return PostgresBackend()
    from tokenlens.storage.sqlite import SqliteBackend

    return SqliteBackend(db)


def _run_with_backend(db: str | None, fn: Any) -> Any:
    """asyncio.run(fn(backend)) with a fresh backend, closed afterwards."""

    async def go() -> Any:
        backend = _make_backend(db)
        try:
            return await fn(backend)
        finally:
            await backend.close()

    return asyncio.run(go())


_WINDOW_RE = re.compile(r"^(\d+)\s*([mhdw]?)$")
_WINDOW_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks", "": "days"}


def _parse_window(value: str) -> timedelta:
    """Parse "7d" / "12h" / "30m" / "2w" (bare numbers mean days)."""
    match = _WINDOW_RE.match(value.strip().lower())
    if not match:
        console.print(
            f"[red]Can't parse time window {value!r}[/] — use e.g. [bold]7d[/], "
            "[bold]12h[/], [bold]30m[/], [bold]2w[/]."
        )
        raise typer.Exit(2)
    amount, unit = match.groups()
    return timedelta(**{_WINDOW_UNITS[unit]: int(amount)})


def _since_filters(since: str | None) -> TraceFilters:
    if since is None:
        return TraceFilters()
    return TraceFilters(since=datetime.now(UTC) - _parse_window(since))


def _no_traces_exit(since: str | None = None) -> NoReturn:
    if since is not None:
        console.print(f"[yellow]No traces in the last {since}.[/] Widen [bold]--since[/]?")
    else:
        console.print(
            "[yellow]No traces yet[/] — instrument your app with tokenlens, "
            "or generate sample traffic:\n"
            "  [bold]python examples/demo_langgraph_agent.py[/]\n"
            f"Quickstart: [link={QUICKSTART_URL}]{QUICKSTART_URL}[/link]"
        )
    raise typer.Exit(1)


def _usd(value: Decimal | float | None) -> str:
    if value is None:
        return "—"
    n = float(value)
    if n == 0:
        return "$0"
    a = abs(n)
    digits = 0 if a >= 100 else 2 if a >= 1 else 4 if a >= 0.01 else 6
    return f"${n:,.{digits}f}"


def _tok(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _dur(ms: float | None) -> str:
    if ms is None:
        return "—"
    if ms < 1_000:
        return f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1_000:.2f}s"
    return f"{ms / 60_000:.0f}m {ms % 60_000 / 1_000:.0f}s"


def _tier_style(cost: float) -> str:
    return "green" if cost < 0.01 else "yellow" if cost < 0.1 else "red"


# ── commands ─────────────────────────────────────────────────────────────────


@app.command()
def version() -> None:
    """Print the installed tokenlens version."""
    from tokenlens import __version__

    typer.echo(__version__)


@app.command()
def server(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on.")] = 8321,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    db: DbOption = None,
) -> None:
    """Serve the tokenlens API and dashboard."""
    try:
        import uvicorn

        from tokenlens.server.app import create_app
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The tokenlens server requires the 'server' extra. "
            "Install with: pip install 'tokenlens[server]'"
        ) from exc

    backend = _make_backend(db)
    console.print(f"tokenlens dashboard → [bold]http://{host}:{port}[/]")
    console.print(f"API docs           → http://{host}:{port}/docs")
    uvicorn.run(create_app(backend), host=host, port=port)


class GroupBy(StrEnum):
    feature = "feature"
    user = "user"
    model = "model"
    node = "node"


_GROUP_KEY: dict[GroupBy, str] = {
    GroupBy.feature: "feature_tag",
    GroupBy.user: "user_id",
    GroupBy.model: "model_name",
    GroupBy.node: "node_name",
}


@app.command()
def report(
    group_by: Annotated[
        GroupBy, typer.Option("--group-by", "-g", help="Dimension to break cost down by.")
    ] = GroupBy.feature,
    since: Annotated[
        str | None, typer.Option(help='Only traces newer than this, e.g. "7d", "12h".')
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Machine-readable JSON instead of a table.")
    ] = False,
    db: DbOption = None,
) -> None:
    """Cost breakdown: cost, tokens, calls, retry waste."""
    filters = _since_filters(since)

    async def go(backend: StorageBackend) -> tuple[Any, Any]:
        return (
            await backend.overview(filters),
            await backend.aggregate(_GROUP_KEY[group_by], filters=filters),
        )

    overview, entries = _run_with_backend(db, go)
    if overview.trace_count == 0:
        if as_json:
            typer.echo(jsonlib.dumps({"traces": 0, "entries": []}))
            raise typer.Exit(0)
        _no_traces_exit(since)

    if as_json:
        payload = {
            "group_by": group_by.value,
            "since": since,
            "totals": {
                "cost_usd": str(overview.total_cost_usd),
                "total_tokens": overview.total_tokens,
                "traces": overview.trace_count,
                "error_rate": overview.error_rate,
                "retry_waste_usd": str(overview.retry_waste_usd),
            },
            "entries": [
                {
                    "key": e.key,
                    "cost_usd": str(e.cost_usd),
                    "total_tokens": e.total_tokens,
                    "call_count": e.call_count,
                    "avg_cost_per_call": str(e.avg_cost_per_call),
                }
                for e in entries
            ],
        }
        typer.echo(jsonlib.dumps(payload, indent=2))
        return

    table = Table(
        title=f"cost by {group_by.value}" + (f" · last {since}" if since else ""),
        title_justify="left",
        header_style="bold",
    )
    table.add_column(group_by.value)
    table.add_column("calls", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("avg $/call", justify="right")
    table.add_column("cost", justify="right", style="bold")
    total = overview.total_cost_usd
    for e in entries:
        share = float(e.cost_usd / total) if total else 0.0
        table.add_row(
            e.key,
            str(e.call_count),
            _tok(e.total_tokens),
            _usd(e.avg_cost_per_call),
            f"{_usd(e.cost_usd)} [dim]{share:4.0%}[/]",
        )
    console.print(table)
    waste_style = "red" if overview.retry_waste_usd else "green"
    console.print(
        f"  total [bold]{_usd(total)}[/] · {_tok(overview.total_tokens)} tokens · "
        f"{overview.trace_count} traces · retry waste "
        f"[{waste_style}]{_usd(overview.retry_waste_usd)}[/] · "
        f"error rate {overview.error_rate:.1%}"
    )


@app.command()
def top(
    n: Annotated[int, typer.Option("--n", "-n", min=1, help="How many traces to show.")] = 10,
    since: Annotated[
        str | None, typer.Option(help='Only traces newer than this, e.g. "7d".')
    ] = None,
    db: DbOption = None,
) -> None:
    """The N most expensive traces, with a cost bar each."""
    filters = _since_filters(since)

    async def go(backend: StorageBackend) -> list[TraceSummary]:
        return await backend.list_traces(limit=1000, filters=filters)

    summaries: list[TraceSummary] = _run_with_backend(db, go)
    if not summaries:
        _no_traces_exit(since)

    ranked = sorted(summaries, key=lambda s: -s.total_cost_usd)[:n]
    max_cost = ranked[0].total_cost_usd or 1.0

    table = Table(
        header_style="bold", title=f"top {len(ranked)} traces by cost", title_justify="left"
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("cost", justify="right")
    table.add_column("", no_wrap=True)
    table.add_column("trace")
    table.add_column("trace id", style="dim")
    table.add_column("when", style="dim")
    for i, s in enumerate(ranked, start=1):
        style = _tier_style(s.total_cost_usd)
        bar = "█" * max(1, round(s.total_cost_usd / max_cost * BAR_WIDTH))
        name = s.root_name + (" [red]✗[/]" if s.has_error else "")
        table.add_row(
            str(i),
            f"[{style}]{_usd(s.total_cost_usd)}[/]",
            f"[{style}]{bar}[/]",
            name,
            s.trace_id[:12],
            s.started_at.astimezone().strftime("%H:%M:%S"),
        )
    console.print(table)


@app.command()
def trace(
    trace_id: Annotated[str, typer.Argument(help="Trace id (see `tokenlens top`).")],
    db: DbOption = None,
) -> None:
    """ASCII tree of one trace with per-span cost; retries in red."""

    async def go(backend: StorageBackend) -> Trace | None:
        return await backend.get_trace(trace_id)

    t: Trace | None = _run_with_backend(db, go)
    if t is None:
        console.print(
            f"[red]Trace {trace_id!r} not found.[/] List candidates with [bold]tokenlens top[/]."
        )
        raise typer.Exit(1)

    children: dict[str, list[Span]] = defaultdict(list)
    for span in t.spans:
        if span.parent_span_id is not None:
            children[span.parent_span_id].append(span)
    for siblings in children.values():
        siblings.sort(key=lambda s: s.start_time)

    def label(span: Span) -> str:
        style = _KIND_STYLE[span.kind]
        parts = [f"[{style}]{span.name}[/]", f"[dim]{span.kind.value.lower()}[/]"]
        if span.retry_index > 0:
            parts.append(f"[red bold]↻ retry #{span.retry_index}[/]")
        if span.status is SpanStatus.ERROR:
            parts.append(f"[red]✗ {span.error_message or 'error'}[/]")
        if span.model_name:
            parts.append(f"[dim]{span.model_name}[/]")
        cost_style = _tier_style(span.cost_usd or 0.0)
        cost_text = _usd(span.cost_usd) if span.cost_usd is not None else "—"
        parts.append(f"[{cost_style}]{cost_text}[/]")
        if span.total_tokens is not None:
            parts.append(f"{_tok(span.total_tokens)} tok")
        parts.append(_dur(span.duration_ms))
        return "  ".join(parts)

    started = t.root.start_time.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        f"[bold]{t.trace_id}[/] · started {started} · total "
        f"[bold]{_usd(t.total_cost())}[/] · {_tok(t.total_tokens())} tokens"
    )
    tree = Tree(label(t.root))

    def attach(node: Tree, span: Span) -> None:
        for child in children.get(span.span_id, []):
            attach(node.add(label(child)), child)

    attach(tree, t.root)
    console.print(tree)


@app.command()
def prune(
    older_than: Annotated[
        str, typer.Option("--older-than", help='Delete traces older than this, e.g. "30d".')
    ] = "30d",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
    db: DbOption = None,
) -> None:
    """Delete traces (and their spans) older than the cutoff."""
    window = _parse_window(older_than)
    days = window.total_seconds() / 86_400
    if not yes:
        typer.confirm(f"Delete all traces older than {older_than}?", abort=True)

    async def go(backend: StorageBackend) -> int:
        return await backend.prune(older_than_days=days)  # type: ignore[arg-type]

    deleted: int = _run_with_backend(db, go)
    console.print(f"Pruned [bold]{deleted}[/] trace{'s' if deleted != 1 else ''}.")


@app.command()
def pricing() -> None:
    """Show the loaded pricing table and its version."""
    from tokenlens.cost.pricing import get_default_table

    data = get_default_table().as_dict()
    console.print(
        f"pricing table [bold]{data['version']}[/] · source {data['source']} · "
        f"rates in USD per million tokens"
    )
    table = Table(header_style="bold")
    table.add_column("model")
    table.add_column("provider", style="dim")
    table.add_column("input", justify="right")
    table.add_column("output", justify="right")
    table.add_column("cache read", justify="right")
    table.add_column("cache write", justify="right")
    for name in sorted(data["models"]):
        p = data["models"][name]
        table.add_row(
            name,
            p.provider or "—",
            f"${p.input_per_mtok}",
            f"${p.output_per_mtok}",
            f"${p.cache_read_per_mtok}" if p.cache_read_per_mtok is not None else "—",
            f"${p.cache_write_per_mtok}" if p.cache_write_per_mtok is not None else "—",
        )
    console.print(table)
    aliases: dict[str, str] = data["aliases"]
    if aliases:
        console.print(
            f"[dim]{len(aliases)} aliases, e.g. "
            + ", ".join(f"{a} → {b}" for a, b in list(aliases.items())[:3])
            + "[/]"
        )


if __name__ == "__main__":
    app()
