"""tokenlens — a flamegraph for token spend.

Public API:
    - init(storage=...): single setup call choosing where traces persist
      ("sqlite" | "postgres" | a StorageBackend instance). Optional — the
      default without init is SQLite at ~/.tokenlens/traces.db.
    - span(name, kind=..., **attrs): context manager that creates a traced
      span, nesting under whatever span is currently open.
    - set_user / set_feature / set_session: stamp attribution onto the
      current context; inherited by all descendant spans.
    - get_collector(): the process-wide TraceCollector.

Instrumentation adapters (LangChain, LangGraph, raw OpenAI/Anthropic SDKs)
build on top of these primitives — see `tokenlens.instrument`. Retention:
`tokenlens.storage.prune(older_than_days=30)`.

Status: under active development. The `profile` decorator is not implemented yet.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path

from tokenlens.core.collector import TraceCollector, get_collector
from tokenlens.core.context import set_feature, set_session, set_user, span
from tokenlens.core.span import Span, SpanKind, SpanStatus, Trace
from tokenlens.storage.base import StorageBackend


def init(
    storage: "str | StorageBackend" = "sqlite",
    *,
    db_path: "str | Path | None" = None,
    dsn: "str | None" = None,
) -> "StorageBackend":
    """Configure where completed traces are persisted.

    storage: "sqlite" (default; db_path overrides ~/.tokenlens/traces.db),
    "postgres" (dsn or the TOKENLENS_PG_DSN env var), or any StorageBackend
    instance. Replaces the collector's exporters, so call it early — before
    the first traced span is flushed. Returns the configured backend.
    """
    from tokenlens import storage as _storage

    backend = _storage.resolve_backend(storage, db_path=db_path, dsn=dsn)
    _storage.set_backend(backend)
    collector = get_collector()
    collector.set_exporters([_storage.StorageExporter(backend)])
    return backend


try:
    __version__ = _version("tokenlens")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "init",
    "span",
    "set_user",
    "set_feature",
    "set_session",
    "get_collector",
    "TraceCollector",
    "Span",
    "SpanKind",
    "SpanStatus",
    "Trace",
]
