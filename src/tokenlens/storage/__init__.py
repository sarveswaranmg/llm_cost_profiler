"""Storage backends.

Defines a common StorageBackend protocol implemented by SQLite (default,
zero-config) and Postgres (for production/multi-user deployments), plus the
glue that connects them to the rest of tokenlens:

- get_backend()/set_backend(): the process-wide backend. The default (no
  tokenlens.init call) is SQLite at ~/.tokenlens/traces.db.
- StorageExporter: adapts a StorageBackend to the collector's synchronous
  exporter interface, driving the async backend on a dedicated event-loop
  thread. This is what flush() hands completed traces to.
- prune(older_than_days=30): synchronous retention helper.
"""

import asyncio
import logging
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

from tokenlens.core.span import Trace
from tokenlens.storage.base import StorageBackend, TraceFilters, TraceSummary
from tokenlens.storage.sqlite import SqliteBackend

logger = logging.getLogger("tokenlens.storage")

T = TypeVar("T")


class _AsyncRunner:
    """Drives backend coroutines from synchronous code.

    Owns a daemon thread running an event loop; run() submits a coroutine to
    it and blocks for the result. All storage I/O triggered from sync code
    (collector flushes, prune) goes through the same loop, so backend
    connections are only ever touched by one event loop.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            with self._lock:
                if self._loop is None or self._loop.is_closed():
                    loop = asyncio.new_event_loop()
                    thread = threading.Thread(
                        target=loop.run_forever, name="tokenlens-storage", daemon=True
                    )
                    thread.start()
                    self._loop = loop
        return self._loop

    def run(self, coro: Coroutine[Any, Any, T], timeout: float = 30.0) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        return future.result(timeout)


_runner = _AsyncRunner()

_backend: StorageBackend | None = None
_backend_lock = threading.Lock()


def set_backend(backend: StorageBackend) -> None:
    """Replace the process-wide storage backend."""
    global _backend
    if not isinstance(backend, StorageBackend):
        raise TypeError(f"{backend!r} does not implement the StorageBackend protocol")
    with _backend_lock:
        _backend = backend


def get_backend() -> StorageBackend:
    """Return the process-wide backend, defaulting to zero-config SQLite.

    Creating SqliteBackend is lazy and cheap — no file is touched until the
    first trace is saved or queried.
    """
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = SqliteBackend()
    return _backend


def resolve_backend(
    storage: str | StorageBackend = "sqlite",
    *,
    db_path: str | Path | None = None,
    dsn: str | None = None,
) -> StorageBackend:
    """Turn tokenlens.init()'s `storage` argument into a backend instance."""
    if isinstance(storage, str):
        if storage == "sqlite":
            return SqliteBackend(db_path)
        if storage == "postgres":
            from tokenlens.storage.postgres import PostgresBackend

            return PostgresBackend(dsn)
        raise ValueError(
            f"unknown storage {storage!r}; expected 'sqlite', 'postgres', "
            "or a StorageBackend instance"
        )
    if not isinstance(storage, StorageBackend):
        raise TypeError(f"{storage!r} does not implement the StorageBackend protocol")
    return storage


class StorageExporter:
    """Collector exporter that persists each completed trace to a backend.

    Synchronous and swallow-all: a storage failure is logged, never raised —
    losing a trace must not take down the flush thread or the user's app.
    """

    def __init__(self, backend: StorageBackend | None = None) -> None:
        self.backend = backend if backend is not None else get_backend()

    def __call__(self, trace: Trace) -> None:
        try:
            _runner.run(self.backend.save_traces([trace]))
        except Exception:
            logger.exception(
                "tokenlens: failed to persist trace %s to %r", trace.trace_id, self.backend
            )


def prune(older_than_days: int = 30) -> int:
    """Delete traces older than the cutoff from the configured backend.

    Synchronous convenience for scripts/cron; in async code call
    `await get_backend().prune(...)` instead. Returns the number of traces
    deleted.
    """
    return _runner.run(get_backend().prune(older_than_days))


__all__ = [
    "SqliteBackend",
    "StorageBackend",
    "StorageExporter",
    "TraceFilters",
    "TraceSummary",
    "get_backend",
    "prune",
    "resolve_backend",
    "set_backend",
]
