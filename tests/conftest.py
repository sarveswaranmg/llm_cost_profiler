"""Shared fixtures for the whole suite.

Layout: tests/unit (components in isolation), tests/integration
(instrumentation → collector → storage → API), tests/e2e (real server +
CLI). Tests are auto-marked unit/integration/e2e from their directory, so
`pytest -m "not e2e"` is the fast default (see Makefile `test-fast`).
"""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest
from tests import factories, llm_mocks

import tokenlens.storage as _storage
from tokenlens.core.collector import TraceCollector, set_collector
from tokenlens.core.context import reset_context
from tokenlens.core.span import Trace
from tokenlens.cost.pricing import reset_default_table
from tokenlens.storage.sqlite import SqliteBackend

_TESTS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark tests unit/integration/e2e based on their directory."""
    for item in items:
        try:
            rel = Path(item.path).relative_to(_TESTS_DIR)
        except ValueError:
            continue
        tier = rel.parts[0] if len(rel.parts) > 1 else None
        if tier in ("unit", "integration", "e2e"):
            item.add_marker(getattr(pytest.mark, tier))


@pytest.fixture(autouse=True)
def _isolated_context(tmp_path, monkeypatch) -> Iterator[None]:
    # Belt and braces: even if a test reaches the default storage path, it
    # must never touch the real ~/.tokenlens/traces.db.
    monkeypatch.setenv("TOKENLENS_DB_PATH", str(tmp_path / "tokenlens-test.db"))
    _storage._backend = None
    reset_context()
    reset_default_table()
    yield
    _storage._backend = None
    reset_context()
    reset_default_table()


@pytest.fixture
def collector() -> Iterator[TraceCollector]:
    c = TraceCollector(flush_interval_seconds=0)
    set_collector(c)
    yield c
    c.shutdown()


# -- mocked LLM responses ----------------------------------------------------


@pytest.fixture
def fake_llm_response() -> ModuleType:
    """Factory namespace for mocked OpenAI/Anthropic responses and streams.

    Exposes tests/llm_mocks.py: openai_response, openai_stream,
    openai_async_stream, anthropic_response, anthropic_stream,
    anthropic_async_stream — all configurable token counts, no network.
    """
    return llm_mocks


# -- sample trace ------------------------------------------------------------


@pytest.fixture
def sample_trace() -> Callable[..., Trace]:
    """Factory for the canonical 6-span trace with hand-computed costs.

    See tests/factories.py for the tree shape and the cost arithmetic;
    expected totals are exported there as constants.
    """
    return factories.sample_trace


# -- storage -----------------------------------------------------------------


@pytest.fixture
async def tmp_sqlite_storage(tmp_path):
    """A fresh SQLite backend on a per-test temp file."""
    backend = SqliteBackend(tmp_path / "storage.db")
    yield backend
    await backend.close()


# -- deterministic clock -----------------------------------------------------


class FrozenClock:
    """Manually advanced clock injected into span creation/closing.

    span durations become exact: open a span, tick(ms), close it, and
    duration_ms equals the ticked amount.
    """

    def __init__(self, start: datetime) -> None:
        self.current = start

    def tick(self, ms: float = 1.0) -> datetime:
        self.current += timedelta(milliseconds=ms)
        return self.current


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> FrozenClock:
    clock = FrozenClock(datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC))

    class _FrozenDatetime:
        @staticmethod
        def now(tz: object = None) -> datetime:
            return clock.current

    # Span.start_time's default_factory (_utcnow) resolves `datetime` as a
    # module global at call time, so patch the module's datetime — patching
    # _utcnow itself would miss the reference Field() captured at class
    # definition. Same trick for context.span()'s end_time stamp.
    monkeypatch.setattr("tokenlens.core.span.datetime", _FrozenDatetime)
    monkeypatch.setattr("tokenlens.core.context.datetime", _FrozenDatetime)
    return clock
