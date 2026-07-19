"""contextvars-based span propagation.

Exposes the `span()` context manager plus `set_user`/`set_feature`/
`set_session` helpers that stamp attribution onto the current context so
every descendant span inherits it automatically. Built on contextvars, so
this is correct across asyncio tasks (each task gets its own copy of the
context on creation) and across threads (each thread starts with its own
context).
"""

import contextlib
import uuid
from collections.abc import Generator
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from tokenlens.core.collector import get_collector
from tokenlens.core.span import Span, SpanKind, SpanStatus

_current_trace_id: ContextVar[str | None] = ContextVar("tokenlens_trace_id", default=None)
_current_span_id: ContextVar[str | None] = ContextVar("tokenlens_span_id", default=None)
_current_user_id: ContextVar[str | None] = ContextVar("tokenlens_user_id", default=None)
_current_feature_tag: ContextVar[str | None] = ContextVar("tokenlens_feature_tag", default=None)
_current_session_id: ContextVar[str | None] = ContextVar("tokenlens_session_id", default=None)


def set_user(user_id: str) -> None:
    """Stamp a user id onto the current context; inherited by descendant spans."""
    _current_user_id.set(user_id)


def set_feature(tag: str) -> None:
    """Stamp a feature tag onto the current context; inherited by descendant spans."""
    _current_feature_tag.set(tag)


def set_session(session_id: str) -> None:
    """Stamp a session id onto the current context; inherited by descendant spans."""
    _current_session_id.set(session_id)


def current_attribution() -> dict[str, str | None]:
    """Return the user/feature/session ids currently stamped on the context.

    Used by instrumentation adapters (e.g. the LangChain callback handler)
    that build Spans directly rather than through `span()`, so attribution
    set via `set_user`/`set_feature`/`set_session` still reaches them.
    """
    return {
        "user_id": _current_user_id.get(),
        "feature_tag": _current_feature_tag.get(),
        "session_id": _current_session_id.get(),
    }


def reset_context() -> None:
    """Clear current-trace and attribution state.

    Useful in tests, and for thread-pool workers that must not leak trace
    context from a previous, unrelated unit of work into the next one.
    """
    _current_trace_id.set(None)
    _current_span_id.set(None)
    _current_user_id.set(None)
    _current_feature_tag.set(None)
    _current_session_id.set(None)


@contextlib.contextmanager
def span(name: str, kind: SpanKind = SpanKind.CUSTOM, **attrs: Any) -> Generator[Span, None, None]:
    """Create a Span, make it "current", and report it to the collector on exit.

    Nests under whatever span is current, if any. Attribution fields
    (user_id/feature_tag/session_id) are inherited from the context unless
    passed explicitly in `attrs`.
    """
    parent_span_id = _current_span_id.get()
    trace_id = _current_trace_id.get() or uuid.uuid4().hex

    for key, value in current_attribution().items():
        attrs.setdefault(key, value)

    current = Span(
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        name=name,
        kind=kind,
        **attrs,
    )

    span_token = _current_span_id.set(current.span_id)
    trace_token = _current_trace_id.set(trace_id)
    try:
        yield current
    except Exception as exc:
        current.status = SpanStatus.ERROR
        current.error_message = str(exc)
        raise
    finally:
        current.end_time = datetime.now(UTC)
        _current_span_id.reset(span_token)
        _current_trace_id.reset(trace_token)
        get_collector().record(current)
