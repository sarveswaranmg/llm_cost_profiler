"""Raw Anthropic SDK instrumentation.

Monkey-patches `anthropic.resources.messages.Messages.create` (and its
async counterpart) at the class level, so every call through any
`Anthropic`/`AsyncAnthropic` client becomes a traced LLM_CALL span — no
LangChain required. Streaming responses are returned as wrapped iterators:
usage arrives piecemeal (input/cache tokens on the `message_start` event,
output tokens on `message_delta` events), so we accumulate as events are
pulled through and only close the span once the stream is exhausted.

`anthropic` is imported lazily inside `patch()` so importing this module
(or the rest of tokenlens) never requires it to be installed.
"""

import functools
import sys
from collections.abc import AsyncIterator, Iterator
from typing import Any

from tokenlens.core.context import span as tokenlens_span
from tokenlens.core.span import Span, SpanKind
from tokenlens.instrument._common import preview_from_messages

_PATCHED_MARKER = "_tokenlens_patched"


def patch() -> None:
    """Monkey-patch the Anthropic SDK's messages.create, if installed.

    Safe to call more than once — each of the sync/async methods is only
    wrapped the first time.
    """
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except ImportError:
        return

    if not getattr(Messages.create, _PATCHED_MARKER, False):
        Messages.create = _wrap_create(Messages.create)  # type: ignore[method-assign]
    if not getattr(AsyncMessages.create, _PATCHED_MARKER, False):
        AsyncMessages.create = _wrap_async_create(  # type: ignore[method-assign]
            AsyncMessages.create
        )


def _apply_usage(current: Span, usage: Any) -> None:
    if usage is None:
        return
    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is not None:
        current.input_tokens = input_tokens
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is not None:
        current.output_tokens = output_tokens
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    if cache_read is not None:
        current.cache_read_tokens = cache_read
    cache_write = getattr(usage, "cache_creation_input_tokens", None)
    if cache_write is not None:
        current.cache_write_tokens = cache_write


def _consume_event(current: Span, event: Any) -> None:
    event_type = getattr(event, "type", None)
    if event_type == "message_start":
        message = getattr(event, "message", None)
        model = getattr(message, "model", None)
        if model:
            current.model_name = model
        _apply_usage(current, getattr(message, "usage", None))
    elif event_type == "message_delta":
        _apply_usage(current, getattr(event, "usage", None))


def _start_span(kwargs: dict[str, Any]) -> Any:
    model = kwargs.get("model")
    preview = preview_from_messages(kwargs.get("messages"))
    return tokenlens_span(
        model or "anthropic-messages-create",
        kind=SpanKind.LLM_CALL,
        model_name=model,
        provider="anthropic",
        prompt_preview=preview,
    )


def _wrap_create(original: Any) -> Any:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        stream = bool(kwargs.get("stream"))
        span_cm = _start_span(kwargs)
        current = span_cm.__enter__()
        try:
            result = original(self, *args, **kwargs)
        except BaseException:
            span_cm.__exit__(*sys.exc_info())
            raise

        if stream:
            return _wrap_stream(result, current, span_cm)

        current.model_name = getattr(result, "model", None) or current.model_name
        _apply_usage(current, getattr(result, "usage", None))
        span_cm.__exit__(None, None, None)
        return result

    wrapper._tokenlens_patched = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_async_create(original: Any) -> Any:
    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        stream = bool(kwargs.get("stream"))
        span_cm = _start_span(kwargs)
        current = span_cm.__enter__()
        try:
            result = await original(self, *args, **kwargs)
        except BaseException:
            span_cm.__exit__(*sys.exc_info())
            raise

        if stream:
            return _wrap_async_stream(result, current, span_cm)

        current.model_name = getattr(result, "model", None) or current.model_name
        _apply_usage(current, getattr(result, "usage", None))
        span_cm.__exit__(None, None, None)
        return result

    wrapper._tokenlens_patched = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_stream(events: Any, current: Span, span_cm: Any) -> Iterator[Any]:
    try:
        for event in events:
            _consume_event(current, event)
            yield event
    except BaseException:
        span_cm.__exit__(*sys.exc_info())
        raise
    else:
        span_cm.__exit__(None, None, None)


async def _wrap_async_stream(events: Any, current: Span, span_cm: Any) -> AsyncIterator[Any]:
    try:
        async for event in events:
            _consume_event(current, event)
            yield event
    except BaseException:
        span_cm.__exit__(*sys.exc_info())
        raise
    else:
        span_cm.__exit__(None, None, None)
