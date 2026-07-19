"""Raw OpenAI SDK instrumentation.

Monkey-patches `openai.resources.chat.completions.Completions.create` (and
its async counterpart) at the class level, so every call through any
`OpenAI`/`AsyncOpenAI` client becomes a traced LLM_CALL span — no LangChain
required. Streaming responses are returned as wrapped generators: the span
stays open and accumulates usage as chunks are pulled through, and only
closes once the stream is exhausted (or abandoned early), so we never force
materialization of the stream.

`openai` is imported lazily inside `patch()` so importing this module (or
the rest of tokenlens) never requires it to be installed.
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
    """Monkey-patch the OpenAI SDK's chat.completions.create, if installed.

    Safe to call more than once — each of the sync/async methods is only
    wrapped the first time.
    """
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except ImportError:
        return

    if not getattr(Completions.create, _PATCHED_MARKER, False):
        Completions.create = _wrap_create(Completions.create)  # type: ignore[method-assign]
    if not getattr(AsyncCompletions.create, _PATCHED_MARKER, False):
        AsyncCompletions.create = _wrap_async_create(  # type: ignore[method-assign]
            AsyncCompletions.create
        )


def _apply_usage(current: Span, usage: Any) -> None:
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is not None:
        current.input_tokens = prompt_tokens
    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is not None:
        current.output_tokens = completion_tokens
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached = getattr(prompt_details, "cached_tokens", None)
        if cached is not None:
            current.cache_read_tokens = cached
        cache_write = getattr(prompt_details, "cache_write_tokens", None)
        if cache_write is not None:
            current.cache_write_tokens = cache_write


def _start_span(kwargs: dict[str, Any]) -> Any:
    model = kwargs.get("model")
    preview = preview_from_messages(kwargs.get("messages"))
    return tokenlens_span(
        model or "openai-chat-completion",
        kind=SpanKind.LLM_CALL,
        model_name=model,
        provider="openai",
        prompt_preview=preview,
    )


def _wrap_create(original: Any) -> Any:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        stream = bool(kwargs.get("stream"))
        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

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
        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

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


def _wrap_stream(chunks: Any, current: Span, span_cm: Any) -> Iterator[Any]:
    try:
        for chunk in chunks:
            model = getattr(chunk, "model", None)
            if model:
                current.model_name = model
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                _apply_usage(current, usage)
            yield chunk
    except BaseException:
        span_cm.__exit__(*sys.exc_info())
        raise
    else:
        span_cm.__exit__(None, None, None)


async def _wrap_async_stream(chunks: Any, current: Span, span_cm: Any) -> AsyncIterator[Any]:
    try:
        async for chunk in chunks:
            model = getattr(chunk, "model", None)
            if model:
                current.model_name = model
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                _apply_usage(current, usage)
            yield chunk
    except BaseException:
        span_cm.__exit__(*sys.exc_info())
        raise
    else:
        span_cm.__exit__(None, None, None)
