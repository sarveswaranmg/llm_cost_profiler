"""Mocked OpenAI / Anthropic SDK response objects for tests.

Everything here is a SimpleNamespace shaped like the real SDK response
types — attribute access only, which is all the instrumentation adapters
use. No network, no API keys.

Streaming variants return fresh (async) iterators of chunks/events with
usage carried where each provider actually puts it: OpenAI reports usage
on the final chunk; Anthropic splits it across `message_start` (input +
cache tokens) and `message_delta` (output tokens).
"""

from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

# -- OpenAI ChatCompletion shapes -------------------------------------------


def openai_usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> Any:
    details = None
    if cache_read_tokens is not None or cache_write_tokens is not None:
        details = SimpleNamespace(
            cached_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens
        )
    return SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        prompt_tokens_details=details,
    )


def openai_response(
    model: str = "gpt-4o-mini",
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    content: str = "mocked completion",
) -> Any:
    """A mocked non-streaming ChatCompletion."""
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=content))],
        usage=openai_usage(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens),
    )


def openai_stream_chunks(
    model: str = "gpt-4o-mini",
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    n_content_chunks: int = 3,
) -> list[Any]:
    """Chunk list for a streamed ChatCompletion: usage only on the final chunk."""
    chunks: list[Any] = [
        SimpleNamespace(
            model=model,
            choices=[SimpleNamespace(delta=SimpleNamespace(content=f"part-{i}"))],
            usage=None,
        )
        for i in range(n_content_chunks)
    ]
    chunks.append(
        SimpleNamespace(
            model=model,
            choices=[],
            usage=openai_usage(input_tokens, output_tokens),
        )
    )
    return chunks


def openai_stream(**kwargs: Any) -> Iterator[Any]:
    yield from openai_stream_chunks(**kwargs)


async def openai_async_stream(**kwargs: Any) -> AsyncIterator[Any]:
    for chunk in openai_stream_chunks(**kwargs):
        yield chunk


# -- Anthropic Message shapes ------------------------------------------------


def anthropic_usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> Any:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_tokens,
        cache_creation_input_tokens=cache_write_tokens,
    )


def anthropic_response(
    model: str = "claude-haiku-4-5",
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    content: str = "mocked message",
) -> Any:
    """A mocked non-streaming Anthropic Message."""
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text=content)],
        usage=anthropic_usage(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens),
    )


def anthropic_stream_events(
    model: str = "claude-haiku-4-5",
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read_tokens: int | None = None,
) -> list[Any]:
    """Event list for a streamed Message.

    input/cache tokens arrive on message_start, output tokens on the final
    message_delta — matching the real Anthropic streaming protocol.
    """
    return [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                model=model,
                usage=anthropic_usage(input_tokens, 0, cache_read_tokens),
            ),
        ),
        SimpleNamespace(type="content_block_delta"),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(
                input_tokens=None,
                output_tokens=output_tokens,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
            ),
        ),
    ]


def anthropic_stream(**kwargs: Any) -> Iterator[Any]:
    yield from anthropic_stream_events(**kwargs)


async def anthropic_async_stream(**kwargs: Any) -> AsyncIterator[Any]:
    for event in anthropic_stream_events(**kwargs):
        yield event
