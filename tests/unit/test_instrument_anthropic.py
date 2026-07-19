import sys
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from anthropic.resources.messages import AsyncMessages, Messages

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import SpanKind
from tokenlens.instrument import anthropic_sdk


def _fake_usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int | None = None,
    cache_write: int | None = None,
) -> Any:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def test_sync_create_produces_llm_call_span_with_usage(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            model="claude-sonnet-5", usage=_fake_usage(40, 15, cache_read=10, cache_write=2)
        )

    monkeypatch.setattr(Messages, "create", fake_create)
    anthropic_sdk.patch()

    client = anthropic.Anthropic(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        result = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
        )

    assert result.model == "claude-sonnet-5"
    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.model_name == "claude-sonnet-5"
    assert llm_span.provider == "anthropic"
    assert llm_span.input_tokens == 40
    assert llm_span.output_tokens == 15
    assert llm_span.cache_read_tokens == 10
    assert llm_span.cache_write_tokens == 2
    assert llm_span.prompt_preview == "hello"


async def test_async_create_produces_llm_call_span_with_usage(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    async def fake_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(model="claude-haiku-4-5", usage=_fake_usage(8, 4))

    monkeypatch.setattr(AsyncMessages, "create", fake_acreate)
    anthropic_sdk.patch()

    client = anthropic.AsyncAnthropic(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        result = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert result.model == "claude-haiku-4-5"
    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.input_tokens == 8
    assert llm_span.output_tokens == 4


def test_streaming_accumulates_usage_from_message_start_and_delta(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    pulled: list[int] = []
    message_start = SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(model="claude-sonnet-5", usage=_fake_usage(50, 1, cache_read=20)),
    )
    content_delta = SimpleNamespace(type="content_block_delta")
    message_delta = SimpleNamespace(type="message_delta", usage=_fake_usage(50, 18, cache_read=20))
    events = [message_start, content_delta, message_delta]

    def event_source() -> Any:
        for i, event in enumerate(events):
            pulled.append(i)
            yield event

    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        return event_source()

    monkeypatch.setattr(Messages, "create", fake_create)
    anthropic_sdk.patch()

    client = anthropic.Anthropic(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        stream = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert pulled == []

        assert next(stream) is message_start
        assert pulled == [0]
        assert collector.flush() == []

        rest = list(stream)
        assert rest == [content_delta, message_delta]

    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.model_name == "claude-sonnet-5"
    assert llm_span.input_tokens == 50
    assert llm_span.output_tokens == 18
    assert llm_span.cache_read_tokens == 20


def test_patch_is_idempotent(monkeypatch: pytest.MonkeyPatch, collector: TraceCollector) -> None:
    calls = {"n": 0}

    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return SimpleNamespace(model="m", usage=_fake_usage())

    monkeypatch.setattr(Messages, "create", fake_create)
    anthropic_sdk.patch()
    anthropic_sdk.patch()  # must not double-wrap

    client = anthropic.Anthropic(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        client.messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )

    assert calls["n"] == 1
    traces = collector.flush()
    llm_spans = [s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL]
    assert len(llm_spans) == 1


def test_patch_noop_when_anthropic_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic.resources.messages", None)
    anthropic_sdk.patch()  # must not raise
