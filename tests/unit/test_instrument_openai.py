import sys
from types import SimpleNamespace
from typing import Any

import openai
import pytest
from openai.resources.chat.completions import AsyncCompletions, Completions

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import SpanKind
from tokenlens.instrument import openai_sdk


def _fake_usage(
    prompt_tokens: int = 10, completion_tokens: int = 5, cached_tokens: int | None = None
) -> Any:
    details = None
    if cached_tokens is not None:
        details = SimpleNamespace(cached_tokens=cached_tokens, cache_write_tokens=None)
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=details,
    )


def test_sync_create_produces_llm_call_span_with_usage(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(model="gpt-4o-mini", usage=_fake_usage(20, 8, cached_tokens=6))

    monkeypatch.setattr(Completions, "create", fake_create)
    openai_sdk.patch()

    client = openai.OpenAI(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        result = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hello"}]
        )

    assert result.model == "gpt-4o-mini"
    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.model_name == "gpt-4o-mini"
    assert llm_span.provider == "openai"
    assert llm_span.input_tokens == 20
    assert llm_span.output_tokens == 8
    assert llm_span.cache_read_tokens == 6
    assert llm_span.prompt_preview == "hello"


async def test_async_create_produces_llm_call_span_with_usage(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    async def fake_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(model="gpt-4o", usage=_fake_usage(15, 7))

    monkeypatch.setattr(AsyncCompletions, "create", fake_acreate)
    openai_sdk.patch()

    client = openai.AsyncOpenAI(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        result = await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hello async"}]
        )

    assert result.model == "gpt-4o"
    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.input_tokens == 15
    assert llm_span.output_tokens == 7


def test_streaming_accumulates_usage_without_forcing_materialization(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    pulled: list[int] = []
    chunks = [
        SimpleNamespace(model="gpt-4o-mini", usage=None),
        SimpleNamespace(model="gpt-4o-mini", usage=None),
        SimpleNamespace(model="gpt-4o-mini", usage=_fake_usage(30, 12)),
    ]

    def chunk_source() -> Any:
        for i, chunk in enumerate(chunks):
            pulled.append(i)
            yield chunk

    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("stream_options") == {"include_usage": True}
        return chunk_source()

    monkeypatch.setattr(Completions, "create", fake_create)
    openai_sdk.patch()

    client = openai.OpenAI(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        stream = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        assert pulled == []  # calling create() must not eagerly consume the stream

        assert next(stream) is chunks[0]
        assert pulled == [0]  # only the first chunk pulled so far
        assert collector.flush() == []  # trace isn't complete mid-stream

        rest = list(stream)
        assert rest == chunks[1:]
        assert pulled == [0, 1, 2]

    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.input_tokens == 30
    assert llm_span.output_tokens == 12


async def test_async_streaming_accumulates_usage(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    chunks = [
        SimpleNamespace(model="gpt-4o", usage=None),
        SimpleNamespace(model="gpt-4o", usage=_fake_usage(9, 3)),
    ]

    async def chunk_source() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        return chunk_source()

    monkeypatch.setattr(AsyncCompletions, "create", fake_acreate)
    openai_sdk.patch()

    client = openai.AsyncOpenAI(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        stream = await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        seen = [chunk async for chunk in stream]
        assert seen == chunks

    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.input_tokens == 9
    assert llm_span.output_tokens == 3


def test_patch_is_idempotent(monkeypatch: pytest.MonkeyPatch, collector: TraceCollector) -> None:
    calls = {"n": 0}

    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return SimpleNamespace(model="m", usage=_fake_usage())

    monkeypatch.setattr(Completions, "create", fake_create)
    openai_sdk.patch()
    openai_sdk.patch()  # must not double-wrap

    client = openai.OpenAI(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

    assert calls["n"] == 1
    traces = collector.flush()
    llm_spans = [s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL]
    assert len(llm_spans) == 1


def test_patch_noop_when_openai_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai.resources.chat.completions", None)
    openai_sdk.patch()  # must not raise
