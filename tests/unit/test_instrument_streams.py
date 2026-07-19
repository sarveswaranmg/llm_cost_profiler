"""Stream abandonment and auto_patch() idempotency for the raw-SDK adapters.

A stream the caller walks away from must still close its span — an open
span would keep its whole trace buffered in the collector forever.
"""

from typing import Any

import anthropic
import openai
import pytest
from anthropic.resources.messages import Messages
from openai.resources.chat.completions import AsyncCompletions, Completions
from tests import llm_mocks

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import SpanKind
from tokenlens.instrument import auto_patch


def _llm_spans(collector: TraceCollector) -> list[Any]:
    return [s for t in collector.flush() for s in t.spans if s.kind == SpanKind.LLM_CALL]


def test_abandoned_openai_stream_still_closes_span(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        return llm_mocks.openai_stream(input_tokens=40, output_tokens=20)

    monkeypatch.setattr(Completions, "create", fake_create)
    auto_patch()

    client = openai.OpenAI(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        stream = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        next(stream)  # pull one chunk...
        stream.close()  # ...then abandon the stream

    (llm_span,) = _llm_spans(collector)
    assert llm_span.end_time is not None  # closed, not leaked
    # Usage lives on the final chunk, which was never pulled — so this is a
    # "partial" span: closed, but without token counts.
    assert llm_span.input_tokens is None
    assert llm_span.output_tokens is None
    assert collector.flush() == []  # nothing stuck behind an open span


async def test_abandoned_async_openai_stream_still_closes_span(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    async def fake_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        return llm_mocks.openai_async_stream(input_tokens=40, output_tokens=20)

    monkeypatch.setattr(AsyncCompletions, "create", fake_acreate)
    auto_patch()

    client = openai.AsyncOpenAI(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        stream = await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        await anext(stream)
        await stream.aclose()

    (llm_span,) = _llm_spans(collector)
    assert llm_span.end_time is not None


def test_abandoned_anthropic_stream_keeps_partial_usage(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        return llm_mocks.anthropic_stream(input_tokens=100, output_tokens=50)

    monkeypatch.setattr(Messages, "create", fake_create)
    auto_patch()

    client = anthropic.Anthropic(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        stream = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        # message_start (with input tokens) was pulled; output tokens on the
        # later message_delta never arrive.
        next(stream)
        stream.close()

    (llm_span,) = _llm_spans(collector)
    assert llm_span.end_time is not None
    assert llm_span.input_tokens == 100  # partial usage kept
    assert llm_span.output_tokens == 0  # message_delta never consumed


def test_fully_consumed_stream_captures_final_usage(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        return llm_mocks.openai_stream(input_tokens=33, output_tokens=11)

    monkeypatch.setattr(Completions, "create", fake_create)
    auto_patch()

    client = openai.OpenAI(api_key="sk-test")
    with tokenlens.span("root", kind=SpanKind.CHAIN):
        stream = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        list(stream)  # consume fully

    (llm_span,) = _llm_spans(collector)
    assert llm_span.input_tokens == 33
    assert llm_span.output_tokens == 11


def test_auto_patch_twice_does_not_double_wrap_either_sdk(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector
) -> None:
    calls = {"openai": 0, "anthropic": 0}

    def fake_openai_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["openai"] += 1
        return llm_mocks.openai_response()

    def fake_anthropic_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["anthropic"] += 1
        return llm_mocks.anthropic_response()

    monkeypatch.setattr(Completions, "create", fake_openai_create)
    monkeypatch.setattr(Messages, "create", fake_anthropic_create)

    auto_patch()
    auto_patch()  # must be a no-op the second time

    with tokenlens.span("root", kind=SpanKind.CHAIN):
        openai.OpenAI(api_key="sk-test").chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )
        anthropic.Anthropic(api_key="sk-test").messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert calls == {"openai": 1, "anthropic": 1}  # underlying hit exactly once each
    (trace,) = collector.flush()
    llm_spans = [s for s in trace.spans if s.kind == SpanKind.LLM_CALL]
    assert len(llm_spans) == 2  # exactly one span per call — no double wrap
