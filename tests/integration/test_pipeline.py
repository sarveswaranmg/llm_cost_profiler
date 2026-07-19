"""Full pipeline: instrumented app → collector → tokenlens.init storage.

The mocked-OpenAI test is the money path: token counts are chosen so the
expected costs can be computed by hand (gpt-4o-mini at $0.15/$0.60 per
MTok), then asserted bit-for-bit against what SQLite hands back.
"""

from decimal import Decimal
from typing import Any

import openai
import pytest
from openai.resources.chat.completions import Completions
from tests import llm_mocks

import tokenlens
import tokenlens.storage as storage
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import SpanKind, SpanStatus
from tokenlens.instrument import openai_sdk
from tokenlens.storage.base import TraceFilters

# 1000 in × 0.15/MTok + 500 out × 0.60/MTok = 0.00045
_EXPECTED_CALL_COST = Decimal("0.000450")
# The retry re-sends the 1000-token prompt and dies before any output.
_EXPECTED_RETRY_COST = Decimal("0.000150")


def test_mocked_openai_app_lands_in_storage_with_exact_costs(
    monkeypatch: pytest.MonkeyPatch, collector: TraceCollector, tmp_path
) -> None:
    def fake_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        return llm_mocks.openai_response("gpt-4o-mini", input_tokens=1000, output_tokens=500)

    monkeypatch.setattr(Completions, "create", fake_create)
    openai_sdk.patch()

    backend = tokenlens.init(storage="sqlite", db_path=tmp_path / "pipeline.db")

    # -- the "user's app" -----------------------------------------------------
    tokenlens.set_user("u-pipe")
    tokenlens.set_feature("rag")
    client = openai.OpenAI(api_key="sk-test")
    with tokenlens.span("rag_request", kind=SpanKind.CHAIN):
        with tokenlens.span("fetch_docs", kind=SpanKind.RETRIEVER):
            pass
        client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "question"}]
        )
        # A failed first attempt that still billed its prompt tokens.
        with (
            pytest.raises(TimeoutError),
            tokenlens.span(
                "draft_answer",
                kind=SpanKind.LLM_CALL,
                model_name="gpt-4o-mini",
                input_tokens=1000,
                output_tokens=0,
                retry_index=1,
            ),
        ):
            raise TimeoutError("simulated 429")

    collector.flush()  # → StorageExporter → SQLite

    # -- what storage says ----------------------------------------------------
    overview = storage._runner.run(backend.overview())
    assert overview.trace_count == 1
    assert overview.total_cost_usd == _EXPECTED_CALL_COST + _EXPECTED_RETRY_COST
    assert overview.retry_waste_usd == _EXPECTED_RETRY_COST  # the retry carries cost
    assert overview.total_tokens == 2500
    assert overview.error_rate == 1.0  # the retry span errored

    (summary,) = storage._runner.run(backend.list_traces(filters=TraceFilters(user_id="u-pipe")))
    stored = storage._runner.run(backend.get_trace(summary.trace_id))
    assert stored is not None
    by_name = {s.name: s for s in stored.spans}

    retry_span = by_name["draft_answer"]
    assert retry_span.retry_index == 1
    assert retry_span.status == SpanStatus.ERROR
    assert Decimal(str(retry_span.cost_usd)) == _EXPECTED_RETRY_COST

    llm_span = by_name["gpt-4o-mini"]  # the adapter names the span after the model
    assert Decimal(str(llm_span.cost_usd)) == _EXPECTED_CALL_COST
    assert llm_span.parent_span_id == stored.root.span_id

    # Attribution set once on the context reached every span, incl. the
    # auto-instrumented one.
    assert all(s.user_id == "u-pipe" for s in stored.spans)
    assert all(s.feature_tag == "rag" for s in stored.spans)


def test_demo_rag_app_round_trips_through_storage(collector: TraceCollector, tmp_path) -> None:
    from examples.demo_rag_app import run_demo

    backend = tokenlens.init(storage="sqlite", db_path=tmp_path / "demo.db")
    tokenlens.set_feature("rag_demo")

    trace = run_demo()  # flushes internally → StorageExporter → SQLite

    stored = storage._runner.run(backend.get_trace(trace.trace_id))
    assert stored is not None
    assert len(stored.spans) == len(trace.spans)
    assert {s.span_id for s in stored.spans} == {s.span_id for s in trace.spans}
    assert stored.total_tokens() == trace.total_tokens() == 16  # 12 in + 4 out

    # The flaky model failed once, then succeeded: first attempt ERROR,
    # second attempt marked retry_index=1.
    llm_attempts = sorted(
        (s for s in stored.spans if s.kind == SpanKind.LLM_CALL and s.retry_index in (0, 1)),
        key=lambda s: s.start_time,
    )
    errored = [s for s in stored.spans if s.status == SpanStatus.ERROR]
    retried = [s for s in stored.spans if s.retry_index > 0]
    assert len(errored) >= 1
    assert len(retried) == 1
    assert llm_attempts, "expected LLM spans in the demo trace"

    # Fake demo models are not in the pricing table → cost stays null,
    # and the trace still lists with has_error=True.
    assert all(s.cost_usd is None for s in stored.spans)
    (summary,) = storage._runner.run(backend.list_traces())
    assert summary.has_error is True
    assert summary.feature_tag == "rag_demo"
