import json
import logging
from decimal import Decimal
from pathlib import Path

import pytest

import tokenlens
from tokenlens.core.collector import TraceCollector
from tokenlens.core.span import Span, SpanKind
from tokenlens.cost.pricing import (
    PricingTable,
    compute_cost,
    get_default_table,
    load_pricing,
)

CUSTOM_TABLE = {
    "version": "test-1",
    "models": {
        "my-model": {"provider": "acme", "input_per_mtok": 1.0, "output_per_mtok": 2.0},
    },
    "aliases": {},
}


def _table() -> PricingTable:
    return PricingTable.bundled()


# -- resolution ---------------------------------------------------------------


def test_resolve_exact_match() -> None:
    assert _table().resolve("gpt-4o") == "gpt-4o"


def test_resolve_alias_match() -> None:
    assert _table().resolve("gpt-4o-2024-11-20") == "gpt-4o"


def test_resolve_fuzzy_prefix_prefers_longest_match() -> None:
    # Must resolve to gpt-4o-mini, not the shorter gpt-4o prefix.
    assert _table().resolve("gpt-4o-mini-2024-01-01") == "gpt-4o-mini"


def test_resolve_fuzzy_prefix_requires_word_boundary() -> None:
    # "gpt-4o2..." starts with "gpt-4o" but '2' is not a separator.
    assert _table().resolve("gpt-4o2-bogus") is None


def test_resolve_fuzzy_prefix_through_alias() -> None:
    assert _table().resolve("mistral-large-latest-v2") == "mistral-large"


def test_resolve_unknown_returns_none_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    table = _table()
    with caplog.at_level(logging.WARNING, logger="tokenlens.cost"):
        assert table.resolve("totally-unknown-model") is None
        assert table.resolve("totally-unknown-model") is None
    warnings = [r for r in caplog.records if "totally-unknown-model" in r.getMessage()]
    assert len(warnings) == 1


def test_alias_to_unknown_model_is_rejected_at_load() -> None:
    with pytest.raises(ValueError, match="alias"):
        PricingTable({"version": "x", "models": {}, "aliases": {"a": "missing"}})


# -- cost math ----------------------------------------------------------------


def _llm_span(**kwargs: object) -> Span:
    return Span(name="llm", kind=SpanKind.LLM_CALL, **kwargs)  # type: ignore[arg-type]


def test_compute_cost_hand_computed_input_output() -> None:
    # gpt-4o: 2.50 in / 10.00 out per mtok.
    span = _llm_span(model_name="gpt-4o", input_tokens=1000, output_tokens=500)
    assert _table().compute_cost(span) == Decimal("0.007500")


def test_compute_cost_hand_computed_with_cache_tokens() -> None:
    # claude-sonnet-4-6: 3.00 in / 15.00 out / 0.30 cache-read / 3.75 cache-write.
    # 1000*3.00 + 200*15.00 + 10000*0.30 + 2000*3.75 = 16500 per-mtok-units.
    span = _llm_span(
        model_name="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=200,
        cache_read_tokens=10_000,
        cache_write_tokens=2_000,
    )
    assert _table().compute_cost(span) == Decimal("0.016500")


def test_compute_cost_rounds_half_up_to_six_places() -> None:
    # 1 token at 2.50/mtok = 0.0000025 → rounds up to 0.000003.
    span = _llm_span(model_name="gpt-4o", input_tokens=1)
    assert _table().compute_cost(span) == Decimal("0.000003")


def test_compute_cost_cache_rates_fall_back_to_input_rate() -> None:
    # mistral-large has no cache rates: 1000 cache-read tokens bill at 2.00/mtok.
    span = _llm_span(model_name="mistral-large", cache_read_tokens=1000)
    assert _table().compute_cost(span) == Decimal("0.002000")


def test_compute_cost_embedding_model() -> None:
    span = _llm_span(model_name="text-embedding-3-small", input_tokens=1_000_000)
    assert _table().compute_cost(span) == Decimal("0.020000")


def test_compute_cost_none_without_model_or_tokens() -> None:
    table = _table()
    assert table.compute_cost(_llm_span(input_tokens=10)) is None  # no model
    assert table.compute_cost(_llm_span(model_name="gpt-4o")) is None  # no tokens
    assert table.compute_cost(_llm_span(model_name="unknown-x", input_tokens=10)) is None


# -- table selection ----------------------------------------------------------


def _write_table(path: Path) -> Path:
    target = path / "pricing.json"
    target.write_text(json.dumps(CUSTOM_TABLE))
    return target


def test_env_var_overrides_bundled_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENLENS_PRICING_PATH", str(_write_table(tmp_path)))
    table = get_default_table()
    assert table.version == "test-1"
    assert table.resolve("my-model") == "my-model"


def test_load_pricing_installs_process_default(tmp_path: Path) -> None:
    load_pricing(_write_table(tmp_path))
    span = _llm_span(model_name="my-model", input_tokens=1_000_000, output_tokens=1_000_000)
    assert compute_cost(span) == Decimal("3.000000")


# -- collector wiring ---------------------------------------------------------


def test_collector_enriches_closed_spans_with_cost(collector: TraceCollector) -> None:
    with (
        tokenlens.span("root", kind=SpanKind.CHAIN),
        tokenlens.span(
            "call",
            kind=SpanKind.LLM_CALL,
            model_name="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        ),
    ):
        pass

    traces = collector.flush()
    llm_span = next(s for s in traces[0].spans if s.kind == SpanKind.LLM_CALL)
    assert llm_span.cost_usd == pytest.approx(0.0075)
    assert traces[0].root.cost_usd is None  # chain span has no token data


def test_enricher_failure_does_not_break_recording() -> None:
    def broken(span: Span) -> None:
        raise RuntimeError("boom")

    collector = TraceCollector(flush_interval_seconds=0, enrichers=[broken])
    collector.record(Span(name="root", trace_id="t1"))
    traces = collector.flush()
    assert len(traces) == 1
    collector.shutdown()


def test_enricher_never_overwrites_explicit_cost(collector: TraceCollector) -> None:
    with tokenlens.span(
        "call", kind=SpanKind.LLM_CALL, model_name="gpt-4o", input_tokens=1000, cost_usd=42.0
    ):
        pass
    traces = collector.flush()
    assert traces[0].root.cost_usd == 42.0
