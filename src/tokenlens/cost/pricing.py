"""Pricing table loader and cost math.

Loads a versioned per-model pricing table (USD per million tokens) and
computes the dollar cost of a span from its model name and token counts.
All math is Decimal, rounded to 6 decimal places.

Model resolution is deliberately forgiving: exact match, then alias match,
then longest-prefix fuzzy match (so "gpt-4o-mini-2024-07-18" bills as
"gpt-4o-mini"). An unknown model logs a warning once per name and costs
resolve to None — pricing gaps must never crash the user's app.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from importlib.resources import files
from pathlib import Path
from typing import Any

from tokenlens.core.span import Span

logger = logging.getLogger("tokenlens.cost")

_MTOK = Decimal(1_000_000)
_USD_QUANTUM = Decimal("0.000001")
# Fuzzy prefix matches must end at a word boundary so e.g. a hypothetical
# "gpt-4" table entry can never claim "gpt-4o-...".
_BOUNDARY_CHARS = frozenset("-_.:@/")


@dataclass(frozen=True)
class ModelPricing:
    provider: str | None
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal | None = None
    cache_write_per_mtok: Decimal | None = None


def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(value))


class PricingTable:
    """A versioned model → per-mtok-rates table with forgiving name resolution."""

    def __init__(self, data: dict[str, Any], *, source: str = "<dict>") -> None:
        self.version: str = str(data.get("version", "unknown"))
        self.source = source
        self._models: dict[str, ModelPricing] = {}
        self._aliases: dict[str, str] = {str(k): str(v) for k, v in data.get("aliases", {}).items()}
        self._warned_unknown: set[str] = set()
        self._warn_lock = threading.Lock()

        for name, entry in data.get("models", {}).items():
            self._models[str(name)] = ModelPricing(
                provider=entry.get("provider"),
                input_per_mtok=_to_decimal(entry.get("input_per_mtok", 0)),
                output_per_mtok=_to_decimal(entry.get("output_per_mtok", 0)),
                cache_read_per_mtok=(
                    _to_decimal(entry["cache_read_per_mtok"])
                    if entry.get("cache_read_per_mtok") is not None
                    else None
                ),
                cache_write_per_mtok=(
                    _to_decimal(entry["cache_write_per_mtok"])
                    if entry.get("cache_write_per_mtok") is not None
                    else None
                ),
            )

        for alias, target in self._aliases.items():
            if target not in self._models:
                raise ValueError(
                    f"pricing table {source}: alias {alias!r} points at unknown model {target!r}"
                )

    @classmethod
    def from_path(cls, path: str | Path) -> "PricingTable":
        path = Path(path)
        with path.open() as f:
            return cls(json.load(f), source=str(path))

    @classmethod
    def bundled(cls) -> "PricingTable":
        raw = files("tokenlens.cost").joinpath("pricing_data.json").read_text()
        return cls(json.loads(raw), source="<bundled>")

    def resolve(self, model_name: str) -> str | None:
        """Resolve a raw model name to a canonical pricing key, or None.

        Exact match → alias → longest boundary-respecting prefix match.
        Unknown names log one warning per name (per table) and return None.
        """
        if model_name in self._models:
            return model_name
        if model_name in self._aliases:
            return self._aliases[model_name]

        best: str | None = None
        for known in list(self._models) + list(self._aliases):
            if (
                model_name.startswith(known)
                and model_name[len(known)] in _BOUNDARY_CHARS
                and (best is None or len(known) > len(best))
            ):
                best = known
        if best is not None:
            return self._aliases.get(best, best)

        with self._warn_lock:
            if model_name not in self._warned_unknown:
                self._warned_unknown.add(model_name)
                logger.warning(
                    "tokenlens: no pricing for model %r (table version %s) — "
                    "cost will be recorded as null",
                    model_name,
                    self.version,
                )
        return None

    def as_dict(self) -> dict[str, Any]:
        """Read-only snapshot of the table (version, source, models, aliases).

        What /api/pricing serves so the dashboard can display billing rates.
        """
        return {
            "version": self.version,
            "source": self.source,
            "models": dict(self._models),
            "aliases": dict(self._aliases),
        }

    def pricing_for(self, model_name: str) -> ModelPricing | None:
        key = self.resolve(model_name)
        return self._models[key] if key is not None else None

    def compute_cost(self, span: Span) -> Decimal | None:
        """Dollar cost of a span, or None if model or token data is missing.

        Cache reads/writes are billed at their own rates when the table has
        them, falling back to the input rate otherwise. Token fields are
        treated as independent buckets (Anthropic-style usage, where
        input_tokens excludes cache tokens).
        """
        if span.model_name is None:
            return None
        pricing = self.pricing_for(span.model_name)
        if pricing is None:
            return None
        if span.total_tokens is None:
            return None

        cache_read_rate = (
            pricing.cache_read_per_mtok
            if pricing.cache_read_per_mtok is not None
            else pricing.input_per_mtok
        )
        cache_write_rate = (
            pricing.cache_write_per_mtok
            if pricing.cache_write_per_mtok is not None
            else pricing.input_per_mtok
        )

        cost = (
            Decimal(span.input_tokens or 0) * pricing.input_per_mtok
            + Decimal(span.output_tokens or 0) * pricing.output_per_mtok
            + Decimal(span.cache_read_tokens or 0) * cache_read_rate
            + Decimal(span.cache_write_tokens or 0) * cache_write_rate
        ) / _MTOK
        return cost.quantize(_USD_QUANTUM, rounding=ROUND_HALF_UP)


_ENV_VAR = "TOKENLENS_PRICING_PATH"

_user_table: PricingTable | None = None
_cached_tables: dict[str, PricingTable] = {}
_default_lock = threading.Lock()


def load_pricing(path: str | Path) -> PricingTable:
    """Load a custom pricing table and install it as the process default.

    Lets users bill with negotiated/custom rates: every subsequent cost
    computation (including the collector's automatic enrichment) uses this
    table until the process exits.
    """
    global _user_table
    table = PricingTable.from_path(path)
    with _default_lock:
        _user_table = table
    return table


def get_default_table() -> PricingTable:
    """The active pricing table: load_pricing() > $TOKENLENS_PRICING_PATH > bundled."""
    with _default_lock:
        if _user_table is not None:
            return _user_table
        key = os.environ.get(_ENV_VAR) or "<bundled>"
        table = _cached_tables.get(key)
        if table is None:
            table = PricingTable.bundled() if key == "<bundled>" else PricingTable.from_path(key)
            _cached_tables[key] = table
        return table


def reset_default_table() -> None:
    """Forget any installed/cached default table. Mainly useful for tests."""
    global _user_table
    with _default_lock:
        _user_table = None
        _cached_tables.clear()


def compute_cost(span: Span, table: PricingTable | None = None) -> Decimal | None:
    """Dollar cost of a span using `table` (default: the active table)."""
    return (table or get_default_table()).compute_cost(span)


def enrich_span_cost(span: Span, table: PricingTable | None = None) -> None:
    """Fill in span.cost_usd from token data, if not already set.

    Never raises: an unpriceable span simply keeps cost_usd = None.
    """
    if span.cost_usd is not None or span.model_name is None:
        return
    cost = (table or get_default_table()).compute_cost(span)
    if cost is not None:
        span.cost_usd = float(cost)
