# Custom pricing

tokenlens ships a versioned pricing table (USD per **million** tokens) and
bills every `LLM_CALL` span against it when the span is recorded. See what's
active with:

```bash
tokenlens pricing
```

## Table format

```json
{
  "version": "2026-07-01",
  "models": {
    "gpt-4o": {
      "provider": "openai",
      "input_per_mtok": 2.5,
      "output_per_mtok": 10.0,
      "cache_read_per_mtok": 1.25
    },
    "acme-internal-7b": {
      "provider": "acme",
      "input_per_mtok": 0.05,
      "output_per_mtok": 0.10
    }
  },
  "aliases": {
    "gpt-4o-2024-11-20": "gpt-4o"
  }
}
```

- `cache_read_per_mtok` / `cache_write_per_mtok` are optional; when absent,
  cache tokens bill at the input rate.
- Cache tokens are treated as separate buckets (Anthropic-style usage where
  `input_tokens` excludes cached tokens).

## Loading a custom table

Two ways; both **replace** the bundled table for the whole process:

```bash
export TOKENLENS_PRICING_PATH=/etc/tokenlens/pricing.json
```

```python
from tokenlens.cost.pricing import load_pricing

load_pricing("pricing.json")   # e.g. negotiated enterprise rates
```

Start from the bundled file
([`src/tokenlens/cost/pricing_data.json`](../src/tokenlens/cost/pricing_data.json))
and edit — a custom table must be complete, it is not merged with the
bundled one.

## Model-name resolution

Lookup is deliberately forgiving: exact match → alias → longest prefix
ending at a word boundary, so `gpt-4o-mini-2024-07-18` bills as
`gpt-4o-mini` without an explicit alias.

Unknown models **never crash anything**: the span records `cost_usd = null`,
a warning is logged once per model name, and everything else keeps working.
If you see `—` costs in the dashboard for a model you care about, add it to
your table.

## Precision

All pricing math is `Decimal`, quantized to 6 decimal places (micro-dollar).
The SQL aggregation layer sums integer micro-dollars, so API/CLI totals are
exact and match the Python attribution math bit-for-bit — money fields in
the API are serialized as decimal strings for the same reason.
