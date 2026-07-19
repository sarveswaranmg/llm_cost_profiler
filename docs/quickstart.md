# Quickstart

## Install

```bash
pip install "tokenlens[all]"
```

Extras if you want a smaller install: `[langchain]`, `[langgraph]`,
`[openai]`, `[anthropic]`, `[server]` (dashboard/API), `[postgres]`,
`[cli]`. The core library (spans, pricing, SQLite storage) has no heavy
dependencies.

## 1. Instrument

Pick whichever matches your stack (details: [integrations.md](integrations.md)):

```python
import tokenlens
from tokenlens.instrument.langchain import TokenLensCallbackHandler

tokenlens.init()   # optional: SQLite at ~/.tokenlens/traces.db is the default

chain.invoke(query, config={"callbacks": [TokenLensCallbackHandler()]})
```

Add attribution once, near your request handler — every span created below
it inherits the values:

```python
tokenlens.set_user(request.user_id)
tokenlens.set_feature("support_bot")
tokenlens.set_session(request.session_id)
```

Traces are buffered in memory and flushed to storage every few seconds on a
background thread (and at interpreter exit). Costs are computed from the
bundled per-model pricing table — see
[custom-pricing.md](custom-pricing.md) to override rates.

## 2. Explore

```bash
tokenlens server            # dashboard + API → http://127.0.0.1:8321
```

- **Flamegraph** — block width ∝ cost (toggle duration/tokens); retries are
  red and striped; click any span for full details.
- **Timeline** — the same trace as a waterfall; shows parallelism.
- **Attribution** — totals, retry waste, error rate, cost by
  feature/user/model, and a per-node cost table.

Or stay in the terminal:

```bash
tokenlens report --since 7d --group-by node
tokenlens top --n 10
tokenlens trace <trace_id>
```

## No app yet? Run the demo

```bash
make demo
# or by hand:
python examples/demo_langgraph_agent.py
tokenlens server
```

The demo agent (router → researcher → writer → critic) stores ~20 varied
traces with realistic token counts, a simulated rate-limit retry, and one
hard failure — enough for every dashboard view to light up.

## Where data lives

| Setting | Default | Override |
|---|---|---|
| Database | `~/.tokenlens/traces.db` (SQLite) | `TOKENLENS_DB_PATH`, `tokenlens.init(db_path=...)`, or `--db` on the CLI |
| Postgres | off | `TOKENLENS_PG_DSN` + `tokenlens.init(storage="postgres")` — see [self-hosting.md](self-hosting.md) |
| Pricing table | bundled | `TOKENLENS_PRICING_PATH` or `tokenlens.cost.pricing.load_pricing(path)` |
| Retention | keep everything | `tokenlens prune --older-than 30d` or `tokenlens.storage.prune(older_than_days=30)` |
