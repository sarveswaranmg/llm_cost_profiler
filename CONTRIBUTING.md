# Contributing to tokenlens

Thanks for helping build the flamegraph for token spend. Issues and PRs
welcome — small, focused PRs land fastest.

## Dev setup

```bash
git clone https://github.com/tokenlens/tokenlens
cd tokenlens
python -m venv .venv && source .venv/bin/activate
make install            # pip install -e ".[all,dev]"
```

Sanity check:

```bash
make check              # ruff + mypy (strict) + pytest
```

Seed a database full of traces to hack against:

```bash
make demo               # runs the demo agent, serves the dashboard
```

## Project layout

| Path | What lives there |
|---|---|
| `src/tokenlens/core/` | Span/Trace model, context propagation, collector |
| `src/tokenlens/instrument/` | LangChain / LangGraph / OpenAI / Anthropic adapters |
| `src/tokenlens/cost/` | pricing table, cost math, attribution |
| `src/tokenlens/storage/` | StorageBackend protocol, SQLite, Postgres |
| `src/tokenlens/server/` | FastAPI app, response models, live WebSocket |
| `src/tokenlens/cli.py` | the `tokenlens` command |
| `dashboard/` | React dashboard (own README with dev workflow) |
| `tests/unit/` | components in isolation (core, cost, instrument, storage) |
| `tests/integration/` | instrumentation → collector → storage → API, WebSocket |
| `tests/e2e/` | real uvicorn server smoke test + `tokenlens` CLI |

## Tests

Tests are auto-marked `unit` / `integration` / `e2e` from their directory
(plus an explicit `slow` mark on the uvicorn smoke test). Everything is
mocked — no API keys, no network. Shared fixtures live in
`tests/conftest.py` (mock LLM responses, the hand-costed `sample_trace`,
a frozen clock) and `tests/integration/conftest.py` (the seeded server).

```bash
make test-fast                           # unit + integration (the default)
make test-all                            # everything, incl. the e2e smoke test
make test-cov                            # coverage run with the CI's 85% gate
pytest tests/unit/test_storage_sqlite.py -k aggregate
TOKENLENS_PG_DSN=postgresql://localhost/tokenlens_test pytest tests/unit/test_storage_postgres.py
```

- New behavior needs a test; bug fixes need a regression test.
- CI gates line coverage of `src/tokenlens` at 85% (`make test-cov`
  reproduces it locally).
- The SQL aggregation layer must stay in exact agreement with the
  pure-Python attribution math — the parametrized agreement tests in
  `tests/unit/test_storage_sqlite.py` are the contract. If you touch
  either side, run them first.
- Instrumentation must never break the host app: adapters swallow and log
  their own errors, and tests should cover the failure path.

## Dashboard

```bash
tokenlens server --port 8321        # terminal 1
cd dashboard && npm run dev         # terminal 2 → http://localhost:5173
```

`npm run build` type-checks and emits `dashboard/dist/`, which the server
picks up automatically. See [dashboard/README.md](dashboard/README.md).

## Style

- `ruff` (lint + format) and `mypy --strict` are enforced in CI: `make check`.
- Money is `Decimal` end-to-end; never sum costs as floats. API responses
  serialize money as decimal strings.
- Public functions get docstrings that say *why*, not just what.

## PRs

1. Fork, branch from `main`.
2. `make check` passes locally.
3. Update `CHANGELOG.md` (Unreleased section) for user-visible changes.
4. Describe the behavior change and how you verified it; screenshots for
   dashboard changes.

By contributing you agree your work is MIT-licensed like the rest of the
project.
