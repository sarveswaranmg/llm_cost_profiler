# Changelog

All notable changes to tokenlens are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses
[semantic versioning](https://semver.org/) once released.

## [Unreleased]

## [0.1.0] — 2026-07-17

Initial release.

### Added

- **Core tracing** — `Span`/`Trace` model, `tokenlens.span()` context
  manager on `contextvars` (asyncio- and thread-correct nesting),
  `set_user` / `set_feature` / `set_session` attribution inheritance, and a
  background-flushing `TraceCollector` that never blocks the host app.
- **Instrumentation** — LangChain callback handler, LangGraph
  `instrument_graph()` / `@traced_node`, and `auto_patch()` for the raw
  OpenAI and Anthropic SDKs (sync + async, streaming-safe).
- **Cost engine** — bundled versioned pricing table (USD per mtok, cache
  read/write rates), forgiving model-name resolution, Decimal math
  quantized to micro-dollars, custom tables via `load_pricing()` /
  `TOKENLENS_PRICING_PATH`.
- **Attribution** — `cost_by()` breakdowns (user/feature/model/node/kind),
  `retry_waste()`, and d3-flamegraph export.
- **Storage** — async `StorageBackend` protocol with zero-config SQLite
  (default, `~/.tokenlens/traces.db`) and Postgres (asyncpg) backends;
  SQL aggregation in exact integer micro-dollars; `tokenlens.init()` as the
  single setup call; `prune()` retention.
- **API server** — FastAPI app: trace list/tree/flamegraph endpoints,
  aggregation, stats overview, pricing, `/ws/live` WebSocket streaming, and
  static serving of the built dashboard.
- **Dashboard** — dark-theme React app: live trace list with filters, cost
  flamegraph (weight by cost/duration/tokens, retries striped red), Gantt
  timeline, attribution view with charts and a sortable per-node table,
  span detail drawer with copyable JSON.
- **CLI** — `tokenlens server | report | top | trace | prune | pricing`
  with rich tables, `--json` output, and shared storage resolution.
- **Demo** — `examples/demo_langgraph_agent.py` (instrumented LangGraph
  agent seeding ~20 realistic traces) and `make demo`.

[Unreleased]: https://github.com/tokenlens/tokenlens/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tokenlens/tokenlens/releases/tag/v0.1.0
