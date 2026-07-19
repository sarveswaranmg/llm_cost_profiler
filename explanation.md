# tokenlens — explained

This document is the deeper "how and why" behind tokenlens, for anyone
picking up the project cold: what it is, the gap it fills, who it's for,
and how each layer is actually built. The [README](README.md) is the
pitch and quickstart; this is the design doc.

## What it is, in one paragraph

tokenlens is a Python library + local web app that traces LLM pipelines
(LangChain, LangGraph, raw OpenAI/Anthropic SDK calls, or hand-rolled code)
as a **tree of spans**, prices every span against a per-model rate table,
and renders the result as a **flamegraph** — the same mental model as a CPU
profiler, applied to token spend instead of CPU time. It installs with one
`pip install`, needs no account, no API key of its own, and no cloud
service: traces flush to a local SQLite file, and `tokenlens server` serves
both the API and the dashboard from that file.

## The gap it fills

Every LLM app eventually accumulates enough complexity — chains, agents,
retries, sub-calls — that a flat request log stops being useful for the one
question that actually matters: **where is the money going?**

Three things are consistently missing from the default tooling people reach
for (print statements, a logging table, or a generic APM tool):

1. **Cost has shape, but logs are flat.** A request isn't one number — it's
   a tree (router → retriever → two LLM calls → a retry). A flat log of
   "20 LLM calls, $4.12 total" can't tell you *which node* in that tree is
   expensive. Rolling cost up a span tree is exactly what a flamegraph does
   for CPU time; tokenlens does the same thing for dollars.
2. **Retries are invisible spend.** A retried call is money spent doing
   work that already failed once. Nothing about a typical log line marks a
   call as "attempt 2 of 2" — so retry waste hides inside the total instead
   of being called out as its own number.
3. **"Who/what spent this" requires manual plumbing.** Attributing cost to
   a user or a product feature normally means threading an id through every
   log call by hand. tokenlens stamps it once per request
   (`set_user`/`set_feature`) and every span underneath inherits it
   automatically, so "cost by feature" is a query, not an ETL job.

The broader observability players (LangSmith, Langfuse, Helicone) solve
adjacent problems well — prompt management, evals, hosted team dashboards —
but none of them treat **cost profiling** as the primary lens, and all of
them assume you want a cloud account or a deployment. tokenlens is
deliberately narrow: it does one thing (find out where the money went) and
optimizes the whole stack for it, starting from "it just works on my
laptop with no signup."

| | tokenlens | LangSmith | Langfuse | Helicone |
|---|---|---|---|---|
| Local-first, no account | SQLite file | cloud | self-host = a deployment | self-host = a deployment |
| Zero-config setup | one `pip install` | — | — | proxy swap |
| Cost flamegraph | native | — | — | — |
| Retry-waste accounting | headline metric | — | — | — |
| Per-user / per-feature cost | native | via metadata | native | native |
| Prompt mgmt, evals, datasets | — | native | native | partial |
| Team/collab, hosted UI | — | native | native | native |

If you need evals or a hosted team workspace, use one of those (tokenlens
happily coexists with them). If the question is "why did this request cost
$2, and is it the retries", that's the gap tokenlens fills.

## Who it's for / when to reach for it

- **Solo devs and small teams** shipping an LLM feature who want cost
  visibility without standing up infrastructure or handing prompt data to a
  third party.
- **Debugging a specific expensive request** — open the flamegraph for one
  trace and see immediately which node/retry/model dominates the cost,
  instead of guessing from a log line.
- **Catching retry storms** — a flaky upstream or an aggressive retry
  policy shows up as a red, striped subtree and a nonzero "retry waste"
  number, not silently inflated totals.
- **Chargeback / cost-by-feature reporting** — `tokenlens report --group-by
  feature` or `/api/aggregate?group_by=user_id` answers "which customer/
  feature is expensive" without bolting billing logic onto the app.
- **Choosing models with real numbers** — since every span carries actual
  token counts and a priced cost, `tokenlens pricing` plus the aggregate
  views let you compare what a cheaper model tier would actually have cost
  on real traffic, not a guess.
- **CI/local dev sanity checks** — `tokenlens trace <id>` gives an ASCII
  span tree with per-node cost in a terminal, no dashboard required.

It is explicitly *not* trying to be a prompt registry, an eval harness, or
a hosted multi-tenant product — see the comparison table above.

## Mental model: the pipeline

```
YOUR APP (LangChain / LangGraph / raw SDK / hand-written)
        │
        ▼
INSTRUMENTATION            LangChain callback handler
  (adapters convert         LangGraph node wrapper       ──┐
   framework hooks into     OpenAI/Anthropic monkey-patch   │
   tokenlens spans)         tokenlens.span() (manual)      ─┘
        │
        ▼
CONTEXT PROPAGATION        contextvars — current span/trace id,
                            user/feature/session attribution
        │
        ▼
TRACE COLLECTOR            buffers finished spans in memory,
  (in-process)              groups into trace trees, background-
                             flushes every few seconds + at exit
        │
        ▼
COST ENGINE                 prices each span against a per-model
                             rate table as it's recorded
        │
        ▼
STORAGE                     SQLite (default) or Postgres —
                             async StorageBackend protocol
        │
   ┌────┴─────┐
   ▼           ▼
FASTAPI      CLI
SERVER       (report / top / trace / prune / pricing)
(REST +
 /ws/live)
   │
   ▼
REACT DASHBOARD
(flamegraph · timeline · attribution)
```

The core insight that makes this a *profiler* rather than a *logger*: every
LLM call, retry, and graph node becomes a **span** in a **trace tree**
(parent-child relationships tracked explicitly). Cost rolls up the tree
exactly like CPU time rolls up in a flamegraph — that's what makes
rendering an actual flamegraph possible instead of a flat table.

## How it's built, layer by layer

### 1. Core tracing engine — `src/tokenlens/core/`

The data spine everything else builds on.

- **`span.py`** — `Span` (pydantic, `extra="forbid"` so a typo'd field
  fails loudly) with `SpanKind` (`LLM_CALL`, `CHAIN`, `GRAPH_NODE`, `TOOL`,
  `RETRIEVER`, `EMBEDDING`, `RETRY`, `CUSTOM`), timing, status/error, LLM
  token fields (input/output/cache-read/cache-write), attribution fields
  (`user_id`/`feature_tag`/`session_id`), and computed properties
  (`duration_ms`, `total_tokens`) that serialize automatically via
  `@computed_field`. `prompt_preview` is capped at 200 chars by a validator
  — spans never retain more raw prompt text than a preview needs. `Trace`
  assembles a flat span list into a tree (`to_tree()`) and aggregates it
  (`total_cost()`, `total_tokens()`).
- **`context.py`** — `tokenlens.span(name, kind=..., **attrs)`, a
  `contextvars`-based context manager: it nests under whatever span is
  currently open, inherits attribution from `set_user`/`set_feature`/
  `set_session` unless overridden, marks the span `ERROR` on an exception
  while still closing it, and reports the finished span to the collector.
  Because it's built on `contextvars` rather than a manually-threaded
  stack, nesting is correct across `await` boundaries, `asyncio.gather`
  fan-out (each task gets its own copy of the context), and threads —
  without the caller doing anything special.
- **`collector.py`** — `TraceCollector` buffers finished spans in memory
  keyed by trace id, runs a pluggable chain of **enrichers** on each span as
  it's recorded (the default enricher fills in `cost_usd` from the pricing
  table — failures here are caught and logged, never allowed to break the
  host app), and hands complete traces to pluggable **exporters** on
  `flush()`. A trace is "complete" once its root span (no parent) has
  closed, which relies on children closing before their parent's `with`
  block exits — true for nested/awaited spans. Flushing runs on a
  background daemon thread every few seconds and once more at interpreter
  exit, so instrumentation never blocks the host app's request path.

### 2. Instrumentation adapters — `src/tokenlens/instrument/`

The hooks that capture spans without the caller changing much code. All
four adapters produce the identical `Span`/`Trace` model, so they compose
freely — a LangGraph node that calls LangChain that calls the OpenAI SDK
nests into one coherent trace.

- **`langchain.py`** — `TokenLensCallbackHandler(BaseCallbackHandler)`.
  Deliberately does **not** use `tokenlens.span()`'s contextvars stack:
  LangChain dispatches callbacks in whatever order its own execution engine
  runs steps, which isn't guaranteed to be a well-nested call stack. Instead
  it keys spans directly by LangChain's own `run_id`/`parent_run_id`
  (`span_id = str(run_id)`), which guarantees the resulting tree matches
  LangChain's actual execution graph regardless of dispatch order. Token
  usage is extracted from **both** shapes LangChain uses across versions —
  `llm_output["token_usage"]` (older/OpenAI-style) and
  `usage_metadata` on the generation's message (newer, includes Anthropic
  cache-read/cache-write tokens). Retries are tracked with a per-parent
  attempt counter: LangChain's `.with_retry()` reuses one `parent_run_id`
  across attempts with a fresh `run_id` each time, so counting attempts per
  parent gives a correct `retry_index` without needing to inspect tenacity
  internals.
- **`langgraph.py`** — two integration modes. `instrument_graph(graph)`
  wraps each compiled node's underlying sync/async function in place (state
  passed between nodes is only observed, never mutated) so every node
  execution becomes a `GRAPH_NODE` span; `@traced_node("name")` is the same
  wrapping as a plain decorator for hand-rolled node functions. Unlike the
  LangChain handler, this *does* use `tokenlens.span()`'s contextvars stack
  directly — LangGraph nodes execute within the same call stack (sync) or a
  context-propagating asyncio task (async) as whatever invoked them, so
  nesting is correct even under parallel fan-out. It intentionally does
  *not* auto-wrap the whole graph invocation: wrap
  `compiled_graph.invoke(...)` in `tokenlens.span(...)` yourself so node
  spans land in one trace — explicit beats hidden magic.
- **`openai_sdk.py` / `anthropic_sdk.py`** — `tokenlens.instrument.auto_patch()`
  monkey-patches `Completions.create`/`Messages.create` (sync + async) at
  the class level, so every client instance is covered after one call.
  Streaming responses are returned as **wrapped generators**: usage is
  accumulated as chunks/events are pulled through (OpenAI's final chunk
  with `stream_options={"include_usage": True}`; Anthropic's
  `message_start` + `message_delta` events), and the span only closes once
  the stream is fully consumed — the wrapper never forces materialization
  of the underlying stream. Both SDKs are imported lazily inside `patch()`,
  so importing tokenlens (or calling `auto_patch()` with only one SDK
  installed) never requires the other to exist.

### 3. Cost engine — `src/tokenlens/cost/`

- **`pricing.py`** — `PricingTable` loads a versioned JSON rate table (USD
  per **million** tokens) and resolves a raw model name forgivingly: exact
  match → alias → longest prefix ending at a word boundary (so
  `gpt-4o-mini-2024-07-18` bills as `gpt-4o-mini` without an explicit
  alias entry). All money math is `Decimal`, quantized to 6 decimal places
  (micro-dollar precision) — floats never touch a cost calculation. An
  unknown model logs one warning per name and prices as `null`; pricing
  gaps must never crash the caller's app. Custom tables (negotiated rates,
  internal models) load via `load_pricing(path)` or
  `$TOKENLENS_PRICING_PATH` and replace the bundled table for the process.
- **`attribution.py`** — the aggregation layer over collected traces:
  `cost_by(traces, key)` breaks cost/tokens/call-count down by user,
  feature, model, node name, or span kind; `retry_waste(traces)` sums the
  cost of every span with `retry_index > 0` — the headline "money spent
  redoing failed work" number; `trace_flame_data(trace)` converts a trace
  into the `{name, value, children, data}` shape d3-flamegraph expects,
  where `value` is the *inclusive* cost of the subtree in integer
  micro-dollars so a parent frame always covers its children.

### 4. Storage — `src/tokenlens/storage/`

An async `StorageBackend` protocol (`save_traces`, `get_trace`,
`list_traces`, `aggregate`, `overview`, `prune`, `close`) implemented by:

- **`sqlite.py`** — the zero-config default (`~/.tokenlens/traces.db`),
  good for single-machine/single-team volumes.
- **`postgres.py`** (asyncpg) — for concurrent writer processes or a shared
  team dashboard; tables and indexes are created automatically and
  idempotently on first use.

Both backends compute `aggregate()`/`overview()` **in SQL**, summing cost
as integer micro-dollars so API/CLI totals are exact and match the Python
`attribution.cost_by()` math bit-for-bit — money in the API is serialized
as decimal strings for the same reason (never parse it as a float). A
shared set of row⇄model conversion helpers in `base.py` keeps both backends
serializing spans identically. `tokenlens.init(storage=...)` is the single
call that picks a backend and rewires the collector's exporters to it.

### 5. API server — `src/tokenlens/server/`

A FastAPI app (`create_app()`) exposing:

- `GET /api/traces`, `/api/traces/{id}`, `/api/traces/{id}/flame` — list,
  full span tree, d3-flamegraph-shaped export.
- `GET /api/aggregate?group_by=...`, `/api/stats/overview` — cost
  breakdowns and dashboard headline stats.
- `GET /api/pricing` — the currently active rate table.
- `WS /ws/live` — pushes a trace summary the moment the collector flushes
  it, so the dashboard updates live without polling (best-effort: messages
  drop rather than queue unboundedly for a slow client).

It also serves the built React dashboard as static files from the same
port (falling back to a "build the dashboard" placeholder page if
`dashboard/dist` doesn't exist yet). CORS is restricted to localhost
origins — tokenlens has no authentication of its own, so anything beyond a
local/trusted network needs a reverse proxy in front of it.

### 6. Dashboard — `dashboard/`

React 18 + TypeScript + Vite + Tailwind v4, with `d3-hierarchy`/`d3-scale`
driving the flamegraph layout and `recharts` for the attribution charts.
Three views over one selected trace (`App.tsx` keeps the tab and trace id
in the URL, so a view is shareable/deep-linkable):

- **Flamegraph** — block width proportional to cost (toggle to
  duration/tokens); retried spans render red and striped; click any frame
  for full detail.
- **Timeline** — the same trace as a Gantt-style waterfall, showing
  parallelism (useful for spotting sequential calls that could run
  concurrently).
- **Attribution** — headline totals, retry waste, error rate, cost broken
  down by feature/user/model, and a sortable per-node cost table.

The sidebar lists traces live (via `/ws/live`) with filters; a detail
drawer shows the selected span's full JSON.

### 7. CLI — `src/tokenlens/cli.py`

Typer + Rich, sharing the same storage-resolution rule as the library
(`--db` → `$TOKENLENS_PG_DSN` → `$TOKENLENS_DB_PATH`/default SQLite file):

```
tokenlens server                              # API + dashboard
tokenlens report --since 7d --group-by node   # cost table (or --json)
tokenlens top --n 10                          # most expensive traces
tokenlens trace <trace_id>                    # ASCII span tree, retries in red
tokenlens prune --older-than 30d              # retention
tokenlens pricing                             # the active rate table
```

`tokenlens trace` in particular is the "no dashboard needed" path: a full
ASCII tree with per-span cost, tokens, duration, and retry markers, useful
in CI logs or over SSH.

## Design principles that shaped the build

- **Local-first, zero-config.** No account, no API key of tokenlens's own,
  no daemon. Default storage is one SQLite file; the server reads the same
  file the library writes.
- **Never block or break the host app.** Flushing is background and
  periodic; cost enrichment failures are caught and logged, not raised;
  unknown models degrade to `cost_usd = null` instead of an exception.
- **Exact money math.** `Decimal` everywhere, quantized to micro-dollars,
  summed as integers in SQL — API responses serialize money as strings so
  clients can't silently lose precision by parsing as float.
- **Correctness under concurrency, for free.** `contextvars` (not a
  manually threaded stack or thread-locals) is what makes span nesting
  correct across `await`, parallel `asyncio.gather`/LangGraph fan-out, and
  threads without the caller doing anything special.
- **One data model, many producers.** Every instrumentation path — a
  LangChain callback, a LangGraph node wrapper, a monkey-patched SDK call,
  or a hand-written `tokenlens.span()` — produces the same `Span`/`Trace`
  shape, so they compose into one trace and every downstream consumer
  (storage, API, CLI, dashboard) only has to understand one model.
- **Forgiving, not silent, about pricing gaps.** An unpriced model doesn't
  crash anything, but it does log — a `—` cost in the dashboard is a
  visible prompt to add the model to the pricing table, not a swallowed
  bug.

## Testing strategy

Tests are tiered by directory and auto-marked accordingly
(`tests/unit`, `tests/integration`, `tests/e2e`; `pytest -m "not e2e"` is
the fast default):

- **`unit/`** — components in isolation: span/collector/context behavior
  (including a dedicated thread-safety test), cost pricing and
  attribution math, each instrumentation adapter (including streaming
  accumulation and lazy-iterator behavior), property-based tests
  (`hypothesis`) for invariants like "cost never goes negative", and
  lazy-import guarantees (tokenlens must import fine with zero optional
  extras installed).
- **`integration/`** — instrumentation → collector → storage → API working
  together against real (SQLite/Postgres) storage and a real ASGI test
  client, plus a contract test asserting the CLI's numbers and the API's
  numbers agree on the same data.
- **`e2e/`** — a real server subprocess and the actual CLI binary, smoke-
  testing the full path a user would take.

`tests/llm_mocks.py` and `tests/factories.py` centralize mocked
OpenAI/Anthropic response shapes and a canonical hand-computed sample trace
(with expected costs pre-derived), so cost-math assertions aren't
re-deriving arithmetic in every test.

## Repository map

```
src/tokenlens/
  core/          Span/Trace model, contextvars propagation, TraceCollector
  instrument/    LangChain, LangGraph, OpenAI, Anthropic adapters
  cost/          pricing table + cost math, attribution/aggregation
  storage/       StorageBackend protocol, SQLite + Postgres implementations
  server/        FastAPI app, REST routes, /ws/live broadcaster
  cli.py         typer CLI (server/report/top/trace/prune/pricing)
dashboard/       React + TS + Vite + Tailwind dashboard (flamegraph, timeline, attribution)
examples/        demo_rag_app.py, demo_langgraph_agent.py — fake-LLM traffic generators
docs/            quickstart, integrations, custom-pricing, self-hosting
tests/           unit / integration / e2e, plus shared factories and LLM mocks
```

## Try it

```bash
pip install -e ".[all,dev]"
python examples/demo_langgraph_agent.py   # seeds ~20 realistic traces, no API keys
tokenlens server                          # dashboard → http://127.0.0.1:8321
# or: make demo
```
