# tokenlens dashboard

Dark-theme React dashboard for [tokenlens](../README.md): a live trace list,
a cost **flamegraph**, a Gantt-style **timeline**, and an **attribution**
view (who/what is spending the budget). Vite + React + TypeScript +
Tailwind, charts via d3-hierarchy (flamegraph) and recharts (bars).

## Dev workflow

Two terminals:

```bash
# 1. the API (serves traces from ~/.tokenlens/traces.db)
tokenlens server --port 8321

# 2. the dashboard with hot reload
cd dashboard
npm install
npm run dev            # → http://localhost:5173
```

`vite.config.ts` proxies `/api/*` and the `/ws/live` WebSocket to
`http://localhost:8321`, so the app always talks same-origin relative URLs.
No traces yet? Generate sample traffic:

```bash
python examples/demo_rag_app.py
```

### Pointing at a different API

Set `VITE_TOKENLENS_API` at build/dev time to skip the proxy and talk to an
absolute base URL (CORS for localhost origins is already enabled server-side):

```bash
VITE_TOKENLENS_API=http://localhost:8321 npm run dev
```

## Production build

```bash
npm run build          # type-checks, then emits dashboard/dist/
```

The FastAPI server automatically serves `dashboard/dist/` at `/` when it
exists (override the location with `TOKENLENS_DASHBOARD_DIST`). So after a
build, `tokenlens server` is the only process you need.

## Layout

- **Sidebar** — live-updating trace list (`/ws/live`), newest first, with
  cost color-coding (green < $0.01, yellow < $0.10, red ≥ $0.10), token
  counts, error badges, and user/feature chips. Filter by user, feature,
  model, and minimum cost.
- **Flamegraph** — the span tree as an icicle; block width ∝ cost (toggle
  to duration or tokens). Colors by span kind; retries are red *and*
  striped. Hover for details, click for the full span drawer with metadata,
  prompt preview, and copyable JSON.
- **Timeline** — the same trace as a waterfall (x = wall-clock time),
  which shows parallelism and where latency went.
- **Attribution** — totals, retry waste, error rate, cost-by
  feature/user/model bars, and a sortable per-node cost table across all
  stored traces.
