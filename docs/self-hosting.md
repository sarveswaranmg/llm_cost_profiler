# Self-hosting

The default setup — SQLite + `tokenlens server` on your machine — needs no
hosting at all. This page is for shared/team deployments.

## Postgres

For concurrent writers (several app processes) or a shared team dashboard,
use the Postgres backend:

```bash
pip install "tokenlens[postgres]"
export TOKENLENS_PG_DSN="postgresql://tokenlens:secret@db.internal:5432/tokenlens"
```

```python
import tokenlens

tokenlens.init(storage="postgres")          # reads TOKENLENS_PG_DSN
# or explicitly:
tokenlens.init(storage="postgres", dsn="postgresql://...")
```

Tables (`traces`, `spans`, indexes, a `tokenlens_meta` schema-version row)
are created automatically on first use — idempotent, safe to run from
multiple processes. The CLI and server pick up the same DSN:

```bash
tokenlens server --port 8321      # uses TOKENLENS_PG_DSN when set
tokenlens report --since 7d
```

The Postgres test suite runs only when a DSN is present:

```bash
TOKENLENS_PG_DSN=postgresql://localhost/tokenlens_test pytest tests/test_storage_postgres.py
```

## Serving the dashboard

`tokenlens server` serves the REST API, the `/ws/live` WebSocket, and the
built dashboard from one port. The dashboard is static files — build once,
serve anywhere:

```bash
cd dashboard && npm install && npm run build   # → dashboard/dist
tokenlens server --host 0.0.0.0 --port 8321
```

`TOKENLENS_DASHBOARD_DIST` overrides where the server looks for the built
files (useful in containers). CORS is enabled for localhost origins only;
put the server behind your usual reverse proxy / auth layer for anything
beyond localhost — **tokenlens has no authentication of its own**, and
traces contain prompt previews.

## Programmatic serving

```python
from tokenlens.server.app import create_app
from tokenlens.storage.postgres import PostgresBackend

app = create_app(PostgresBackend())   # any ASGI server: uvicorn, hypercorn...
```

## Retention

Traces accumulate forever by default. Cron something like:

```bash
tokenlens prune --older-than 30d --yes
```

or from Python: `tokenlens.storage.prune(older_than_days=30)`.

## Sizing notes

- A typical trace is a handful of KB (spans keep a 200-char prompt preview,
  never full prompts).
- The collector buffers in memory and flushes every few seconds; a crashed
  process loses at most the unflushed window.
- One SQLite file comfortably handles single-team local volumes; move to
  Postgres when several processes write concurrently or the dashboard is
  shared.
