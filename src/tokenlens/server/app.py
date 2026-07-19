"""FastAPI application factory.

create_app(storage_backend) wires the configured StorageBackend and the
live-trace broadcaster into a FastAPI app, and serves the built React
dashboard from / when a built dashboard is present (a placeholder page
otherwise).
Requires the `server` extra: pip install "tokenlens[server]".
"""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from tokenlens import __version__
from tokenlens.core.collector import TraceCollector, get_collector
from tokenlens.server import routes
from tokenlens.server.live import LiveBroadcaster, LiveExporter
from tokenlens.storage import get_backend
from tokenlens.storage.base import StorageBackend

_DESCRIPTION = """\
A flamegraph for token spend — trace, attribute, and profile LLM pipeline costs.

Money fields are serialized as decimal strings (e.g. `"0.000125"`) to keep
exact micro-dollar precision; parse them as decimals, not floats.
"""

_OPENAPI_TAGS = [
    {"name": "traces", "description": "List traces and fetch individual span trees."},
    {"name": "aggregate", "description": "Cost breakdowns by user, feature, model, or node."},
    {"name": "stats", "description": "Headline totals for the dashboard."},
    {"name": "pricing", "description": "The active per-model billing rates."},
]

_PLACEHOLDER_HTML = """\
<!doctype html>
<title>tokenlens</title>
<style>
  body { font: 16px/1.6 system-ui, sans-serif; max-width: 40rem;
         margin: 6rem auto; padding: 0 1rem; }
  code { background: rgba(127, 127, 127, .15); padding: .1em .35em; border-radius: 4px; }
</style>
<h1>tokenlens</h1>
<p>The dashboard hasn't been built yet. From the repository root run:</p>
<p><code>cd dashboard &amp;&amp; npm install &amp;&amp; npm run build</code></p>
<p>Meanwhile the API is live — see the <a href="/docs">interactive API docs</a>.</p>
"""


def default_dashboard_dist() -> Path:
    env = os.environ.get("TOKENLENS_DASHBOARD_DIST")
    if env:
        return Path(env).expanduser()
    # Dev checkout: <root>/src/tokenlens/server/app.py → <root>/dashboard/dist.
    # The directory exists even before a build (it holds a tracked .gitkeep),
    # so require an actual build output.
    repo_dist = Path(__file__).resolve().parents[3] / "dashboard" / "dist"
    if (repo_dist / "index.html").is_file():
        return repo_dist
    # Installed wheel: the built dashboard ships inside the package.
    return Path(__file__).resolve().parent.parent / "_dashboard"


def create_app(
    storage_backend: StorageBackend | None = None,
    *,
    collector: TraceCollector | None = None,
    dashboard_dist: Path | None = None,
) -> FastAPI:
    """Build the tokenlens API server.

    storage_backend defaults to the process-wide backend (SQLite unless
    tokenlens.init() chose otherwise); collector defaults to the process-wide
    collector, whose flushed traces feed /ws/live.
    """
    backend = storage_backend if storage_backend is not None else get_backend()
    broadcaster = LiveBroadcaster()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        broadcaster.bind(asyncio.get_running_loop())
        exporter = LiveExporter(broadcaster)
        coll = collector if collector is not None else get_collector()
        coll.add_exporter(exporter)
        try:
            yield
        finally:
            coll.remove_exporter(exporter)

    app = FastAPI(
        title="tokenlens",
        version=__version__,
        description=_DESCRIPTION,
        openapi_tags=_OPENAPI_TAGS,
        lifespan=lifespan,
    )
    app.state.backend = backend
    app.state.broadcaster = broadcaster

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes.api)
    app.include_router(routes.ws)

    dist = dashboard_dist if dashboard_dist is not None else default_dashboard_dist()
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    else:

        @app.get("/", include_in_schema=False)
        async def dashboard_placeholder() -> HTMLResponse:
            return HTMLResponse(_PLACEHOLDER_HTML)

    return app
