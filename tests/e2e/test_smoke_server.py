"""End-to-end smoke test: the real demo app seeds a real DB, a real uvicorn
server serves it, and the overview endpoint reports the numbers the README
promises (cost > 0, retry waste > 0, the injected errors).

Marked slow — the demo generates ~20 traces and the server boots in a
subprocess. Runs in CI's e2e job; skipped by `make test-fast` and the
coverage gate run.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def test_demo_to_real_server_to_overview(tmp_path) -> None:
    env = {**os.environ, "TOKENLENS_DB_PATH": str(tmp_path / "smoke.db")}
    env.pop("TOKENLENS_PG_DSN", None)

    # 1. Seed: run the demo agent exactly as a user would.
    demo = subprocess.run(
        [sys.executable, "examples/demo_langgraph_agent.py"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert demo.returncode == 0, demo.stderr

    # 2. Serve: the real CLI entry point, real uvicorn, real port.
    port = _free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from tokenlens.cli import app; app()",
            "server",
            "--port",
            str(port),
        ],
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        # 3. Poll until traces appear (with a hard timeout).
        overview: dict | None = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                data = _get_json(f"{base}/api/stats/overview")
                if data["trace_count"] >= 20:
                    overview = data
                    break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(0.5)

        assert overview is not None, "server never reported the demo's traces"
        assert overview["trace_count"] >= 20
        assert float(overview["total_cost_usd"]) > 0
        assert float(overview["retry_waste_usd"]) > 0  # the simulated 429 retries
        assert overview["error_rate"] > 0  # the injected failing run
        assert len(overview["top_traces"]) == 5

        # 4. The dashboard route serves something tokenlens-shaped (the
        #    built dashboard, or the "not built" placeholder).
        with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
            assert resp.status == 200
            assert b"tokenlens" in resp.read()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
