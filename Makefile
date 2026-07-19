PY ?= python
PORT ?= 8321

.PHONY: help install test test-fast test-all test-cov lint typecheck check dashboard demo serve clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## editable install with every extra + dev tools
	pip install -e ".[all,dev]"

test: test-fast  ## alias for test-fast

test-fast:  ## unit + integration tests (skips e2e)
	pytest -m "not e2e"

test-all:  ## the whole suite, including e2e (real server + CLI)
	pytest

test-cov:  ## everything but the slow smoke test, with the 85% coverage gate
	pytest -m "not slow" --cov --cov-report=term-missing --cov-fail-under=85

lint:  ## ruff lint + format check
	ruff check .
	ruff format --check .

typecheck:  ## mypy (strict)
	mypy

check: lint typecheck test  ## everything CI runs

dashboard:  ## build the React dashboard into dashboard/dist
	cd dashboard && npm install && npm run build

demo:  ## seed ~20 demo traces, then serve the dashboard and open it
	$(PY) examples/demo_langgraph_agent.py
	( sleep 1.5; $(PY) -m webbrowser "http://127.0.0.1:$(PORT)" ) &
	tokenlens server --port $(PORT)

serve:  ## just serve the dashboard/API
	tokenlens server --port $(PORT)

clean:  ## remove build artifacts and caches
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache dashboard/dist
	find . -name __pycache__ -type d -not -path "./.venv/*" -not -path "./dashboard/node_modules/*" -exec rm -rf {} +
