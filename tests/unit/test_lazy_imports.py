"""Optional dependencies stay optional.

Runs a fresh interpreter with openai/anthropic/langchain/langgraph imports
force-blocked and proves that `import tokenlens`, core tracing, and
auto_patch() all still work. In-process sys.modules patching can't prove
this (other tests import the real SDKs), so a subprocess is the honest way.
"""

import os
import subprocess
import sys
import textwrap

_BLOCKED_SDKS_SCRIPT = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder

    BLOCKED = {"openai", "anthropic", "langchain", "langchain_core", "langgraph"}

    class Blocker(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in BLOCKED:
                raise ImportError(f"blocked for test: {fullname}")
            return None

    sys.meta_path.insert(0, Blocker())

    import tokenlens
    from tokenlens.instrument import auto_patch

    auto_patch()  # must silently skip both uninstallable SDKs

    with tokenlens.span("root"):
        with tokenlens.span("child"):
            pass

    traces = tokenlens.get_collector().flush()
    assert len(traces) == 1, traces
    assert len(traces[0].spans) == 2, traces[0].spans

    leaked = {m for m in sys.modules if m.split(".")[0] in BLOCKED}
    assert not leaked, f"tokenlens imported blocked SDKs: {leaked}"
    print("LAZY-IMPORTS-OK")
    """
)


def test_core_tracing_works_without_any_llm_sdk_installed(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_SDKS_SCRIPT],
        env={**os.environ, "TOKENLENS_DB_PATH": str(tmp_path / "lazy.db")},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "LAZY-IMPORTS-OK" in result.stdout


def test_importing_tokenlens_does_not_import_llm_sdks(tmp_path) -> None:
    code = textwrap.dedent(
        """
        import sys
        import tokenlens  # noqa: F401

        heavy = {"openai", "anthropic", "langchain_core", "langgraph", "fastapi"}
        loaded = {m.split(".")[0] for m in sys.modules} & heavy
        assert not loaded, f"import tokenlens eagerly pulled in: {loaded}"
        print("IMPORT-CLEAN")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TOKENLENS_DB_PATH": str(tmp_path / "clean.db")},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "IMPORT-CLEAN" in result.stdout
