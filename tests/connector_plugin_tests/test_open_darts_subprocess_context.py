"""Smoke test for the OpenDARTS plugin file's worker-subprocess context.

The plugin file at ``plugins/connectors/opendarts.py`` is staged into the
worker subprocess directory. The worker subprocess has ``src/`` on
``PYTHONPATH`` (injected by :func:`_build_env` in thread mode and via
``ENV PYTHONPATH=/app/src`` in Docker), so absolute imports work in both
contexts.

These tests simulate that contract by setting up a subprocess with ``src/``
on sys.path and the staged plugin file importable as ``connectors.opendarts``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from utils import find_repo_root


def _stage_subprocess_runtime(tmp_path: Path) -> Path:
    """Stage only the plugin file into a worker-temp-like connectors/ directory."""
    repo_root = find_repo_root()
    plugin_file = repo_root / "plugins" / "connectors" / "opendarts.py"

    staged_connectors = tmp_path / "connectors"
    staged_connectors.mkdir()
    (staged_connectors / "__init__.py").touch()
    shutil.copyfile(plugin_file, staged_connectors / "opendarts.py")
    return tmp_path


def _run_subprocess_check(
    staged_root: Path, snippet: str
) -> subprocess.CompletedProcess:
    """Run ``snippet`` in a fresh Python process that mimics the worker subprocess.

    ``src/`` is on sys.path (as injected by _build_env) and ``URGENT_WORKER_SUBPROCESS``
    is set so the simulation_service __init__ guard skips the heavy orchestrator imports.
    The staged dir is added so ``connectors.opendarts`` resolves.
    """
    repo_root = find_repo_root()
    src_path = str(repo_root / "src")
    bootstrap = textwrap.dedent(
        f"""
        import sys, os
        os.environ["URGENT_WORKER_SUBPROCESS"] = "1"
        if {src_path!r} not in sys.path:
            sys.path.insert(0, {src_path!r})
        sys.path.insert(0, {str(staged_root)!r})
        """
    ).strip()
    return subprocess.run(
        [sys.executable, "-c", f"{bootstrap}\n{snippet}"],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.skipif(
    not (find_repo_root() / "plugins" / "connectors" / "opendarts.py").is_file(),
    reason="Built-in OpenDARTS plugin file is missing",
)
def test_open_darts_plugin_imports_in_worker_subprocess(tmp_path: Path) -> None:
    """Plugin file must import correctly when src/ is on sys.path (worker subprocess contract)."""
    staged_root = _stage_subprocess_runtime(tmp_path)

    snippet = textwrap.dedent(
        """
        from connectors.opendarts import OpenDartsConnector, open_darts_input_configuration_injector
        print('name=' + OpenDartsConnector.ConnectorName)
        print('decorator_ok=' + str(callable(open_darts_input_configuration_injector)))
        OpenDartsConnector.broadcast_result('Heat', 1.5)
        """
    ).strip()

    result = _run_subprocess_check(staged_root, snippet)

    assert result.returncode == 0, (
        f"subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "name=opendarts" in result.stdout
    assert "decorator_ok=True" in result.stdout
    assert "OpenDartsConnector: Type:Heat, Value:1.5" in result.stdout


@pytest.mark.skipif(
    not (find_repo_root() / "plugins" / "connectors" / "opendarts.py").is_file(),
    reason="Built-in OpenDARTS plugin file is missing",
)
def test_open_darts_plugin_creates_descriptor_in_worker_subprocess(
    tmp_path: Path,
) -> None:
    """Plugin descriptor is created when src/ (and urgent_plugins) is on sys.path."""
    staged_root = _stage_subprocess_runtime(tmp_path)

    snippet = textwrap.dedent(
        """
        import connectors.opendarts as mod
        print('has_plugin=' + str(hasattr(mod, 'plugin')))
        """
    ).strip()

    result = _run_subprocess_check(staged_root, snippet)

    assert result.returncode == 0, (
        f"subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "has_plugin=True" in result.stdout
