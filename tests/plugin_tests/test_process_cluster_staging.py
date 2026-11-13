from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from services.simulation_service.core.service.process_cluster_manager import (
    ProcessClusterManager,
)

_STUB_PLUGIN_SRC = """
from urgent_plugins import ConnectorPlugin

# Intentionally not a real ConnectorInterface — staging copies the file
# verbatim and does not import it during the copy step.
plugin = "marker-not-a-descriptor"
"""


@pytest.fixture
def env_isolated(monkeypatch):
    for key in (
        "URGENT_CONNECTOR_PLUGIN_NAME",
        "URGENT_PLUGIN_PATH",
        "URGENT_RUN_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch


def _make_manager() -> ProcessClusterManager:
    return object.__new__(ProcessClusterManager)


def test_stage_connector_plugin_copies_file(tmp_path: Path, env_isolated) -> None:
    """Connector plugin is staged from URGENT_PLUGIN_PATH."""
    plugins_root = tmp_path / "ext_plugins"
    (plugins_root / "connectors").mkdir(parents=True)
    src = plugins_root / "connectors" / "stub_plugin.py"
    src.write_text(textwrap.dedent(_STUB_PLUGIN_SRC))

    env_isolated.setenv("URGENT_CONNECTOR_PLUGIN_NAME", "stub_plugin")
    env_isolated.setenv("URGENT_PLUGIN_PATH", str(plugins_root))

    target = tmp_path / "worker"
    (target / "connectors").mkdir(parents=True)
    manager = _make_manager()

    manager._stage_connector_plugin(target)

    staged = target / "connectors" / "stub_plugin.py"
    assert staged.is_file()
    assert staged.read_text() == src.read_text()


def test_stage_connector_plugin_missing_file_raises(
    tmp_path: Path, env_isolated
) -> None:
    """Third-party connector file not found → FileNotFoundError."""
    env_isolated.setenv("URGENT_CONNECTOR_PLUGIN_NAME", "missing")
    env_isolated.setenv("URGENT_PLUGIN_PATH", str(tmp_path))

    target = tmp_path / "worker"
    (target / "connectors").mkdir(parents=True)
    manager = _make_manager()

    with pytest.raises(FileNotFoundError):
        manager._stage_connector_plugin(target)


def test_stage_connector_plugin_requires_plugin_path(
    tmp_path: Path, env_isolated
) -> None:
    """Connector without URGENT_PLUGIN_PATH -> RuntimeError."""
    env_isolated.setenv("URGENT_CONNECTOR_PLUGIN_NAME", "stub_plugin")
    # URGENT_PLUGIN_PATH intentionally not set

    target = tmp_path / "worker"
    (target / "connectors").mkdir(parents=True)
    manager = _make_manager()

    with pytest.raises(RuntimeError, match="URGENT_PLUGIN_PATH"):
        manager._stage_connector_plugin(target)


def test_stage_connector_plugin_copies_package(tmp_path: Path, env_isolated) -> None:
    """Package plugins (<name>/__init__.py) are staged as a whole directory."""
    plugins_root = tmp_path / "ext_plugins"
    pkg = plugins_root / "connectors" / "pkg_conn"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_text(textwrap.dedent(_STUB_PLUGIN_SRC))
    (pkg / "helpers.py").write_text("VALUE = 1\n")
    (pkg / "__pycache__" / "junk.pyc").write_bytes(b"x")

    env_isolated.setenv("URGENT_CONNECTOR_PLUGIN_NAME", "pkg_conn")
    env_isolated.setenv("URGENT_PLUGIN_PATH", str(plugins_root))

    target = tmp_path / "worker"
    (target / "connectors").mkdir(parents=True)

    _make_manager()._stage_connector_plugin(target)

    staged = target / "connectors" / "pkg_conn"
    assert (staged / "__init__.py").is_file()
    assert (staged / "helpers.py").read_text() == "VALUE = 1\n"
    assert not (staged / "__pycache__").exists()


@pytest.mark.parametrize("reserved", ["common", "runner", "factory", "conn_utils"])
def test_stage_connector_plugin_rejects_framework_module_names(
    tmp_path: Path, env_isolated, reserved: str
) -> None:
    plugins_root = tmp_path / "ext_plugins"
    (plugins_root / "connectors").mkdir(parents=True)
    (plugins_root / "connectors" / f"{reserved}.py").write_text(_STUB_PLUGIN_SRC)

    env_isolated.setenv("URGENT_CONNECTOR_PLUGIN_NAME", reserved)
    env_isolated.setenv("URGENT_PLUGIN_PATH", str(plugins_root))

    target = tmp_path / "worker"
    (target / "connectors").mkdir(parents=True)

    with pytest.raises(ValueError, match="collides with a framework module"):
        _make_manager()._stage_connector_plugin(target)
