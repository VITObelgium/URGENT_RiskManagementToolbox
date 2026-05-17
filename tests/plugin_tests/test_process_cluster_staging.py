from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.simulation_service.core.service.process_cluster_manager import (
    ProcessClusterManager,
)

_REGISTRY_MOD = "urgent_plugins.registry"

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


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Minimal fake repo root with a populated plugins/connectors/ directory."""
    connectors_dir = tmp_path / "plugins" / "connectors"
    connectors_dir.mkdir(parents=True)
    (connectors_dir / "__init__.py").write_text("")
    (connectors_dir / "fake_builtin.py").write_text("# fake built-in connector")
    return tmp_path


def _make_manager() -> ProcessClusterManager:
    return object.__new__(ProcessClusterManager)


def _mock_registry(is_builtin: bool):
    """Return two context managers that mock register_builtins + get_registry."""
    mock_reg = MagicMock()
    mock_reg.get.return_value = MagicMock() if is_builtin else None
    return (
        patch(f"{_REGISTRY_MOD}.register_builtins"),
        patch(f"{_REGISTRY_MOD}.get_registry", return_value=mock_reg),
    )


def test_stage_connector_plugin_builtin_stages_to_connectors_dir(
    fake_repo: Path, tmp_path: Path, env_isolated
) -> None:
    """Built-in connector: files staged to connectors/, no plugins/ dir created."""
    env_isolated.setenv("URGENT_CONNECTOR_PLUGIN_NAME", "opendarts")

    target = tmp_path / "worker"
    (target / "connectors").mkdir(parents=True)

    manager = _make_manager()
    p1, p2 = _mock_registry(is_builtin=True)
    with p1, p2:
        manager._stage_connector_plugin(fake_repo, target)

    assert (target / "connectors" / "fake_builtin.py").is_file()
    assert not (target / "plugins").exists()


def test_stage_connector_plugin_copies_file(
    fake_repo: Path, tmp_path: Path, env_isolated
) -> None:
    """Third-party connector: staged to plugins/connectors/ from URGENT_PLUGIN_PATH."""
    plugins_root = tmp_path / "ext_plugins"
    (plugins_root / "connectors").mkdir(parents=True)
    src = plugins_root / "connectors" / "stub_plugin.py"
    src.write_text(textwrap.dedent(_STUB_PLUGIN_SRC))

    env_isolated.setenv("URGENT_CONNECTOR_PLUGIN_NAME", "stub_plugin")
    env_isolated.setenv("URGENT_PLUGIN_PATH", str(plugins_root))

    target = tmp_path / "worker"
    (target / "connectors").mkdir(parents=True)
    manager = _make_manager()

    p1, p2 = _mock_registry(is_builtin=False)
    with p1, p2:
        manager._stage_connector_plugin(fake_repo, target)

    staged = target / "plugins" / "connectors" / "stub_plugin.py"
    assert staged.is_file()
    assert staged.read_text() == src.read_text()


def test_stage_connector_plugin_missing_file_raises(
    fake_repo: Path, tmp_path: Path, env_isolated
) -> None:
    """Third-party connector file not found → FileNotFoundError."""
    env_isolated.setenv("URGENT_CONNECTOR_PLUGIN_NAME", "missing")
    env_isolated.setenv("URGENT_PLUGIN_PATH", str(tmp_path))

    target = tmp_path / "worker"
    (target / "connectors").mkdir(parents=True)
    manager = _make_manager()

    p1, p2 = _mock_registry(is_builtin=False)
    with p1, p2:
        with pytest.raises(FileNotFoundError):
            manager._stage_connector_plugin(fake_repo, target)


def test_stage_connector_plugin_requires_plugin_path(
    fake_repo: Path, tmp_path: Path, env_isolated
) -> None:
    """Third-party connector without URGENT_PLUGIN_PATH → RuntimeError."""
    env_isolated.setenv("URGENT_CONNECTOR_PLUGIN_NAME", "stub_plugin")
    # URGENT_PLUGIN_PATH intentionally not set

    target = tmp_path / "worker"
    (target / "connectors").mkdir(parents=True)
    manager = _make_manager()

    p1, p2 = _mock_registry(is_builtin=False)
    with p1, p2:
        with pytest.raises(RuntimeError, match="URGENT_PLUGIN_PATH"):
            manager._stage_connector_plugin(fake_repo, target)
