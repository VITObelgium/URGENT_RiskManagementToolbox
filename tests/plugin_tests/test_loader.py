from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from urgent_plugins import (
    ConnectorPlugin,
    OptimizerPlugin,
    PluginKind,
    get_registry,
    normalize_plugin_name,
)

from urgent_plugins.loader import load_local_plugin, resolve_plugin_path

from ._builtin_classes import (
    get_builtin_connector_class,
    get_builtin_optimizer_class,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    get_registry().clear()
    yield
    get_registry().clear()


@pytest.fixture
def registry():
    return get_registry()


def _write_plugin(plugins_root: Path, kind: PluginKind, name: str, body: str) -> Path:
    path = resolve_plugin_path(kind, name, plugins_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return path


_VALID_CONNECTOR_TEMPLATE = """
from __future__ import annotations

import threading

from services.simulation_service.core.connectors.common import (
    ConnectorInterface,
    JsonPath,
    SimulationResults,
    SimulationStatus,
)
from urgent_plugins import ConnectorPlugin


class _StubConnector(ConnectorInterface):
    @staticmethod
    def run(
        config_path: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        return SimulationStatus.SUCCESS, {{"answer": 42.0}}


plugin = ConnectorPlugin(name="{name}", implementation=_StubConnector)
"""


def test_normalize_plugin_name_rejects_invalid() -> None:
    assert normalize_plugin_name("Eclipse") == "eclipse"
    assert normalize_plugin_name("my-plugin") == "my_plugin"
    with pytest.raises(ValueError):
        normalize_plugin_name("")
    with pytest.raises(ValueError):
        normalize_plugin_name("1bad-start")
    with pytest.raises(ValueError):
        normalize_plugin_name("has space")


def test_load_local_plugin_success(tmp_path: Path, registry) -> None:
    _write_plugin(
        tmp_path,
        PluginKind.CONNECTOR,
        "stub_conn",
        _VALID_CONNECTOR_TEMPLATE.format(name="stub_conn"),
    )

    descriptor = load_local_plugin(PluginKind.CONNECTOR, "stub_conn", tmp_path)

    assert isinstance(descriptor, ConnectorPlugin)
    assert descriptor.name == "stub_conn"
    registered = registry.require(PluginKind.CONNECTOR, "stub_conn")
    assert registered is descriptor

    status, results = descriptor.implementation.run("/tmp/cfg.json", {})
    assert status.name == "SUCCESS"
    assert results == {"answer": 42.0}


def test_load_local_plugin_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_local_plugin(PluginKind.CONNECTOR, "nope", tmp_path)


def test_load_local_plugin_registers_under_descriptor_name(tmp_path: Path) -> None:
    """Loader registers under descriptor.name even when it differs from the filename.

    The descriptor's own name (sourced from the implementation class's
    ConnectorName/EngineName attribute) is the source of truth — this lets the
    built-in OpenDARTS plugin lives at ``opendarts.py`` and registers
    under the canonical name ``opendarts``.
    """
    _write_plugin(
        tmp_path,
        PluginKind.CONNECTOR,
        "filename_stem",
        _VALID_CONNECTOR_TEMPLATE.format(name="canonical_name"),
    )

    descriptor = load_local_plugin(PluginKind.CONNECTOR, "filename_stem", tmp_path)
    assert descriptor.name == "canonical_name"
    assert get_registry().get(PluginKind.CONNECTOR, "canonical_name") is descriptor
    assert get_registry().get(PluginKind.CONNECTOR, "filename_stem") is None


def test_load_local_plugin_missing_descriptor(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        PluginKind.CONNECTOR,
        "no_descriptor",
        "x = 1\n",
    )

    with pytest.raises(AttributeError):
        load_local_plugin(PluginKind.CONNECTOR, "no_descriptor", tmp_path)


def test_load_local_plugin_wrong_descriptor_type(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        PluginKind.CONNECTOR,
        "wrong_type",
        "plugin = object()\n",
    )

    with pytest.raises(TypeError):
        load_local_plugin(PluginKind.CONNECTOR, "wrong_type", tmp_path)


def test_bundled_plugins_load_explicitly_for_each_kind(registry) -> None:
    connector_cls = get_builtin_connector_class()
    optimizer_cls = get_builtin_optimizer_class()

    connector = registry.require(PluginKind.CONNECTOR, connector_cls.ConnectorName)  # type: ignore[attr-defined]
    optimizer = registry.require(PluginKind.OPTIMIZER, optimizer_cls.EngineName)  # type: ignore[attr-defined]

    get_builtin_connector_class()
    get_builtin_optimizer_class()

    assert (
        registry.require(
            PluginKind.CONNECTOR, connector_cls.ConnectorName
        ).implementation  # type: ignore[attr-defined]
        is connector.implementation
    )
    assert (
        registry.require(PluginKind.OPTIMIZER, optimizer_cls.EngineName).implementation  # type: ignore[attr-defined]
        is optimizer.implementation
    )
    assert isinstance(connector, ConnectorPlugin)
    assert isinstance(optimizer, OptimizerPlugin)
