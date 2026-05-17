from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from services.simulation_service.core.connectors.common import (
    ConnectorInterface,
    SimulationStatus,
)
from services.simulation_service.core.connectors.factory import ConnectorFactory
from urgent_plugins import (
    PluginKind,
    get_registry,
    register_builtins,
)
from urgent_plugins.loader import load_local_plugin, resolve_plugin_path

from ._builtin_classes import get_builtin_connector_class


@pytest.fixture(autouse=True)
def _reset_registry():
    get_registry().clear()
    yield
    get_registry().clear()


_STUB_PLUGIN_SRC = """
from __future__ import annotations

import threading

from services.simulation_service.core.connectors.common import (
    ConnectorInterface,
    JsonPath,
    SimulationResults,
    SimulationStatus,
)
from urgent_plugins import ConnectorPlugin


class StubConnector(ConnectorInterface):
    @staticmethod
    def run(
        config_path: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        return SimulationStatus.SUCCESS, {"answer": 1.0}


plugin = ConnectorPlugin(name="stub_plugin", implementation=StubConnector)
"""


def _write_stub_plugin(plugins_root: Path) -> Path:
    path = resolve_plugin_path(PluginKind.CONNECTOR, "stub_plugin", plugins_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(_STUB_PLUGIN_SRC))
    return path


def test_factory_returns_built_in_opendarts_for_explicit_name() -> None:
    register_builtins()
    builtin_cls = get_builtin_connector_class()
    for name in (builtin_cls.ConnectorName, builtin_cls.ConnectorName.upper()):  # type: ignore[attr-defined]
        connector = ConnectorFactory.get_connector(name)
        assert isinstance(connector, builtin_cls)


def test_factory_raises_for_empty_name() -> None:
    register_builtins()
    with pytest.raises(ValueError):
        ConnectorFactory.get_connector("")


def test_factory_returns_registered_plugin_instance(tmp_path: Path) -> None:
    _write_stub_plugin(tmp_path)
    descriptor = load_local_plugin(PluginKind.CONNECTOR, "stub_plugin", tmp_path)

    instance = ConnectorFactory.get_connector("stub_plugin")
    assert isinstance(instance, descriptor.implementation)
    assert isinstance(instance, ConnectorInterface)
    status, result = instance.run("/tmp/cfg.json", {})
    assert status is SimulationStatus.SUCCESS
    assert result == {"answer": 1.0}


def test_factory_raises_for_unknown_plugin_name() -> None:
    register_builtins()
    with pytest.raises(NotImplementedError):
        ConnectorFactory.get_connector("never_registered")
