from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from services.simulation_service.core.connectors.factory import ConnectorFactory
from services.solution_updater_service.core.engines.factory import (
    OptimizationEngineFactory,
)
from urgent_plugins import (
    PluginKind,
    get_registry,
    resolve_connector_plugin,
    resolve_optimizer_plugin,
)

from urgent_plugins.loader import resolve_plugin_path

from ._builtin_classes import (
    get_builtin_connector_class,
    get_builtin_optimizer_class,
)


_STUB_CONNECTOR_SRC = """
from __future__ import annotations

import threading

from services.simulation_service.core.connectors.common import (
    ConnectorInterface,
    JsonPath,
    SimulationResults,
    SimulationStatus,
)
from urgent_plugins import ConnectorPlugin


class SmokeConnector(ConnectorInterface):
    ConnectorName = "smoke"

    @staticmethod
    def run(
        config_path: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        return SimulationStatus.SUCCESS, {"smoke": 1.0}


plugin = ConnectorPlugin(
    name=SmokeConnector.ConnectorName, implementation=SmokeConnector
)
"""


_STUB_OPTIMIZER_SRC = """
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)
from urgent_plugins import OptimizerPlugin


class IdentityEngine(OptimizationEngineInterface):
    EngineName = "identity"

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    def update_solution_to_next_iter(
        self, parameters, results, lb, ub, indexed_objectives_strategy,
        A=None, b=None, iteration_ratio=None,
    ):
        return parameters

    @property
    def global_best_result(self):
        return np.array([0.0])

    @property
    def global_best_control_vector(self):
        return np.array([0.0])

    def get_checkpoint_state(self) -> dict[str, Any]:
        return {}

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        return None


plugin = OptimizerPlugin(name=IdentityEngine.EngineName, implementation=IdentityEngine)
"""


@pytest.fixture(autouse=True)
def _isolate_registry_and_env(monkeypatch):
    get_registry().clear()
    for key in (
        "URGENT_PLUGIN_PATH",
        "URGENT_CONNECTOR_PLUGIN_NAME",
        "URGENT_OPTIMIZER_PLUGIN_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    get_registry().clear()


def _write_plugin(plugins_root: Path, kind: PluginKind, name: str, body: str) -> Path:
    path = resolve_plugin_path(kind, name, plugins_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return path


def test_resolve_connector_for_bundled_plugin_returns_class_name(monkeypatch) -> None:
    builtin_cls = get_builtin_connector_class()
    # Use the actual repo plugins dir so the loader finds the built-in file.
    resolved = resolve_connector_plugin(builtin_cls.ConnectorName)  # type: ignore[attr-defined]

    assert resolved == builtin_cls.ConnectorName  # type: ignore[attr-defined]
    assert os.environ["URGENT_CONNECTOR_PLUGIN_NAME"] == builtin_cls.ConnectorName  # type: ignore[attr-defined]
    assert isinstance(ConnectorFactory.get_connector(resolved), builtin_cls)


def test_resolve_optimizer_for_bundled_plugin_returns_class_name(monkeypatch) -> None:
    builtin_cls = get_builtin_optimizer_class()
    resolved = resolve_optimizer_plugin(builtin_cls.EngineName)  # type: ignore[attr-defined]

    assert resolved == builtin_cls.EngineName  # type: ignore[attr-defined]
    assert os.environ["URGENT_OPTIMIZER_PLUGIN_NAME"] == builtin_cls.EngineName  # type: ignore[attr-defined]
    assert isinstance(OptimizationEngineFactory.get_engine(resolved), builtin_cls)


def test_resolve_custom_connector_loads_and_factory_returns_instance(
    tmp_path: Path, monkeypatch
) -> None:
    _write_plugin(tmp_path, PluginKind.CONNECTOR, "smoke", _STUB_CONNECTOR_SRC)
    monkeypatch.setenv("URGENT_PLUGIN_PATH", str(tmp_path))

    resolved = resolve_connector_plugin("smoke")

    assert resolved == "smoke"
    instance = ConnectorFactory.get_connector(resolved)
    status, result = instance.run("/tmp/x.json", {})
    assert status.name == "SUCCESS"
    assert result == {"smoke": 1.0}


def test_resolve_custom_optimizer_loads_and_factory_returns_instance(
    tmp_path: Path, monkeypatch
) -> None:
    _write_plugin(tmp_path, PluginKind.OPTIMIZER, "identity", _STUB_OPTIMIZER_SRC)
    monkeypatch.setenv("URGENT_PLUGIN_PATH", str(tmp_path))

    resolved = resolve_optimizer_plugin("identity")

    assert resolved == "identity"
    instance = OptimizationEngineFactory.get_engine(resolved, seed=7)
    assert instance.__class__.__name__ == "IdentityEngine"


def test_resolve_unknown_connector_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("URGENT_PLUGIN_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        resolve_connector_plugin("does_not_exist")


def test_resolve_unknown_optimizer_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("URGENT_PLUGIN_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        resolve_optimizer_plugin("does_not_exist")
