"""Tests for urgent_plugins.verification.verify_plugin_traces.

Builds minimal fake plugin descriptors from the real interfaces so the
trace-verification contract is exercised exactly as the bootstrap uses it.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter

from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from services.simulation_service.core.connectors.common import (
    ConnectorInterface,
    JsonPath,
    SimulationResults,
    SimulationStatus,
)
from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)
from urgent_plugins import (
    ConnectorPlugin,
    DomainServicePlugin,
    OptimizerPlugin,
    PluginTraceError,
    verify_plugin_traces,
)


class _StubConnector(ConnectorInterface):
    ConnectorName = "stub_connector"

    @staticmethod
    def run(
        config_path: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        return SimulationStatus.SUCCESS, {}


class _DemandingConnector(_StubConnector):
    ConnectorName = "demanding_connector"

    @classmethod
    def required_traces(cls) -> frozenset[str]:
        return frozenset({"stub_service"})


class _GreedyConnector(_StubConnector):
    ConnectorName = "greedy_connector"

    @classmethod
    def required_traces(cls) -> frozenset[str]:
        return frozenset({"thermal_output", "stub_service"})


class _StubEngine(OptimizationEngineInterface):
    EngineName = "stub_engine"

    def update_solution_to_next_iter(
        self,
        parameters,
        results,
        lb,
        ub,
        indexed_objectives_strategy,
        A=None,
        b=None,
        iteration_ratio=None,
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


class _DemandingEngine(_StubEngine):
    EngineName = "demanding_engine"

    @classmethod
    def required_traces(cls) -> frozenset[str]:
        return frozenset({"gradient_support"})


class _StubDomainService(DomainServiceInterface):
    ServiceName = "stub_service"

    @classmethod
    def get_item_state_adapter(cls) -> TypeAdapter[Any]:
        return TypeAdapter(dict)

    def build_payload(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {}


def _connector_plugin(implementation: type[ConnectorInterface]) -> ConnectorPlugin:
    return ConnectorPlugin(
        name=implementation.ConnectorName,  # type: ignore[attr-defined]
        implementation=implementation,
    )


def _optimizer_plugin(
    implementation: type[OptimizationEngineInterface],
) -> OptimizerPlugin:
    return OptimizerPlugin(
        name=implementation.EngineName,  # type: ignore[attr-defined]
        implementation=implementation,
    )


_DOMAIN_PLUGIN = DomainServicePlugin(
    name=_StubDomainService.ServiceName, implementation=_StubDomainService
)


def test_verify_plugin_traces_happy_path_no_requirements() -> None:
    verify_plugin_traces(
        connector=_connector_plugin(_StubConnector),
        optimizer=_optimizer_plugin(_StubEngine),
        domain_services=[_DOMAIN_PLUGIN],
    )


def test_verify_plugin_traces_happy_path_required_trace_provided() -> None:
    verify_plugin_traces(
        connector=_connector_plugin(_DemandingConnector),
        optimizer=_optimizer_plugin(_StubEngine),
        domain_services=[_DOMAIN_PLUGIN],
    )


def test_verify_plugin_traces_missing_connector_trace_lists_plugin_and_tags() -> None:
    with pytest.raises(PluginTraceError) as excinfo:
        verify_plugin_traces(
            connector=_connector_plugin(_GreedyConnector),
            optimizer=_optimizer_plugin(_StubEngine),
            domain_services=[_DOMAIN_PLUGIN],
        )

    message = str(excinfo.value)
    assert "connector" in message
    assert "greedy_connector" in message
    assert "thermal_output" in message
    # The satisfied tag must not be reported as missing.
    assert "['thermal_output']" in message
    # Provided traces are listed for diagnostics.
    assert "stub_service" in message


def test_verify_plugin_traces_missing_optimizer_trace_raises() -> None:
    with pytest.raises(PluginTraceError) as excinfo:
        verify_plugin_traces(
            connector=_connector_plugin(_StubConnector),
            optimizer=_optimizer_plugin(_DemandingEngine),
            domain_services=[_DOMAIN_PLUGIN],
        )

    message = str(excinfo.value)
    assert "optimizer" in message
    assert "demanding_engine" in message
    assert "gradient_support" in message


def test_verify_plugin_traces_empty_domain_services_raises() -> None:
    with pytest.raises(PluginTraceError, match="At least one domain service"):
        verify_plugin_traces(
            connector=_connector_plugin(_StubConnector),
            optimizer=_optimizer_plugin(_StubEngine),
            domain_services=[],
        )


def test_verify_plugin_traces_union_across_domain_services() -> None:
    class _SecondDomainService(_StubDomainService):
        ServiceName = "second_service"

        @classmethod
        def trace(cls) -> frozenset[str]:
            return frozenset({cls.ServiceName, "thermal_output"})

    second_plugin = DomainServicePlugin(
        name=_SecondDomainService.ServiceName, implementation=_SecondDomainService
    )

    # greedy connector needs both 'stub_service' and 'thermal_output' —
    # satisfied only by the union of the two services' traces.
    verify_plugin_traces(
        connector=_connector_plugin(_GreedyConnector),
        optimizer=_optimizer_plugin(_StubEngine),
        domain_services=[_DOMAIN_PLUGIN, second_plugin],
    )
