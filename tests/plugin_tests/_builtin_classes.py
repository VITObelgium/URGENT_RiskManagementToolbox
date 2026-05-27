"""Test helpers for resolving bundled plugin implementation classes.

The OpenDARTS, PSO, and domain service implementations live in ordinary plugin
files under ``plugins/``. These helpers load those files explicitly and return
the underlying implementation class.
"""

from __future__ import annotations

from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from services.simulation_service.core.connectors.common import ConnectorInterface
from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)
from urgent_plugins import (
    PluginKind,
)
from urgent_plugins.loader import load_local_plugin
from urgent_plugins.paths import default_plugins_root


def get_builtin_connector_class() -> type[ConnectorInterface]:
    descriptor = load_local_plugin(
        PluginKind.CONNECTOR, "opendarts", default_plugins_root()
    )
    return descriptor.implementation  # type: ignore[return-value]


def get_builtin_optimizer_class() -> type[OptimizationEngineInterface]:
    descriptor = load_local_plugin(PluginKind.OPTIMIZER, "pso", default_plugins_root())
    return descriptor.implementation  # type: ignore[return-value]


def get_builtin_domain_service_class() -> type[DomainServiceInterface]:
    descriptor = load_local_plugin(
        PluginKind.DOMAIN_SERVICE, "builtin", default_plugins_root()
    )
    return descriptor.implementation  # type: ignore[return-value]
