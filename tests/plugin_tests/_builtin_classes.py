"""Test helpers for resolving the built-in plugin implementation classes.

Now that the canonical OpenDARTS and PSO implementations live in plugin files
(``plugins/connectors/opendarts.py`` and ``plugins/optimizers/pso.py``), tests
must access them through the urgent_plugins registry instead of a direct
``src/services/...`` import. These helpers ensure built-ins are registered and
return the underlying implementation class.
"""

from __future__ import annotations

from services.simulation_service.core.connectors.common import ConnectorInterface
from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)
from urgent_plugins import (
    PluginKind,
    get_registry,
    register_builtins,
)

from urgent_plugins.registry import builtin_plugin_name


def get_builtin_connector_class() -> type[ConnectorInterface]:
    register_builtins()
    name = builtin_plugin_name(PluginKind.CONNECTOR)
    descriptor = get_registry().require(PluginKind.CONNECTOR, name)
    return descriptor.implementation  # type: ignore[return-value]


def get_builtin_optimizer_class() -> type[OptimizationEngineInterface]:
    register_builtins()
    name = builtin_plugin_name(PluginKind.OPTIMIZER)
    descriptor = get_registry().require(PluginKind.OPTIMIZER, name)
    return descriptor.implementation  # type: ignore[return-value]
