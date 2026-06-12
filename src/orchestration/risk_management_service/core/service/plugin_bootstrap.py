from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from logger import get_logger
from services.problem_dispatcher_service.core.models import (
    PluginConfig,
    ProblemDispatcherDefinition,
)
from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from urgent_plugins import (
    PluginKind,
    get_registry,
    resolve_connector_plugin,
    resolve_domain_service_plugins,
    resolve_optimizer_plugin,
    verify_plugin_traces,
)

logger = get_logger(__name__)


def validate_problem_definition(
    raw_config: dict[str, Any],
) -> ProblemDispatcherDefinition:
    """Two-phase validation of a raw run configuration.

    - Phase 1 reads only the ``plugins`` section and loads the plugins it names,
    so the domain service implementations (and their pydantic item models)
    are importable.
    - Phase 2 validates the complete definition and runs the
    plugin-dependent checks
    """
    plugins_raw = raw_config.get("plugins")
    if plugins_raw is None:
        raise ValueError(
            "Config must declare a 'plugins' section selecting connector, "
            "optimizer, and domain_services."
        )
    plugin_config = PluginConfig.model_validate(plugins_raw)

    resolve_connector_plugin(plugin_config.connector)
    resolve_optimizer_plugin(plugin_config.optimizer)
    resolve_domain_service_plugins(plugin_config.domain_services)

    problem_definition = ProblemDispatcherDefinition.model_validate(raw_config)
    bootstrap_run_plugins(problem_definition)
    return problem_definition


@dataclass(frozen=True)
class ResolvedRunPlugins:
    """Plugins resolved for one run, keyed for direct orchestrator use."""

    connector_name: str
    optimizer_name: str
    domain_services: dict[str, DomainServiceInterface]


def bootstrap_run_plugins(
    problem_definition: ProblemDispatcherDefinition,
) -> ResolvedRunPlugins:
    """Resolve, verify, and prepare every plugin configured for a run.

    Order matters and mirrors config validation: load the plugins named in
    ``plugins``, fail fast on trace mismatches, then validate and normalise
    each domain service item's ``initial_state`` against the owning plugin's
    pydantic adapter. Idempotent — safe to call on resumed runs.
    """
    registry = get_registry()
    connector_name = resolve_connector_plugin(problem_definition.plugins.connector)
    optimizer_name = resolve_optimizer_plugin(problem_definition.plugins.optimizer)
    domain_service_names = resolve_domain_service_plugins(
        problem_definition.plugins.domain_services
    )

    domain_plugins = [
        registry.require(PluginKind.DOMAIN_SERVICE, name)
        for name in domain_service_names
    ]
    verify_plugin_traces(
        connector=registry.require(PluginKind.CONNECTOR, connector_name),
        optimizer=registry.require(PluginKind.OPTIMIZER, optimizer_name),
        domain_services=domain_plugins,
    )

    domain_services: dict[str, DomainServiceInterface] = {}
    for plugin in domain_plugins:
        _validate_item_states(problem_definition, plugin.name)
        domain_services[plugin.name] = plugin.implementation()

    logger.info(
        "Running with %s Connector, %s Optimizer, and DomainService(s): %s",
        connector_name,
        optimizer_name,
        ", ".join(domain_services),
    )
    return ResolvedRunPlugins(
        connector_name=connector_name,
        optimizer_name=optimizer_name,
        domain_services=domain_services,
    )


def _validate_item_states(
    problem_definition: ProblemDispatcherDefinition,
    plugin_name: str,
) -> None:
    """Run each item's initial_state dict through its plugin's adapter.

    Normalises the dict in-place (validated → dump_python) so downstream code
    sees a canonical representation. Raises ValueError with item and plugin
    context on the first failure.
    """
    plugin = get_registry().require(PluginKind.DOMAIN_SERVICE, plugin_name)
    adapter = plugin.implementation.get_item_state_adapter()
    for item in problem_definition.domain_services[plugin_name]:
        try:
            validated = adapter.validate_python(item.initial_state)
            item.initial_state = adapter.dump_python(validated)
        except ValidationError as exc:
            raise ValueError(
                f"Item '{item.name}': invalid initial_state for domain service "
                f"plugin '{plugin_name}':\n{exc}"
            ) from exc
