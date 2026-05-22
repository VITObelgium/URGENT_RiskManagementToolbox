from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.simulation_service.core.connectors.common import ConnectorInterface
    from services.solution_updater_service.core.engines.common import (
        OptimizationEngineInterface,
    )
    from services.problem_dispatcher_service.core.service.interface import (
        DomainServiceInterface,
    )


class PluginKind(Enum):
    CONNECTOR = "connector"
    OPTIMIZER = "optimizer"
    DOMAIN_SERVICE = "domain_service"

    @property
    def directory(self) -> str:
        return _KIND_DIRECTORIES[self]

    def get_env_var(self) -> str:
        from urgent_plugins import PluginSettings

        match self:
            case PluginKind.CONNECTOR:
                field_name = "connector_plugin_name"
            case PluginKind.OPTIMIZER:
                field_name = "optimizer_plugin_name"
            case PluginKind.DOMAIN_SERVICE:
                field_name = "domain_service_plugin_name"
        return str(PluginSettings.model_fields[field_name].validation_alias)


_KIND_DIRECTORIES: dict[PluginKind, str] = {
    PluginKind.CONNECTOR: "connectors",
    PluginKind.OPTIMIZER: "optimizers",
    PluginKind.DOMAIN_SERVICE: "domain_services",
}


_VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def normalize_plugin_name(name: str) -> str:
    """Normalize and validate a plugin name to a safe identifier.

    Lowercases and replaces hyphens with underscores. Rejects names that do not
    match a strict identifier-like pattern. The normalized form is used both as
    the filename stem and as part of the synthetic module name.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"Plugin name must be a non-empty string, got {name!r}")
    normalized = name.strip().lower().replace("-", "_")
    if not _VALID_NAME_RE.match(normalized):
        raise ValueError(f"Invalid plugin name {name!r}; must match [a-z][a-z0-9_]*")
    return normalized


@dataclass(frozen=True)
class ConnectorPlugin:
    name: str
    implementation: type[ConnectorInterface]

    def __post_init__(self) -> None:
        from services.simulation_service.core.connectors.common import (
            ConnectorInterface as _CI,
        )

        object.__setattr__(self, "name", normalize_plugin_name(self.name))
        if not isinstance(self.implementation, type) or not issubclass(
            self.implementation, _CI
        ):
            raise TypeError(
                "ConnectorPlugin.implementation must be a subclass of "
                f"ConnectorInterface; got {self.implementation!r}"
            )


@dataclass(frozen=True)
class OptimizerPlugin:
    name: str
    implementation: type[OptimizationEngineInterface]

    def __post_init__(self) -> None:
        from services.solution_updater_service.core.engines.common import (
            OptimizationEngineInterface as _OEI,
        )

        object.__setattr__(self, "name", normalize_plugin_name(self.name))
        if not isinstance(self.implementation, type) or not issubclass(
            self.implementation, _OEI
        ):
            raise TypeError(
                "OptimizerPlugin.implementation must be a subclass of "
                f"OptimizationEngineInterface; got {self.implementation!r}"
            )


@dataclass(frozen=True)
class DomainServicePlugin:
    name: str
    implementation: type[DomainServiceInterface]

    def __post_init__(self) -> None:
        from services.problem_dispatcher_service.core.service.interface import (
            DomainServiceInterface as _DSI,
        )

        object.__setattr__(self, "name", normalize_plugin_name(self.name))
        if not isinstance(self.implementation, type) or not issubclass(
            self.implementation, _DSI
        ):
            raise TypeError(
                "DomainServicePlugin.implementation must be a subclass of "
                f"DomainServiceInterface; got {self.implementation!r}"
            )


type PluginDescriptor = ConnectorPlugin | OptimizerPlugin | DomainServicePlugin


DESCRIPTOR_TYPES: dict[PluginKind, type] = {
    PluginKind.CONNECTOR: ConnectorPlugin,
    PluginKind.OPTIMIZER: OptimizerPlugin,
    PluginKind.DOMAIN_SERVICE: DomainServicePlugin,
}
