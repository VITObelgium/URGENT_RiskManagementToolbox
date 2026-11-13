from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from urgent_plugins.api import ConnectorPlugin, DomainServicePlugin, OptimizerPlugin


class PluginTraceError(ValueError):
    """Raised when the configured plugin combination is incompatible."""


def verify_plugin_traces(
    connector: ConnectorPlugin,
    optimizer: OptimizerPlugin,
    domain_services: Sequence[DomainServicePlugin],
) -> None:
    """Fail fast when the configured plugin combination cannot work together.

    Domain services declare the trace tags they provide
    (``DomainServiceInterface.trace()``, defaulting to the service name);
    connectors and optimizers declare the tags they require
    (``required_traces()``, defaulting to none). Every required tag must be
    provided by at least one configured domain service. Picking a sensible
    combination remains the user's job-- this check only catches declared
    mismatches before any expensive work starts.

    Raises:
        PluginTraceError: listing every missing tag per requiring plugin.
    """
    if not domain_services:
        raise PluginTraceError("At least one domain service plugin is required.")

    provided: frozenset[str] = frozenset().union(
        *(plugin.implementation.trace() for plugin in domain_services)
    )

    errors: list[str] = []
    for kind, plugin in (("connector", connector), ("optimizer", optimizer)):
        missing = plugin.implementation.required_traces() - provided
        if missing:
            errors.append(
                f"{kind} plugin {plugin.name!r} requires trace(s) "
                f"{sorted(missing)} not provided by the configured domain "
                f"services (provided: {sorted(provided)})"
            )

    if errors:
        raise PluginTraceError("Incompatible plugin combination:\n" + "\n".join(errors))
