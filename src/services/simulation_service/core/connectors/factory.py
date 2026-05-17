from __future__ import annotations

from urgent_plugins import (
    ConnectorPlugin,
    PluginKind,
    get_registry,
    register_builtins,
)
from .common import ConnectorInterface


class ConnectorFactory:
    @staticmethod
    def get_connector(connector_name: str) -> ConnectorInterface:
        """Resolve a connector implementation by plugin name.

        The factory consults the :mod:`urgent_plugins` registry, which is
        populated by the orchestrator before workers start (built-in plugins
        are seeded via :func:`urgent_plugins.register_builtins`).
        """

        name = connector_name.strip().lower()
        if not name:
            raise ValueError(
                "connector_name must be specified explicitly in the plugins config."
            )

        register_builtins()
        descriptor = get_registry().get(PluginKind.CONNECTOR, name)
        if isinstance(descriptor, ConnectorPlugin):
            return descriptor.implementation()
        raise NotImplementedError(
            f"Unknown connector {connector_name!r}; not registered in urgent_plugins."
        )
