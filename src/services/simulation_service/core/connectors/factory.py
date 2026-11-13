from __future__ import annotations

from urgent_plugins import (
    ConnectorPlugin,
    PluginKind,
    get_registry,
)
from urgent_plugins.loader import load_local_plugin
from urgent_plugins.paths import default_plugins_root
from .common import ConnectorInterface


class ConnectorFactory:
    @staticmethod
    def get_connector(connector_name: str) -> ConnectorInterface:
        """Resolve a connector implementation by plugin name.

        The factory consults the :mod:`urgent_plugins` registry and, when the
        explicitly named plugin has not been loaded in this process yet, loads
        it from the configured plugin root.
        """

        name = connector_name.strip().lower()
        if not name:
            raise ValueError(
                "connector_name must be specified explicitly in the plugins config."
            )

        descriptor = get_registry().get(PluginKind.CONNECTOR, name)
        if descriptor is None:
            descriptor = load_local_plugin(
                PluginKind.CONNECTOR, name, default_plugins_root()
            )
        if isinstance(descriptor, ConnectorPlugin):
            return descriptor.implementation()
        raise NotImplementedError(
            f"Unknown connector {connector_name!r}; not registered in urgent_plugins."
        )
