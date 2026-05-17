from __future__ import annotations

from threading import RLock
from typing import Literal, overload

from pathlib import Path

from urgent_plugins.api import (
    PluginDescriptor,
    PluginKind,
    ConnectorPlugin,
    OptimizerPlugin,
    WellManagementPlugin,
)


class _PluginRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._by_kind: dict[PluginKind, dict[str, PluginDescriptor]] = {
            kind: {} for kind in PluginKind
        }

    @overload
    def require(
        self, kind: Literal[PluginKind.CONNECTOR], name: str
    ) -> ConnectorPlugin: ...

    @overload
    def require(
        self, kind: Literal[PluginKind.OPTIMIZER], name: str
    ) -> OptimizerPlugin: ...

    @overload
    def require(
        self, kind: Literal[PluginKind.WMS], name: str
    ) -> WellManagementPlugin: ...

    def require(self, kind: PluginKind, name: str) -> PluginDescriptor:
        descriptor = self.get(kind, name)
        if descriptor is None:
            raise KeyError(
                f"No plugin registered for kind={kind.value!r} name={name!r}"
            )
        return descriptor

    def register(self, kind: PluginKind, descriptor: PluginDescriptor) -> None:
        with self._lock:
            self._by_kind[kind][descriptor.name] = descriptor

    def get(self, kind: PluginKind, name: str) -> PluginDescriptor | None:
        with self._lock:
            return self._by_kind[kind].get(name)

    def names(self, kind: PluginKind) -> list[str]:
        with self._lock:
            return sorted(self._by_kind[kind].keys())

    def clear(self) -> None:
        with self._lock:
            for entries in self._by_kind.values():
                entries.clear()


_registry = _PluginRegistry()


_BUILTIN_PLUGIN_STEMS: dict[PluginKind, tuple[str, ...]] = {
    PluginKind.CONNECTOR: ("opendarts",),
    PluginKind.OPTIMIZER: ("pso",),
    PluginKind.WMS: ("builtin",),
}


def get_registry() -> _PluginRegistry:
    return _registry


def _builtins_root() -> Path:
    """Return the bundled built-in plugins root (always the repo's plugins/)."""
    from urgent_plugins.paths import find_repo_root

    return find_repo_root() / "plugins"


def register_builtins() -> None:
    """Register the bundled built-in plugins through the same loader as user plugins.

    Always reads the built-in plugin files from the repo's ``plugins/``
    directory. The ``URGENT_PLUGIN_PATH`` environment variable only controls
    where third-party plugins are resolved from, so overriding it (for tests
    or alternate user-plugin trees) does not move the built-ins.

    Idempotent: the loader updates the registry in-place, so repeated calls
    simply re-register the descriptor exposed by each plugin file.
    """
    from urgent_plugins.loader import load_local_plugin

    plugins_root = _builtins_root()
    for kind, stems in _BUILTIN_PLUGIN_STEMS.items():
        for stem in stems:
            load_local_plugin(kind, stem, plugins_root)


def builtin_plugin_name(kind: PluginKind) -> str:
    """Return the canonical built-in plugin name for ``kind``.

    Loads the built-in plugin file when needed and reads its descriptor name,
    so the canonical identifier is owned by the plugin class itself (via its
    ``ConnectorName`` / ``EngineName`` attribute), not by any string literal
    in this package.
    """
    register_builtins()
    stems = _BUILTIN_PLUGIN_STEMS[kind]
    if not stems:
        raise RuntimeError(f"No built-in plugin registered for kind={kind.value!r}")
    descriptor = _registry.get(kind, stems[0])
    if descriptor is None:
        # Fallback: the plugin file might use a different normalized name; pick
        # the first registered descriptor for this kind.
        names = _registry.names(kind)
        if not names:
            raise RuntimeError(
                f"register_builtins did not register any plugin for kind={kind.value!r}"
            )
        return names[0]
    return descriptor.name
