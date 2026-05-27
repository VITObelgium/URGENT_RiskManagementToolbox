from __future__ import annotations

from threading import RLock
from typing import Literal, overload

from urgent_plugins.api import (
    PluginDescriptor,
    PluginKind,
    ConnectorPlugin,
    OptimizerPlugin,
    DomainServicePlugin,
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
        self, kind: Literal[PluginKind.DOMAIN_SERVICE], name: str
    ) -> DomainServicePlugin: ...

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


def get_registry() -> _PluginRegistry:
    return _registry
