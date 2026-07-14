from __future__ import annotations

import os
from collections.abc import Sequence

from logger import get_logger

from urgent_plugins.api import PluginKind
from urgent_plugins.loader import load_local_plugin
from urgent_plugins.paths import default_plugins_root
from urgent_plugins.registry import get_registry

logger = get_logger(__name__)


def list_plugins() -> None:
    plugins_root = default_plugins_root()
    print(f"Plugin root: {plugins_root}\n")
    for kind in PluginKind:
        kind_dir = plugins_root / kind.directory
        files = (
            sorted(p.stem for p in kind_dir.glob("*.py") if p.stem != "__init__")
            if kind_dir.is_dir()
            else []
        )
        print(f"{kind.value}:")
        if files:
            for name in files:
                print(f"  {name}")
        else:
            print("  (none)")


def _resolve_plugin(kind: PluginKind, name: str) -> str:
    """Ensure the requested plugin is loaded and return its normalised name.

    Plugins are loaded only from the explicit plugin name provided by config:
    ``<plugins_root>/<kind.directory>/<name>.py``. The plugins root is
    exported via an environment variable for workers.
    """
    if not name or not name.strip():
        raise ValueError(f"{kind.value} plugin name must be specified in config.")

    plugins_root = default_plugins_root()
    os.environ["URGENT_PLUGIN_PATH"] = str(plugins_root)

    normalized = name.strip().lower()
    existing = get_registry().get(kind, normalized)
    if existing is not None:
        return existing.name

    descriptor = load_local_plugin(kind, normalized, plugins_root)
    logger.info(
        "Loaded %s plugin %r from %s",
        kind.value,
        descriptor.name,
        plugins_root / kind.directory / f"{descriptor.name}.py",
    )
    return descriptor.name


def _resolve_and_export(kind: PluginKind, name: str) -> str:
    resolved = _resolve_plugin(kind, name)
    os.environ[kind.get_env_var()] = resolved
    return resolved


def resolve_connector_plugin(connector_name: str) -> str:
    return _resolve_and_export(PluginKind.CONNECTOR, connector_name)


def resolve_optimizer_plugin(optimizer_name: str) -> str:
    return _resolve_and_export(PluginKind.OPTIMIZER, optimizer_name)


def resolve_domain_service_plugins(domain_service_names: Sequence[str]) -> list[str]:
    """Resolve every configured domain service plugin and export their names.

    Returns the normalised names in config order; duplicates are rejected by
    the config model before this is reached.
    """
    if not domain_service_names:
        raise ValueError(
            "At least one domain_service plugin name must be specified in config."
        )
    resolved = [
        _resolve_plugin(PluginKind.DOMAIN_SERVICE, name)
        for name in domain_service_names
    ]
    os.environ[PluginKind.DOMAIN_SERVICE.get_env_var()] = ",".join(resolved)
    return resolved
