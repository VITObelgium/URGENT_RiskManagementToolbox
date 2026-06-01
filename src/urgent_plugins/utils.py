from __future__ import annotations

import os

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
    ``<plugins_root>/<kind.directory>/<name>.py``. The resolved plugin name and
    plugins root are exported via environment variables for workers.
    """
    if not name or not name.strip():
        raise ValueError(f"{kind.value} plugin name must be specified in config.")

    plugins_root = default_plugins_root()
    env_var = kind.get_env_var()
    os.environ["URGENT_PLUGIN_PATH"] = str(plugins_root)

    normalized = name.strip().lower()
    existing = get_registry().get(kind, normalized)
    if existing is not None:
        os.environ[env_var] = existing.name
        return existing.name

    descriptor = load_local_plugin(kind, normalized, plugins_root)
    os.environ[env_var] = descriptor.name
    logger.info(
        "Loaded %s plugin %r from %s",
        kind.value,
        descriptor.name,
        plugins_root / kind.directory / f"{descriptor.name}.py",
    )
    return descriptor.name


def resolve_connector_plugin(connector_name: str) -> str:
    return _resolve_plugin(PluginKind.CONNECTOR, connector_name)


def resolve_optimizer_plugin(optimizer_name: str) -> str:
    return _resolve_plugin(PluginKind.OPTIMIZER, optimizer_name)


def resolve_domain_service_plugin(domain_service_name: str) -> str:
    return _resolve_plugin(PluginKind.DOMAIN_SERVICE, domain_service_name)
