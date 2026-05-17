from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import cast

from urgent_plugins.api import (
    DESCRIPTOR_TYPES,
    PluginDescriptor,
    PluginKind,
    normalize_plugin_name,
)
from urgent_plugins.registry import get_registry


def resolve_plugin_path(kind: PluginKind, name: str, plugins_root: Path) -> Path:
    normalized = normalize_plugin_name(name)
    return plugins_root / kind.directory / f"{normalized}.py"


def load_local_plugin(
    kind: PluginKind, name: str, plugins_root: Path
) -> PluginDescriptor:
    """Import a one-file plugin from disk and register its descriptor.

    Inserts the module into ``sys.modules`` before executing it so circular
    imports inside the plugin behave normally. Validates the exported
    ``plugin`` object before returning.
    """
    normalized = normalize_plugin_name(name)
    path = resolve_plugin_path(kind, normalized, plugins_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Plugin file not found for kind={kind.value!r} name={normalized!r}: {path}"
        )

    module_name = f"plugins.{kind.directory}.{normalized}"
    existing = sys.modules.get(module_name)
    if existing is not None and getattr(existing, "__file__", None) == str(path):
        module = existing
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to build import spec for plugin at {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    raw_descriptor = getattr(module, "plugin", None)
    if raw_descriptor is None:
        raise AttributeError(
            f"Plugin file {path} does not export a top-level 'plugin' object"
        )

    expected_type = DESCRIPTOR_TYPES[kind]
    if not isinstance(raw_descriptor, expected_type):
        raise TypeError(
            f"Plugin at {path} exports {type(raw_descriptor).__name__}, expected "
            f"{expected_type.__name__}"
        )

    descriptor = cast(PluginDescriptor, raw_descriptor)
    get_registry().register(kind, descriptor)
    return descriptor
