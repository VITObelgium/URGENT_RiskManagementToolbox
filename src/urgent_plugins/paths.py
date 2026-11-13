from __future__ import annotations

import os
from pathlib import Path

from utils import find_repo_root


def default_plugins_root() -> Path:
    """Return the plugins root directory.

    Honours ``URGENT_PLUGIN_PATH`` when set, otherwise defaults to ``<repo>/plugins``.
    """
    env = os.environ.get("URGENT_PLUGIN_PATH")
    if env:
        return Path(env)
    return find_repo_root() / "plugins"
