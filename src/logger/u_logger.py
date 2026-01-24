from __future__ import annotations

import logging
import logging.config
import os
from logging import Logger

from logger.utils import (
    configure_default_profile,
    get_log_config,
    get_log_to_console_value,
    get_logger_profile,
)

_logger_configured = False


def _configure_stdout_only_profile() -> None:
    """Configure logging to stdout only (for server/worker processes or Docker).

    No files, no queue: just a console handler on root.
    """
    cfg = get_log_config()
    console_enabled = bool(get_log_to_console_value())

    if console_enabled:
        cfg["root"]["handlers"] = ["console"]
    else:
        cfg["root"]["handlers"] = []
    cfg.get("handlers", {}).pop("file", None)

    logging.config.dictConfig(cfg)


def configure_logger() -> None:
    global _logger_configured
    if _logger_configured:
        return

    profile = get_logger_profile()
    if profile == "default":
        configure_default_profile()
    elif profile in ("server", "worker"):
        _configure_stdout_only_profile()
    else:
        # Unknown profile – fall back to default orchestrator behavior
        configure_default_profile()

    _logger_configured = True


def get_logger(name: str | None = None, filename: str | None = None) -> Logger:
    """Return a named logger; config is applied on first use.

    - Names like "threading-worker" and "threading-server" are preserved.
    - For normal use, pass __name__ to get a module-specific logger.
    """
    if not _logger_configured:
        configure_logger()
    if os.getenv("OPEN_DARTS_RUNNER", "thread").lower() == "docker" and filename:
        if name in ("threading-server", "threading-worker"):
            name = filename
    return logging.getLogger(name) if name else logging.getLogger()
