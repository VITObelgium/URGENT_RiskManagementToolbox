from __future__ import annotations

import logging
import logging.config
import os
from logging import Logger
from typing import Optional

from logger.utils import (
    _build_console_handler,
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
    console_enabled = (
        True if get_log_to_console_value() is None else get_log_to_console_value()
    )

    if console_enabled:
        cfg["root"]["handlers"] = ["console"]
    else:
        cfg["root"]["handlers"] = [
            h for h in cfg["root"].get("handlers", []) if h != "console"
        ]
    cfg.get("handlers", {}).pop("file", None)

    logging.config.dictConfig(cfg)

    if console_enabled:
        root = logging.getLogger()
        # Clear existing handlers to ensure console-only
        root.handlers.clear()
        level_name = cfg.get("handlers", {}).get("console", {}).get("level") or cfg.get(
            "root", {}
        ).get("level", "INFO")
        level = getattr(logging, str(level_name).upper(), logging.INFO)
        root.setLevel(level)
        root.addHandler(_build_console_handler(level))

    else:
        # If console explicitly disabled, leave default root logger alone
        pass


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


def get_logger(name: Optional[str] = "", filename: Optional[str] = None) -> Logger:
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
