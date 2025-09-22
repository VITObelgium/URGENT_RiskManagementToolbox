from __future__ import annotations

import json
import logging
import logging.config
from logging import Logger
from pathlib import Path
from typing import Optional

from logger.utils import (
    _build_console_handler,
    configure_default_profile,
    get_log_to_console_value,
    get_logger_profile,
)

_logger_configured = False


def _configure_stdout_only_profile() -> None:
    """Configure logging to stdout only (for server/worker processes or Docker).

    No files, no queue: just a console handler on root.
    """
    config_path = Path(__file__).parent / "logging_config.json"
    console_enabled = (
        True if get_log_to_console_value() is None else get_log_to_console_value()
    )

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)

            if not console_enabled:
                # Remove file handler if console is explicitly disabled
                cfg["root"]["handlers"] = [
                    h for h in cfg["root"].get("handlers", []) if h == "console"
                ]
                cfg.get("handlers", {}).pop("file", None)

            logging.config.dictConfig(cfg)
            return
        except Exception:
            pass

    if console_enabled:
        root = logging.getLogger()
        # Clear existing handlers to ensure console-only
        root.handlers.clear()
        root.setLevel(logging.INFO)
        root.addHandler(_build_console_handler(logging.INFO))
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


def get_logger(name: Optional[str] = "") -> Logger:
    """Return a named logger; config is applied on first use.

    - Names like "threading-worker" and "threading-server" are preserved.
    - For normal use, pass __name__ to get a module-specific logger.
    """
    if not _logger_configured:
        configure_logger()
    return logging.getLogger(name) if name else logging.getLogger()
