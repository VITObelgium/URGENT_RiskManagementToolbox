import logging
import logging.config
import os
import sys
from datetime import datetime
from logging import Logger
from typing import Optional

from logger.utils import (
    get_log_to_console_value,
    get_logger_profile,
    log_to_datetime_log_file,
)

_logger_configured = False


def configure_logger() -> None:
    global _logger_configured
    if _logger_configured:
        return

    profile = get_logger_profile()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_conf_file_path = ""
    config_defaults = {}

    if profile == "server":
        log_conf_file_path = os.path.join(
            base_dir, "orchestration_loggers", "server_logging.conf"
        )
    elif profile == "worker":
        log_conf_file_path = os.path.join(
            base_dir, "orchestration_loggers", "worker_logging.conf"
        )
    else:  # Default profile
        log_conf_file_path = os.path.join(base_dir, "logging.conf")
        log_dir = os.path.join(base_dir, "../../log")
        log_file_name_part = (
            "pytest_log.log" if "pytest" in sys.modules else "urgent_log.log"
        )

        if "pytest" not in sys.modules and log_to_datetime_log_file():
            log_file_name_part = (
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_urgent_log.log"
            )

        log_file_path = os.path.join(log_dir, log_file_name_part)

        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, "a", encoding="utf-8"):  # Touch file
                pass
            config_defaults["logfilename"] = log_file_path
        except OSError as e:
            sys.stderr.write(
                f"Warning: Could not create/access log file {log_file_path}: {e}\n"
            )

    if not os.path.exists(log_conf_file_path):
        sys.stderr.write(
            f"Error: Logging configuration file not found: {log_conf_file_path}. Falling back to basic logging.\n"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        _logger_configured = True
        return

    try:
        logging.config.fileConfig(
            log_conf_file_path, defaults=config_defaults, disable_existing_loggers=False
        )
    except Exception as e:
        sys.stderr.write(
            f"Error configuring logging from {log_conf_file_path}: {e}. Falling back to basic logging.\n"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    _logger_configured = True


def get_logger(name: Optional[str] = "") -> Logger:
    """
    Get the logger.

    Parameters
    ----------
    name : str, optional
        The name of the logger. No name results in root logger. For normal use, use __name__ as the name.

    Returns
    -------
    Logger
        The logger object.

    """
    if not _logger_configured:
        configure_logger()
    profile = get_logger_profile()
    if profile == "server" or (name and name.startswith("server")):
        return logging.getLogger("server")
    elif profile == "worker" or (name and name.startswith("worker")):
        return logging.getLogger("worker")
    else:
        log_to_console = get_log_to_console_value()
        _effective_name = "aux.console" if log_to_console else "aux" if name else "root"
        return logging.getLogger(_effective_name)
