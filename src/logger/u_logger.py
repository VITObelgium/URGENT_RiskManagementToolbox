import logging
import logging.config
import os
import sys
from datetime import datetime
from logging import Logger
from typing import Any, Dict, Optional

import tomli


def configure_logger() -> None:
    """
    Configure the logger. The logger configuration is read from logging.conf file and additional configurations can be found in pyproject.toml.
    No need to call this function more than once.
    """
    # Get the log file path
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../log")
    if "pytest" in sys.modules:
        log_file_path = os.path.join(log_dir, "pytest_log.log")
    else:
        # Depending on configuration in pyproject.toml, log file name may contain datetime
        if log_to_datetime_log_file():
            log_file_path = os.path.join(
                log_dir,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_urgent_log.log",
            )
        else:
            log_file_path = os.path.join(log_dir, "urgent_log.log")
    log_conf_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logging.conf"
    )
    # Create the log directory if it does not exist along with the log file
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    open(log_file_path, "a", encoding="utf-8").close()
    # Read and set the logging configuration from logging.conf file
    logging.config.fileConfig(
        log_conf_file_path, defaults={"logfilename": log_file_path}
    )


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
    log_to_console = get_log_to_console_value()
    _effective_name = "aux.console" if log_to_console else "aux" if name else "root"
    return logging.getLogger(_effective_name)


def get_log_to_console_value() -> bool:
    """
    Get the value of log_to_console from pyproject.toml.

    Returns
    -------
    bool
        The value of log_to_console.

    """
    return bool(get_logging_output()["log_to_console"])


def log_to_datetime_log_file() -> bool:
    """
    Get the value of datetime_log_file from pyproject.toml.

    Returns
    -------
    bool
        The value of datetime_log_file.

    """
    return bool(get_logging_output()["datetime_log_file"])


def get_logging_output() -> Dict[str, Any]:
    """
    Get the logging output configuration from pyproject.toml.

    Returns
    -------
    Dict[str, Any]
        The logging output configuration.

    """
    pyproject_toml_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyproject.toml"))
    )
    with open(pyproject_toml_path, "rb") as f:
        pyproject_toml = tomli.load(f)
    return dict(pyproject_toml["logging"]["output"])
