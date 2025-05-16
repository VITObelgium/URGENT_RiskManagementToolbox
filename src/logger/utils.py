import os
from typing import Any, Dict


def get_logger_profile() -> str:
    """
    Get the logger profile from the environment variable.
    If the environment variable is not set, return "default".

    Returns
    -------
    str
        The logger profile.
    """
    return os.environ.get("URGENT_LOGGER_PROFILE", "default")


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
    try:
        datetime_log_file = bool(get_logging_output()["datetime_log_file"])
    except FileNotFoundError:
        # If pyproject.toml is not found, default to False
        datetime_log_file = False
    return datetime_log_file


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

    if not os.path.exists(pyproject_toml_path):
        pyproject_toml_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "pyproject.toml"))
        )

    if not os.path.exists(pyproject_toml_path):
        raise FileNotFoundError(
            "pyproject.toml not found in either the default location or current directory"
        )

    import tomli

    with open(pyproject_toml_path, "rb") as f:
        pyproject_toml = tomli.load(f)
    return dict(pyproject_toml["logging"]["output"])


def get_external_console_logging() -> bool:
    """
    Get the value of external_docker_log_console from pyproject.toml.

    Returns
    -------
    bool
        The value of external_docker_log_console.
    """
    return bool(get_logging_output()["external_docker_log_console"])
