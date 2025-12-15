from __future__ import annotations

import atexit
import copy
import logging
import os
import subprocess
import sys
from datetime import datetime
from logging import Logger
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from pathlib import Path
from typing import Any

_queue: Queue | None = None
_listener: QueueListener | None = None
_shutdown_registered: bool = False


def find_pyproject_toml(start: Path | None = None) -> Path | None:
    """
    Locate pyproject.toml by checking:
    1) Current working directory
    2) Walking up from the provided start path (or this file) to filesystem root

    Returns
    -------
    Path | None
        Returns the resolved Path if found, else None.
    """
    try:
        cwd_candidate = Path.cwd() / "pyproject.toml"
        if cwd_candidate.exists():
            return cwd_candidate.resolve()
    except Exception:
        pass

    base = start or Path(__file__).resolve()
    for candidate in [base] + list(base.parents):
        pp = candidate / "pyproject.toml"
        if pp.exists():
            return pp.resolve()

    # Not found
    return None


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
    try:
        return bool(get_logging_output().get("log_to_console", False))
    except FileNotFoundError:
        return False


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


def get_logging_output() -> dict[str, Any]:
    """
    Get the logging output configuration from pyproject.toml.

    Returns
    -------
    dict[str, Any]
        The logging output configuration.

    """
    pyproject_path = find_pyproject_toml()
    if pyproject_path is None or not pyproject_path.exists():
        # Returning safe defaults if pyproject.toml is not found.
        return {
            "log_to_console": False,
            "datetime_log_file": False,
            "external_docker_log_console": False,
        }

    import tomli

    with open(pyproject_path, "rb") as f:
        pyproject_toml = tomli.load(f)
    return dict(pyproject_toml.get("logging", {}).get("output", {}))


def get_external_console_logging() -> bool:
    """
    Get the value of external_docker_log_console from pyproject.toml.

    Returns
    -------
    bool
        The value of external_docker_log_console. Returns False if pyproject.toml is not found.
    """
    try:
        return bool(get_logging_output().get("external_docker_log_console", False))
    except FileNotFoundError:
        return False


def get_services(logger: Logger) -> list[str]:
    """Return a list of all docker compose service names, including numbered workers."""
    try:
        compose_services = subprocess.check_output(
            ["docker", "ps", "--format", "'{{.Names}}'"], text=True
        ).splitlines()
        return [service.strip("'") for service in compose_services if service.strip()]
    except Exception as e:
        raise RuntimeError(
            "Failed to get Docker services. Ensure Docker is running and you have access to it."
        ) from e


def get_services_id() -> list[str]:
    try:
        container_ids = subprocess.check_output(
            ["docker", "ps", "-q"], text=True
        ).splitlines()

        services = set()
        for cid in container_ids:
            inspect_output = subprocess.check_output(
                [
                    "docker",
                    "inspect",
                    "--format",
                    '{{ index .Config.Labels "com.docker.compose.service" }}',
                    cid,
                ],
                text=True,
            ).strip()
            if inspect_output and inspect_output != "<no value>":
                services.add(inspect_output)
        return list(services)
    except Exception as e:
        raise RuntimeError(
            "Failed to get Docker services. Ensure Docker is running and you have access to it."
        ) from e


def _build_console_handler(
    level: int = logging.INFO, format: str = "simple"
) -> logging.Handler:
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    cfg = get_log_config()
    fmt = cfg.get("formatters", {}).get(format, {})
    console.setFormatter(
        logging.Formatter(
            fmt=fmt["format"],
            datefmt=fmt["datefmt"],
        )
    )
    return console


def configure_default_profile() -> None:
    """Configure logging for the orchestrator (default profile).

    - All loggers propagate to root.
    - Root has a QueueHandler; a QueueListener owns the file/console handlers.
    - No module writes to files directly.
    """
    logging_config = get_log_config()
    cfg = copy.deepcopy(logging_config)
    cfg.setdefault("root", {}).update({"handlers": []})
    cfg.get("handlers", {}).pop("file", None)
    cfg.get("handlers", {}).pop("console", None)
    logging.config.dictConfig(cfg)
    _start_queue_listener()


def _start_queue_listener() -> None:
    global _listener, _queue
    if _listener is not None:
        return

    _queue = Queue(-1)
    file_path = _ensure_logfile_path()
    logging_config = get_log_config()

    handlers: list[logging.Handler] = []
    if get_log_to_console_value() and "pytest" not in sys.modules:
        handlers.append(_build_console_handler(logging.INFO))
    if file_path is not None:

        class _SelectiveThreadFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                tn = getattr(record, "threadName", "")
                # For worker/server threads, only forward ERROR+ to the file
                if tn == "server" or tn.startswith("server:"):
                    return record.levelno >= logging.ERROR
                if tn.startswith("worker-"):
                    return record.levelno >= logging.ERROR
                return True

        fh = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        fh.setLevel(
            logging_config.get("handlers", {})
            .get("file", {})
            .get("level", logging.INFO)
        )
        ffmt_name = logging_config.get("handlers", {}).get("file", {}).get("formatter")
        formatter_cfg = logging_config.get("formatters", {}).get(ffmt_name, {})
        ffmt = formatter_cfg.get("format")
        fdatefmt = formatter_cfg.get("datefmt")
        fh.setFormatter(
            logging.Formatter(
                fmt=ffmt
                or "%(asctime)s.%(msecs)03d %(module)s:%(lineno)d %(levelname)s - %(message)s",
                datefmt=fdatefmt or "%Y-%m-%d %H:%M:%S",
            )
        )
        fh.addFilter(_SelectiveThreadFilter())
        handlers.append(fh)

    _listener = QueueListener(_queue, *handlers, respect_handler_level=True)
    _listener.start()

    # For clean shutdown. Using atexit exit handler module to stop listener.
    global _shutdown_registered
    if not _shutdown_registered:
        try:
            atexit.register(_stop_listener)
        finally:
            _shutdown_registered = True

    # Configure root to enqueue records only
    qh = QueueHandler(_queue)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(qh)
    root_level_name = logging_config.get("root", {}).get("level", "INFO")
    root.setLevel(root_level_name)


def _ensure_logfile_path() -> str | None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "../../log")
    file_name = "pytest_log.log" if "pytest" in sys.modules else "urgent_log.log"
    if "pytest" not in sys.modules and log_to_datetime_log_file():
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_urgent_log.log"
    try:
        os.makedirs(log_dir, exist_ok=True)
        full_path = os.path.join(log_dir, file_name)
        with open(full_path, "a", encoding="utf-8"):
            pass
        return full_path
    except OSError as e:
        sys.stderr.write(
            f"Warning: Could not create/access log directory/file in {log_dir}: {e}\n"
        )
        return None


def _stop_listener() -> None:
    global _listener, _queue
    try:
        if _listener is not None:
            _listener.stop()
    except Exception:
        pass
    finally:
        _listener = None
        _queue = None


def get_log_config() -> dict[str, Any]:
    """
    Load the logging.config dictionary from pyproject.toml.

    Returns
    -------
    dict[str, Any]
        The logging configuration compatible with logging.config.dictConfig.
    """
    pyproject_path = find_pyproject_toml()
    import tomli

    if pyproject_path and pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                pyproject_toml = tomli.load(f)
            config = pyproject_toml.get("logging", {}).get("config")
            if isinstance(config, dict):
                return config
        except Exception as e:
            sys.stderr.write(
                f"Warning: Failed to parse [logging.config] from {pyproject_path}: {e}\n"
            )

    # Fallback
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s.%(msecs)03d %(module)s:%(lineno)d %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(asctime)s.%(msecs)03d - %(module)-30s %(lineno)-4d - %(levelname)-8s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {},
        "root": {"level": "INFO", "handlers": ["console"]},
    }


def get_log_config_path() -> Path:
    """
    Return the path to pyproject.toml (source of logging configuration).

    Returns
    -------
    Path
        The path to pyproject.toml.
    """
    p = find_pyproject_toml()
    if p is not None:
        return p
    return Path(__file__).resolve().parents[2] / "pyproject.toml"
