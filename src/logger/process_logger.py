import logging
import sys
from functools import lru_cache
from logging.handlers import MemoryHandler
from pathlib import Path

from logger.orchestration_logger import _start_external_log_terminal
from logger.utils import get_external_console_logging, get_file_log_mode, get_log_config

TAIL_LINES = 0


class _ThreadNameFilter(logging.Filter):
    """Filter records by exact thread name."""

    def __init__(self, thread_name: str) -> None:
        super().__init__()
        self._thread_name = str(thread_name)

    def filter(self, record: logging.LogRecord) -> bool:
        name = getattr(record, "threadName", "")
        # Allow exact match or prefixed sub-threads, e.g., "worker-1:stdout"
        return name == self._thread_name or name.startswith(self._thread_name + ":")


@lru_cache(maxsize=1)
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _ensure_log_dir() -> Path:
    log_dir = _project_root() / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _is_pytest_env() -> bool:
    """Detect if we're running under pytest."""
    return "pytest" in sys.modules


def _pytest_log_path() -> Path:
    """Return the expected pytest aggregate log file path."""
    return _ensure_log_dir() / "pytest_log.log"


def _formatter() -> logging.Formatter:
    """Build a formatter using the 'detailed' formatter from logging config."""
    cfg = get_log_config()
    fmt_cfg = cfg.get("formatters", {}).get("detailed", {})
    fmt = fmt_cfg.get(
        "format",
        "%(asctime)s.%(msecs)03d %(module)s:%(lineno)d %(levelname)s - %(message)s",
    )
    datefmt = fmt_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    return logging.Formatter(fmt=fmt, datefmt=datefmt)


# Buffer size for MemoryHandler - flushes after this many records
_MEMORY_HANDLER_CAPACITY = 100


def _add_unique_file_handler(
    target_logger: logging.Logger,
    file_path: Path,
    level: int = logging.INFO,
    record_filter: logging.Filter | None = None,
    use_buffering: bool = True,
) -> None:
    """Attach a FileHandler to logger if an identical one isn't present.

    If use_buffering is True, wraps the FileHandler in a MemoryHandler that
    buffers records and flushes them in batches, reducing I/O overhead.
    """
    file_path = file_path.resolve()
    for h in target_logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                existing = Path(getattr(h, "baseFilename", "")).resolve()
            except Exception:
                existing = None
            if existing == file_path:
                return
        elif isinstance(h, MemoryHandler) and hasattr(h, "target"):
            if isinstance(h.target, logging.FileHandler):
                try:
                    existing = Path(getattr(h.target, "baseFilename", "")).resolve()
                except Exception:
                    existing = None
                if existing == file_path:
                    return

    logger_mode = get_file_log_mode()
    file_handler = logging.FileHandler(file_path, mode=logger_mode, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(_formatter())
    if record_filter is not None:
        file_handler.addFilter(record_filter)

    if use_buffering:
        memory_handler = MemoryHandler(
            capacity=_MEMORY_HANDLER_CAPACITY,
            flushLevel=logging.ERROR,
            target=file_handler,
            flushOnClose=True,
        )
        memory_handler.setLevel(level)
        if record_filter is not None:
            memory_handler.addFilter(record_filter)
        target_logger.addHandler(memory_handler)
    else:
        target_logger.addHandler(file_handler)

    try:
        target_logger.propagate = True
    except Exception:
        pass


def configure_worker_logger(worker_id: int) -> Path:
    """Create a file handler for a specific worker thread.

    The handler is attached to the root logger with a filter matching the worker thread
    name ("worker-{id}"). This captures all log records emitted within the
    worker thread, regardless of the logger name, and writes them to `log/worker_{id}.log`.

    """
    if _is_pytest_env():
        return _pytest_log_path()

    log_dir = _ensure_log_dir()
    file_path = log_dir / f"simulation_worker_{worker_id}.log"

    thread_filter = _ThreadNameFilter(f"worker-{worker_id}")
    tw_logger = logging.getLogger("threading-worker")
    _add_unique_file_handler(tw_logger, file_path, record_filter=thread_filter)
    try:
        tw_logger.propagate = False
    except Exception:
        pass

    if get_external_console_logging() and not _is_pytest_env():
        try:
            _start_external_log_terminal(
                title=f"Simulation Worker {worker_id} Logs",
                command=f"tail -n {TAIL_LINES} -f '{file_path}'",
            )
        except Exception:
            pass

    return file_path


def configure_server_logger() -> Path:
    """Create a file handler for the in-process gRPC server thread.

    The handler is attached to the root logger with a filter matching the server
    thread name ("server"). This captures all log records emitted within the
    server thread, regardless of the logger name, and writes them to `log/server.log`.

    Returns the path to the server log file.
    """

    # During pytest, aggregate into pytest_log.log
    if _is_pytest_env():
        return _pytest_log_path()

    log_dir = _ensure_log_dir()
    file_path = log_dir / "simulation_server.log"

    thread_filter = _ThreadNameFilter("server")
    ts_logger = logging.getLogger("threading-server")
    _add_unique_file_handler(ts_logger, file_path, record_filter=thread_filter)
    try:
        ts_logger.propagate = False
    except Exception:
        pass

    if get_external_console_logging() and not _is_pytest_env():
        try:
            _start_external_log_terminal(
                title="Simulation Server Logs",
                command=f"tail -n {TAIL_LINES} -f '{file_path}'",
            )
        except Exception:
            pass

    return file_path
