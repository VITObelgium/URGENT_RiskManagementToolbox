import logging
import sys
from pathlib import Path

from logger.orchestration_logger import _start_external_log_terminal
from logger.utils import get_external_console_logging

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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    return logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(module)s:%(lineno)d %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _add_unique_file_handler(
    target_logger: logging.Logger,
    file_path: Path,
    level: int = logging.DEBUG,
    record_filter: logging.Filter | None = None,
) -> None:
    """Attach a FileHandler to logger if an identical one isn't present."""
    file_path = file_path.resolve()
    for h in target_logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                existing = Path(getattr(h, "baseFilename", "")).resolve()
            except Exception:
                existing = None
            if existing == file_path:
                return

    handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_formatter())
    if record_filter is not None:
        handler.addFilter(record_filter)
    target_logger.addHandler(handler)
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
    file_path = log_dir / f"worker_{worker_id}.log"

    thread_filter = _ThreadNameFilter(f"worker-{worker_id}")
    aux_logger = logging.getLogger("aux")
    aux_logger.setLevel(logging.DEBUG)
    _add_unique_file_handler(aux_logger, file_path, record_filter=thread_filter)
    tw_logger = logging.getLogger("threading-worker")
    tw_logger.setLevel(logging.DEBUG)
    _add_unique_file_handler(tw_logger, file_path, record_filter=thread_filter)

    if get_external_console_logging() and not _is_pytest_env():
        try:
            _start_external_log_terminal(
                title=f"Worker {worker_id} Logs",
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
    file_path = log_dir / "server.log"

    thread_filter = _ThreadNameFilter("server")
    ts_logger = logging.getLogger("threading-server")
    ts_logger.setLevel(logging.DEBUG)
    _add_unique_file_handler(ts_logger, file_path, record_filter=thread_filter)

    if get_external_console_logging() and not _is_pytest_env():
        try:
            _start_external_log_terminal(
                title="Server Logs",
                command=f"tail -n {TAIL_LINES} -f '{file_path}'",
            )
        except Exception:
            pass

    return file_path
