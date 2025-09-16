import asyncio
import logging
import shutil
import signal
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from logger import get_logger
from services.simulation_service.core.infrastructure.server.src._simulation_server_grpc import (
    request_server_shutdown,
)
from services.simulation_service.core.infrastructure.worker.src._simulation_worker_grpc import (
    main as worker_main,
)

logger = get_logger(__name__)


class ServerStartupError(Exception): ...


class ProcessClusterManager:
    """Process-based cluster manager that launches worker scripts.

    Each worker is started as a separate Python process that runs the
    `_simulation_worker_grpc.py` script.
    """

    def __init__(self) -> None:
        # When running server in-process we keep a reference to a daemon thread
        self._server_thread: threading.Thread | None = None
        self._worker_count = 0

        # In-process worker management
        self._worker_threads: list[threading.Thread] = []
        self._worker_stops: list[threading.Event] = []
        self._stopping = threading.Event()

        # TODO: Change this static host/port
        self.host = "127.0.0.1"
        self.port = 50051

    def _wait_for_server_readiness(self, timeout=25.0, interval=0.25):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._server_thread is not None and not self._server_thread.is_alive():
                raise ServerStartupError(
                    "Server thread terminated unexpectedly during startup"
                )

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(interval)
                try:
                    sock.connect((self.host, self.port))
                    logger.info(
                        "Server is ready and accepting connections on %s:%d",
                        self.host,
                        self.port,
                    )
                    return
                except OSError:
                    pass

            time.sleep(interval)
        raise TimeoutError("Server did not become ready within timeout")

    def _spawn_server(self):
        from services.simulation_service.core.infrastructure.server.src._simulation_server_grpc import (
            driver,
        )

        self._server_thread = threading.Thread(
            target=driver, daemon=True, name="server"
        )
        self._server_thread.start()

    def _start_output_threads(self, proc, name):
        def reader(stream, level):
            for line in iter(stream.readline, ""):
                logger.log(level, "[%s pid=%s] %s", name, proc.pid, line.rstrip())

        threading.Thread(
            target=reader, args=(proc.stdout, logging.INFO), daemon=True
        ).start()
        threading.Thread(
            target=reader, args=(proc.stderr, logging.WARNING), daemon=True
        ).start()

    def _collect_process_failure(self, proc, role):
        try:
            out, err = proc.communicate(timeout=0.5)
        except Exception:
            out = err = ""
        tail = (err or out).splitlines()[-50:]
        return f"{role} exited code {proc.returncode}. Tail:\\n" + "\\n".join(tail)

    def _spawn_worker(self, worker_id: int):
        self.copy_worker_dependencies(worker_id)
        stop_flag = threading.Event()

        def _runner():
            try:
                asyncio.run(worker_main(stop_flag=stop_flag, worker_id=str(worker_id)))
            except Exception:
                logger.exception("Worker %s crashed", worker_id)

        t = threading.Thread(target=_runner, name=f"worker-{worker_id}", daemon=True)
        t.start()

        self._worker_stops.append(stop_flag)
        self._worker_threads.append(t)

        return t

    def start(self, worker_count: int = 3) -> None:
        """Start worker processes."""

        # Install signal handlers for graceful shutdown
        def _handle_signal(signum, frame):
            logger.info("Received signal %s, initiating graceful shutdown...", signum)
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except Exception:
            # Not all environments allow setting signal handlers; ignore if so.
            pass
        try:
            self._spawn_server()
        except Exception as e:
            logger.exception("Failed to start server thread: %s", e)
            raise ServerStartupError("Failed to start server thread") from e
        self._wait_for_server_readiness()

        self._worker_count = max(1, int(worker_count))

        for i in range(self._worker_count):
            th = self._spawn_worker(worker_id=i + 1)
            logger.info("Launched worker thread %d (name=%s)", i + 1, th.name)

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop all worker processes, then force-kill if necessary."""
        if self._stopping.is_set():
            return
        self._stopping.set()
        logger.info("Stopping %d local worker thread(s)...", len(self._worker_threads))
        for ev in self._worker_stops:
            ev.set()

        end = time.time() + timeout
        for th in self._worker_threads:
            remaining = max(0.0, end - time.time())
            th.join(timeout=remaining)
            if th.is_alive():
                logger.warning("Worker thread %s did not stop in time", th.name)

        self._worker_threads.clear()
        self._worker_stops.clear()

        try:
            request_server_shutdown(timeout=1.0)
        except Exception:
            logger.debug("request_server_shutdown not available or failed; proceeding")

        if self._server_thread and self._server_thread.is_alive():
            logger.info("Waiting for server thread to shut down...")
            self._server_thread.join(timeout=timeout)

    def copy_worker_dependencies(self, worker_id: int):
        """helper to copy dependencies to worker temp directory TODO: refactor"""
        scripts_path = Path(__file__).parent
        target_dir = (
            scripts_path.parent.parent.parent.parent.parent
            / f"orchestration_files/.worker_{worker_id}_temp"
        )
        connectors_dir = scripts_path.parent / "connectors"
        logger_dir = target_dir.parent.parent / "src" / "logger"
        logger.info("Copying worker dependencies to %s", target_dir)
        # Ensure base target directory exists
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create target directory %s: %s", target_dir, e)
            raise

        def _replace_tree(src: Path, dest: Path):
            if not src.exists():
                raise FileNotFoundError(f"Source path does not exist: {src}")
            try:
                if dest.exists():
                    logger.info("Removing existing target %s", dest)
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            except Exception as e:
                logger.exception("Failed to copy %s to %s: %s", src, dest, e)
                if dest.exists():
                    try:
                        shutil.rmtree(dest)
                    except Exception:
                        logger.exception(
                            "Failed to cleanup destination %s after failed copy", dest
                        )
                        raise

        for src, dst_name in ((connectors_dir, "connectors"), (logger_dir, "logger")):
            _replace_tree(src, target_dir / dst_name)


@contextmanager
def simulation_process_context_manager(
    worker_count: int = 3,
) -> Generator[None, None, None]:
    """Context manager to start/stop local worker processes."""
    logger.info("Entering local process cluster context.")
    manager = ProcessClusterManager()
    try:
        manager.start(worker_count=worker_count)
        try:
            yield
        except KeyboardInterrupt:
            logger.info(
                "KeyboardInterrupt received in context manager; shutting down..."
            )
            raise
    finally:
        try:
            manager.stop()
        finally:
            logger.info("Exited local process cluster context.")
