import asyncio
import concurrent.futures
import logging
import shutil
import signal
import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from common.models import RunMode
from logger import (
    configure_server_logger,
    configure_worker_logger,
    get_logger,
)
from services.simulation_service.core.config import get_simulation_config
from services.simulation_service.core.infrastructure.server.src._simulation_server_grpc import (
    driver,
    request_server_shutdown,
)
from services.simulation_service.core.infrastructure.worker.src._simulation_worker_grpc import (
    main as worker_main,
)
from services.simulation_service.core.service.cluster_manager import ClusterManager

logger = get_logger(__name__)


class ServerStartupError(Exception): ...


class ProcessClusterManager(ClusterManager):
    """Process-based cluster manager that launches worker scripts.

    Each worker is started as a separate Python process that runs the
    `_simulation_worker_grpc.py` script.
    """

    def __init__(self) -> None:
        self._server_thread: threading.Thread | None = None
        self._worker_count = 0

        self._worker_threads: list[threading.Thread] = []
        self._worker_stops: list[threading.Event] = []
        self._stopping = threading.Event()

        # Cache config once — avoids repeated config lookups
        config = get_simulation_config()
        self.run_mode = config.run_mode
        self.host = config.server_host
        self.port = config.server_port
        self._server_startup_timeout = config.server_startup_timeout

        # Cache the expensive path resolution
        self._orchestration_files_path: Path = (
            Path(__file__).parent.parent.parent.parent.parent.parent
            / "orchestration_files"
        )

        self._cleanup_worker_directories()

    def _wait_for_server_readiness(self, timeout: float | None = None, interval: float = 0.25) -> None:
        if timeout is None:
            timeout = self._server_startup_timeout  # use cached value
        deadline = time.monotonic() + timeout  # monotonic is safer than time.time()
        while time.monotonic() < deadline:
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

    def _spawn_server(self) -> None:
        try:
            configure_server_logger()
        except Exception:
            logger.debug(
                "Failed to configure server logger; continuing without per-thread file handler"
            )
        self._server_thread = threading.Thread(target=driver, daemon=True, name="server")
        self._server_thread.start()

    def _spawn_worker(self, worker_id: int) -> threading.Thread:
        self.copy_worker_dependencies(worker_id)
        stop_flag = threading.Event()

        def _runner() -> None:
            try:
                asyncio.run(worker_main(stop_flag=stop_flag, worker_id=str(worker_id)))
            except Exception:
                logger.exception("Worker %s crashed", worker_id)

        t = threading.Thread(target=_runner, name=f"worker-{worker_id}", daemon=True)
        t.start()

        self._worker_stops.append(stop_flag)
        self._worker_threads.append(t)

        return t

    def start(self, worker_count: int) -> None:
        """Start worker processes."""

        def _handle_signal(signum, _frame) -> None:
            logger.info("Received signal %s, initiating graceful shutdown...", signum)
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except Exception:
            pass

        try:
            self._spawn_server()
        except Exception as e:
            logger.exception("Failed to start server thread: %s", e)
            raise ServerStartupError("Failed to start server thread") from e

        self._wait_for_server_readiness()
        self._worker_count = max(1, int(worker_count))

        # Configure all worker loggers first (must stay on the main thread)
        for i in range(self._worker_count):
            try:
                configure_worker_logger(i + 1)
            except Exception:
                logger.debug(
                    "Failed to configure worker %d logger; continuing without per-thread file handler",
                    i + 1,
                )

        # Spawn all workers in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._worker_count, thread_name_prefix="spawn-worker"
        ) as executor:
            futures = {
                executor.submit(self._spawn_worker, worker_id=i + 1): i + 1
                for i in range(self._worker_count)
            }
            for future in concurrent.futures.as_completed(futures):
                worker_id = futures[future]
                try:
                    th = future.result()
                    logger.info("Launched worker thread %d (name=%s)", worker_id, th.name)
                except Exception:
                    logger.exception("Failed to spawn worker %d", worker_id)

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop all worker threads, then wait for them concurrently."""
        if self._stopping.is_set():
            return
        self._stopping.set()
        logger.info("Stopping %d local worker thread(s)...", len(self._worker_threads))

        # Signal all workers to stop at once
        for ev in self._worker_stops:
            ev.set()

        # Join all worker threads concurrently
        deadline = time.monotonic() + timeout

        def _join(th: threading.Thread) -> None:
            remaining = max(0.0, deadline - time.monotonic())
            th.join(timeout=remaining)
            if th.is_alive():
                logger.warning("Worker thread %s did not stop in time", th.name)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(self._worker_threads)),
            thread_name_prefix="stop-worker",
        ) as executor:
            list(executor.map(_join, self._worker_threads))

        self._worker_threads.clear()
        self._worker_stops.clear()

        if self.run_mode == RunMode.Optimization:
            self._cleanup_worker_directories()

        try:
            request_server_shutdown(timeout=1.0)
        except Exception:
            logger.debug("request_server_shutdown not available or failed; proceeding")

        if self._server_thread and self._server_thread.is_alive():
            logger.info("Waiting for server thread to shut down...")
            self._server_thread.join(timeout=timeout)

    def copy_worker_dependencies(self, worker_id: int) -> None:
        """Copy dependencies to worker temp directory."""
        scripts_path = Path(__file__).parent
        target_dir = self._orchestration_files_path / f".worker_{worker_id}_temp"
        connectors_dir = scripts_path.parent / "connectors"
        logger_dir = self._orchestration_files_path.parent / "src" / "logger"

        logger.info("Copying worker dependencies to %s", target_dir)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create target directory %s: %s", target_dir, e)
            raise

        def _replace_tree(src: Path, dest: Path) -> None:
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

        # Copy connectors and logger trees concurrently
        pairs = [(connectors_dir, "connectors"), (logger_dir, "logger")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_replace_tree, src, target_dir / dst_name)
                for src, dst_name in pairs
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # re-raise any exceptions

    def _cleanup_worker_directories(self) -> None:
        if not self._orchestration_files_path.exists():
            return

        logger.info("Cleaning up worker temp directories...")
        worker_dirs = [
            d for d in self._orchestration_files_path.glob(".worker_*_temp")
            if d.is_dir()
        ]

        def _remove(worker_dir: Path) -> None:
            try:
                logger.debug("Removing worker directory: %s", worker_dir)
                shutil.rmtree(worker_dir)
            except Exception as e:
                logger.warning("Failed to remove worker directory %s: %s", worker_dir, e)

        # Remove all worker dirs concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(worker_dirs))) as executor:
            list(executor.map(_remove, worker_dirs))

        logger.info("Worker temp directory cleanup complete.")


@contextmanager
def simulation_process_context_manager(
    worker_count: int,
) -> Generator[None, None, None]:
    """Context manager to start/stop local worker processes."""
    logger.debug("Entering local process cluster context.")
    manager = ProcessClusterManager()
    try:
        manager.start(worker_count=worker_count)
        try:
            yield
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in context manager; shutting down...")
            raise
    finally:
        try:
            manager.stop()
        finally:
            logger.info("Exited local process cluster context.")