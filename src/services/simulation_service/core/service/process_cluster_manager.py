import asyncio
import shutil
import signal
import socket
import threading
import time
from collections.abc import Generator
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_EXCEPTION,
    Future,
)
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

# Maximum threads used for parallel I/O tasks (dep-copy, dir cleanup).
# Kept separate from the worker pool so I/O concurrency is bounded independently.
_IO_POOL_THREADS = 8


class ServerStartupError(Exception): ...


class ProcessClusterManager(ClusterManager):
    """Process-based cluster manager that launches worker scripts.

    Each worker is started as a separate Python thread that runs the
    `_simulation_worker_grpc` async main.

    Parallelism improvements over the original implementation
    ---------------------------------------------------------
    1. **Parallel dependency copy** – ``copy_worker_dependencies`` is now
       called for all workers simultaneously via a ``ThreadPoolExecutor``
       instead of sequentially in the spawn loop.  The first copy error is
       re-raised immediately so startup fails fast.

    2. **Parallel worker spawn** – threads are started concurrently; the main
       thread only blocks once *all* of them have been submitted.

    3. **Parallel stop/join** – all stop-events are set at once, and threads
       are joined with a *shared* deadline rather than a per-thread timeout
       that resets on every join call (the original could take N×timeout when
       many workers were stuck).

    4. **Parallel directory cleanup** – ``_cleanup_worker_directories`` fans
       out ``shutil.rmtree`` calls across the same I/O executor.

    5. **Logger config moved into the worker thread** – previously
       ``configure_worker_logger`` was called in the main thread before
       spawning, which serialised startup and could block signal delivery.
       Now it runs inside the worker thread immediately on entry.

    Public API is identical to the original.
    """

    def __init__(self) -> None:
        self._server_thread: threading.Thread | None = None
        self._worker_count = 0

        self._worker_threads: list[threading.Thread] = []
        self._worker_stops: list[threading.Event] = []
        self._stopping = threading.Event()

        config = get_simulation_config()
        self.run_mode = config.run_mode
        self.host = config.server_host
        self.port = config.server_port

        self._cleanup_worker_directories()

    # ------------------------------------------------------------------
    # Server helpers
    # ------------------------------------------------------------------

    def _wait_for_server_readiness(self, timeout=None, interval=0.25):
        if timeout is None:
            timeout = get_simulation_config().server_startup_timeout
        deadline = time.monotonic() + timeout
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

    def _spawn_server(self):
        try:
            configure_server_logger()
        except Exception:
            logger.debug(
                "Failed to configure server logger; continuing without per-thread file handler"
            )
        self._server_thread = threading.Thread(
            target=driver, daemon=True, name="server"
        )
        self._server_thread.start()

    # ------------------------------------------------------------------
    # Worker helpers
    # ------------------------------------------------------------------

    def _spawn_worker(self, worker_id: int) -> tuple[threading.Thread, threading.Event]:
        """Create and start a single worker thread.

        Logger configuration now happens *inside* the thread so that the main
        thread is never blocked by per-worker I/O setup.

        Returns the (thread, stop_event) pair so the caller can register them.
        """
        stop_flag = threading.Event()

        def _runner() -> None:
            try:
                configure_worker_logger(worker_id)
            except Exception:
                logger.debug(
                    "Failed to configure worker %d logger; continuing without per-thread file handler",
                    worker_id,
                )
            try:
                asyncio.run(worker_main(stop_flag=stop_flag, worker_id=str(worker_id)))
            except Exception:
                logger.exception("Worker %s crashed", worker_id)

        t = threading.Thread(target=_runner, name=f"worker-{worker_id}", daemon=True)
        t.start()
        return t, stop_flag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, worker_count: int) -> None:
        """Start the server thread and all worker threads.

        Steps
        -----
        1. Install signal handlers.
        2. Start the gRPC server thread and wait for it to be ready.
        3. Copy worker dependencies **in parallel** using a thread pool.
        4. Spawn all worker threads **concurrently**.
        """

        def _handle_signal(signum, _frame):
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
        worker_ids = list(range(1, self._worker_count + 1))

        # --- Step 1: copy dependencies for all workers in parallel ----------
        logger.info(
            "Copying worker dependencies for %d worker(s) in parallel…",
            self._worker_count,
        )
        with ThreadPoolExecutor(
            max_workers=min(self._worker_count, _IO_POOL_THREADS),
            thread_name_prefix="dep-copy",
        ) as pool:
            futures: dict[Future[None], int] = {
                pool.submit(self.copy_worker_dependencies, wid): wid
                for wid in worker_ids
            }
            # Raise the first exception immediately; cancel the rest.
            for fut in as_completed(futures):
                wid = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    logger.exception(
                        "Dependency copy failed for worker %d: %s", wid, exc
                    )
                    # Cancel remaining futures (best-effort).
                    for f in futures:
                        f.cancel()
                    raise

        # --- Step 2: spawn all worker threads concurrently ------------------
        logger.info("Spawning %d worker thread(s)…", self._worker_count)
        with ThreadPoolExecutor(
            max_workers=self._worker_count,
            thread_name_prefix="worker-spawn",
        ) as pool:
            spawn_futures: dict[
                Future[tuple[threading.Thread, threading.Event]], int
            ] = {pool.submit(self._spawn_worker, wid): wid for wid in worker_ids}
            done, _ = wait(spawn_futures, return_when=FIRST_EXCEPTION)
            for spawn_fut in done:
                wid = spawn_futures[spawn_fut]
                try:
                    t, stop_flag = spawn_fut.result()
                    self._worker_threads.append(t)
                    self._worker_stops.append(stop_flag)
                    logger.info("Launched worker thread %d (name=%s)", wid, t.name)
                except Exception as exc:
                    logger.exception("Failed to spawn worker %d: %s", wid, exc)
                    raise

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop all worker threads with a shared deadline.

        The shared deadline means the total join time is bounded by *timeout*
        regardless of how many workers there are, unlike the original
        implementation which could take N×timeout in the worst case.
        """
        if self._stopping.is_set():
            return
        self._stopping.set()

        logger.info("Stopping %d local worker thread(s)…", len(self._worker_threads))

        # Signal all workers simultaneously.
        for ev in self._worker_stops:
            ev.set()

        # Join all threads against a single shared deadline.
        deadline = time.monotonic() + timeout
        for th in self._worker_threads:
            remaining = max(0.0, deadline - time.monotonic())
            th.join(timeout=remaining)
            if th.is_alive():
                logger.warning("Worker thread %s did not stop in time", th.name)

        self._worker_threads.clear()
        self._worker_stops.clear()

        if self.run_mode == RunMode.Optimization:
            self._cleanup_worker_directories()

        try:
            request_server_shutdown(timeout=1.0)
        except Exception:
            logger.debug("request_server_shutdown not available or failed; proceeding")

        if self._server_thread and self._server_thread.is_alive():
            logger.info("Waiting for server thread to shut down…")
            self._server_thread.join(timeout=timeout)

    def copy_worker_dependencies(self, worker_id: int) -> None:
        """Copy shared file dependencies into a per-worker temp directory.

        This method is now called from a thread pool during ``start()``, so
        multiple workers copy simultaneously.  The method itself is stateless
        and does not mutate any shared instance variables, making it safe for
        concurrent execution.
        """
        scripts_path = Path(__file__).parent
        target_dir = (
            scripts_path.parent.parent.parent.parent.parent
            / f"orchestration_files/.worker_{worker_id}_temp"
        )
        connectors_dir = scripts_path.parent / "connectors"
        logger_dir = target_dir.parent.parent / "src" / "logger"
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

        for src, dst_name in ((connectors_dir, "connectors"), (logger_dir, "logger")):
            _replace_tree(src, target_dir / dst_name)

    def _cleanup_worker_directories(self) -> None:
        """Remove all per-worker temp directories in parallel."""
        scripts_path = Path(__file__).parent
        orchestration_files = (
            scripts_path.parent.parent.parent.parent.parent / "orchestration_files"
        )

        if not orchestration_files.exists():
            return

        worker_dirs = [
            d for d in orchestration_files.glob(".worker_*_temp") if d.is_dir()
        ]
        if not worker_dirs:
            return

        logger.info(
            "Cleaning up %d worker temp director(ies) in parallel…", len(worker_dirs)
        )

        def _remove(p: Path) -> None:
            try:
                shutil.rmtree(p)
                logger.debug("Removed worker directory: %s", p)
            except Exception as e:
                logger.warning("Failed to remove worker directory %s: %s", p, e)

        with ThreadPoolExecutor(
            max_workers=min(len(worker_dirs), _IO_POOL_THREADS),
            thread_name_prefix="dir-cleanup",
        ) as pool:
            list(pool.map(_remove, worker_dirs))

        logger.info("Worker temp directory cleanup complete.")


# ---------------------------------------------------------------------------
# Public context manager (unchanged API)
# ---------------------------------------------------------------------------


@contextmanager
def simulation_process_context_manager(
    worker_count: int,
) -> Generator[None, None, None]:
    """Context manager to start/stop local worker threads."""
    logger.debug("Entering local process cluster context.")
    manager = ProcessClusterManager()
    try:
        manager.start(worker_count=worker_count)
        try:
            yield
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in context manager; shutting down…")
            raise
    finally:
        try:
            manager.stop()
        finally:
            logger.info("Exited local process cluster context.")
