"""
NOTE:
This module must be aligned with python 3.10 syntax, as open-darts whl requires it.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Callable, Protocol, Tuple

from logger import get_logger, stream_reader

from .common import (
    SerializedJson,
    SimulationResults,
    SimulationResultType,
    SimulationStatus,
)
from .conn_utils.managed_subprocess import ManagedSubprocess

logger = get_logger("threading-worker", filename=__name__)


class SimulationRunner(Protocol):
    def run(
        self, config: SerializedJson, stop: threading.Event | None = None
    ) -> Tuple[SimulationStatus, SimulationResults]: ...


class SubprocessRunner:
    """Run the simulation using the existing ManagedSubprocess logic."""

    def __init__(
        self,
        managed_subprocess_factory: Callable | None = None,
        broadcast_results_parser: Callable[[str], SimulationResults] | None = None,
        repo_root_getter: Callable[[], Path] | None = None,
        worker_id_getter: Callable[[], str | None] | None = None,
    ):
        self._managed_subprocess_factory = managed_subprocess_factory
        self._broadcast_results_parser = broadcast_results_parser
        self._repo_root_getter = repo_root_getter
        self._worker_id_getter = worker_id_getter

    def run(
        self, config: SerializedJson, stop: threading.Event | None = None
    ) -> Tuple[SimulationStatus, SimulationResults]:
        managed_factory = self._managed_subprocess_factory
        if managed_factory is None:
            managed_factory = self._default_managed_subprocess_factory

        if self._broadcast_results_parser is None:
            raise RuntimeError(
                "SubprocessRunner requires a broadcast_results_parser to be provided"
            )

        default_failed_value: float = float("nan")
        default_failed_results: SimulationResults = {
            k: default_failed_value for k in SimulationResultType
        }

        if self._repo_root_getter and self._worker_id_getter:
            try:
                repo_root = self._repo_root_getter()
            except Exception:
                repo_root = None
            wid = None
            try:
                wid = self._worker_id_getter()
            except Exception:
                wid = None

            work_dir = None
            if repo_root is not None:
                work_dir = repo_root / f"orchestration_files/.worker_{wid}_temp"

            # best-effort cleanup of old caches
            try:
                for p in Path.cwd().glob("obl_point_data_*.pkl"):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                logger.warning(
                    "Failed to scan/remove old 'obl_point_data_*.pkl' caches; proceeding anyway."
                )

            command = [
                "pixi",
                "run",
                "-e",
                "worker",
                "python",
                "-u",
                "main.py",
                config,
            ]

            env = os.environ.copy()
            if work_dir is not None:
                env.update({"PWD": str(work_dir)})

            _tw_logger = get_logger("threading-worker")
            manager = managed_factory(
                command_args=command,
                stream_reader_func=stream_reader,
                logger_info_func=_tw_logger.info,
                logger_error_func=_tw_logger.error,
                env=env,
                thread_name_prefix=f"worker-{wid}" if wid else None,
            )

            try:
                with manager as process:
                    timeout_duration = 15 * 60
                    try:
                        waited = 0.0
                        poll_step = 0.25
                        while True:
                            if stop is not None and stop.is_set():
                                logger.warning(
                                    "Stop requested; terminating OpenDarts subprocess."
                                )
                                if process.poll() is None:
                                    process.terminate()
                                    try:
                                        process.wait(timeout=3)
                                    except subprocess.TimeoutExpired:
                                        process.kill()
                                        process.wait()
                                return SimulationStatus.FAILED, default_failed_results
                            try:
                                process.wait(timeout=poll_step)
                                break
                            except subprocess.TimeoutExpired:
                                waited += poll_step
                                if waited >= timeout_duration:
                                    raise subprocess.TimeoutExpired(
                                        process.args, timeout_duration
                                    )
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"Subprocess timed out after {timeout_duration} seconds. Terminating."
                        )
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                logger.warning(
                                    "Subprocess did not terminate gracefully. Killing."
                                )
                                process.kill()
                                process.wait()
                        if manager.stdout_thread and manager.stdout_thread.is_alive():
                            manager.stdout_thread.join(timeout=2)
                        if manager.stderr_thread and manager.stderr_thread.is_alive():
                            manager.stderr_thread.join(timeout=2)
                        return SimulationStatus.TIMEOUT, default_failed_results

                    if process.returncode != 0:
                        stderr_tail = "\n".join(manager.stderr_lines[-100:])
                        stdout_tail = "\n".join(manager.stdout_lines[-100:])
                        if process.returncode == -9:
                            logger.error(
                                "OpenDarts subprocess was killed (rc=-9). This is often due to OOM kill. "
                                "Consider lowering worker_count, reducing model size, or limiting threads. "
                                "Stdout tail:\n%s\nStderr tail:\n%s",
                                stdout_tail,
                                stderr_tail,
                            )
                        else:
                            logger.error(
                                "OpenDarts subprocess failed rc=%s. Stdout tail:\n%s\nStderr tail:\n%s",
                                process.returncode,
                                stdout_tail,
                                stderr_tail,
                            )
                            if (
                                "BlockingIOError" in stderr_tail
                                and "h5py" in stderr_tail
                                and "Unable to synchronously create file" in stderr_tail
                            ):
                                logger.error(
                                    "Detected HDF5 file locking error from h5py. The worker sets HDF5_USE_FILE_LOCKING=FALSE, "
                                    "but if the error persists, ensure each worker runs in an isolated directory and no other process "
                                    "holds the same HDF5 file open."
                                )
                        return SimulationStatus.FAILED, default_failed_results

                    full_stdout = "\n".join(manager.stdout_lines)
                    broadcast_results = self._broadcast_results_parser(full_stdout)
                    return SimulationStatus.SUCCESS, broadcast_results

            except FileNotFoundError:
                logger.exception(
                    f"Failed to start subprocess. Command '{' '.join(command)}' not found."
                )
                return SimulationStatus.FAILED, default_failed_results
            except Exception as e:
                logger.exception(
                    f"An error occurred while running the simulation subprocess: {e}"
                )
                return SimulationStatus.FAILED, default_failed_results

        return SimulationStatus.FAILED, default_failed_results

    def _default_managed_subprocess_factory(*a, **k):
        return ManagedSubprocess(*a, **k)


class ThreadRunner:
    """Lightweight runner that intends to run simulation inside a thread or
    in-process.
    """

    def __init__(self, subprocess_runner: SubprocessRunner | None = None):
        self._subprocess_runner = subprocess_runner or SubprocessRunner()

    def run(
        self, config: SerializedJson, stop: threading.Event | None = None
    ) -> Tuple[SimulationStatus, SimulationResults]:
        os.environ.setdefault("OPEN_DARTS_THREAD_MODE", "1")
        return self._subprocess_runner.run(config, stop)
