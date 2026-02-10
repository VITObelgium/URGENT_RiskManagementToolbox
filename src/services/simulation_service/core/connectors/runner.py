"""
NOTE:
This module must be aligned with python 3.10 syntax, as open-darts whl requires it.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Callable, Protocol

from logger import get_logger, stream_reader

from .common import (
    JsonPath,
    SimulationResults,
    SimulationStatus,
)
from .conn_utils import ManagedSubprocess, get_timeout_value

logger = get_logger("threading-worker", filename=__name__)


class SimulationRunner(Protocol):
    def run(
        self,
        config: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]: ...


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
        self._timeout_duration = get_timeout_value()

    def run(
        self,
        config: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        managed_factory = self._managed_subprocess_factory
        if managed_factory is None:
            managed_factory = self._default_managed_subprocess_factory

        if self._broadcast_results_parser is None:
            raise RuntimeError(
                "SubprocessRunner requires a broadcast_results_parser to be provided"
            )

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
                                return (
                                    SimulationStatus.FAILED,
                                    user_cost_function_with_default_values,
                                )
                            try:
                                process.wait(timeout=poll_step)
                                break
                            except subprocess.TimeoutExpired:
                                waited += poll_step
                                if waited >= self._timeout_duration:
                                    raise subprocess.TimeoutExpired(
                                        process.args, self._timeout_duration
                                    )
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"Subprocess timed out after {self._timeout_duration} seconds. Terminating."
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
                        return (
                            SimulationStatus.TIMEOUT,
                            user_cost_function_with_default_values,
                        )

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
                                    "Detected HDF5 file locking error from h5py. The worker sets HDF5_USE_FILE_LOCKING=FALSE,\nbut if the error persists, ensure each worker runs in an isolated directory and no other process\nholds the same HDF5 file open."
                                )
                        return (
                            SimulationStatus.FAILED,
                            user_cost_function_with_default_values,
                        )

                    full_stdout = "\n".join(manager.stdout_lines)
                    broadcast_results = self._broadcast_results_parser(full_stdout)

                    user_cost_function_with_simulation_results = (
                        _update_user_cost_function_with_simulation_results(
                            user_cost_function_with_default_values, broadcast_results
                        )
                    )

                    return (
                        SimulationStatus.SUCCESS,
                        user_cost_function_with_simulation_results,
                    )

            except KeyError as e:
                logger.exception(
                    f"Broadcast results keys do not match user cost-function keys. {e}"
                )
                return (
                    SimulationStatus.EXCEPTION,
                    user_cost_function_with_default_values,
                )
            except FileNotFoundError:
                logger.exception(
                    f"Failed to start subprocess. Command '{' '.join(command)}' not found."
                )
                return (
                    SimulationStatus.EXCEPTION,
                    user_cost_function_with_default_values,
                )
            except Exception as e:
                logger.exception(
                    f"An error occurred while running the simulation subprocess: {e}"
                )
                return (
                    SimulationStatus.EXCEPTION,
                    user_cost_function_with_default_values,
                )

        return SimulationStatus.FAILED, user_cost_function_with_default_values

    def _default_managed_subprocess_factory(*a, **k):
        return ManagedSubprocess(*a, **k)


class ThreadRunner:
    """Lightweight runner that intends to run simulation inside a thread or
    in-process.
    """

    def __init__(self, subprocess_runner: SubprocessRunner | None = None):
        self._subprocess_runner = subprocess_runner or SubprocessRunner()

    def run(
        self,
        config: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        os.environ.setdefault("OPEN_DARTS_THREAD_MODE", "1")
        return self._subprocess_runner.run(
            config, user_cost_function_with_default_values, stop
        )


def _update_user_cost_function_with_simulation_results(
    user_cost_function_with_default_values: SimulationResults,
    simulation_results: SimulationResults,
) -> SimulationResults:
    """
    Compare keys of user_cost_function_with_default_values and simulation_results.
    If keys mismatch -> raise.
    Else -> return user_cost updated with values from simulation_results.
    """
    user_keys = set(user_cost_function_with_default_values.keys())
    sim_keys = set(simulation_results.keys())

    if user_keys != sim_keys:
        missing_in_sim = sorted(user_keys - sim_keys)
        extra_in_sim = sorted(sim_keys - user_keys)
        raise KeyError(
            "Broadcast results keys do not match user cost-function keys. "
            f"present in config and missing in connector={missing_in_sim}, present in connector and missing in config={extra_in_sim}"
        )

    updated = dict(user_cost_function_with_default_values)
    updated.update(simulation_results)
    return updated
