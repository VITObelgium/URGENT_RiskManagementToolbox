from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from logger import get_logger, stream_reader
from .common import (
    JsonPath,
    SimulationResults,
    SimulationStatus,
)
from .conn_utils import (
    ManagedSubprocess,
    docker_job_workspace,
    static_workspace,
)

logger = get_logger("threading-worker", filename=__name__)

_GRACEFUL_TERMINATE_TIMEOUT = 5
_STOP_TERMINATE_TIMEOUT = 3
_THREAD_JOIN_TIMEOUT = 2
_POLL_STEP = 0.25

T = TypeVar("T")


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
        worker_simulation_timeout_seconds: int,
        repo_root_getter: Callable[[], Path],
        worker_id_getter: Callable[[], str | None],
        managed_subprocess_factory: Callable[..., ManagedSubprocess] | None = None,
        broadcast_results_parser: Callable[[str], SimulationResults] | None = None,
    ):

        self._managed_subprocess_factory = (
            managed_subprocess_factory or self._default_managed_subprocess_factory

        )
        self._broadcast_results_parser = broadcast_results_parser
        self._repo_root_getter = repo_root_getter
        self._worker_id_getter = worker_id_getter
        self._timeout_duration = worker_simulation_timeout_seconds

    def run(
        self,
        config: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        defaults = user_cost_function_with_default_values

        env, thread_name = self._build_env()
        command = self._build_command(config)
        self._cleanup_stale_caches()

        manager = self._managed_subprocess_factory(
            command_args=command,
            stream_reader_func=stream_reader,
            logger_info_func=logger.info,
            logger_error_func=logger.error,
            env=env,
            thread_name_prefix=thread_name,
        )

        try:
            with manager as m:
                if m.process is None:
                    return SimulationStatus.FAILED, defaults
                status = self._wait_for_process(m.process, stop)

                if status == SimulationStatus.TIMEOUT:
                    self._terminate_process(m.process, graceful=True)
                    self._join_io_threads(m)
                    return SimulationStatus.TIMEOUT, defaults

                if status == SimulationStatus.FAILED:
                    return SimulationStatus.FAILED, defaults

                if m.returncode is None or m.returncode != 0:
                    self._log_nonzero_returncode(m.returncode or 1, m)
                    return SimulationStatus.FAILED, defaults

                return self._parse_and_merge(m, defaults)

        except KeyError as e:
            logger.exception(
                "Broadcast results keys do not match user cost-function keys. "
                "Expected keys: %s. Raw error: %s\nStdout tail:\n%s",
                sorted(defaults.keys()),
                e,
                _tail(manager.stdout_lines),
            )
            return SimulationStatus.EXCEPTION, defaults
        except FileNotFoundError:
            logger.exception(
                "Failed to start subprocess. Command '%s' not found.",
                " ".join(command),
            )
            return SimulationStatus.EXCEPTION, defaults
        except Exception as e:
            logger.exception(
                "An error occurred while running the simulation subprocess: %s", e
            )
            return SimulationStatus.EXCEPTION, defaults

    def _build_env(self) -> tuple[dict[str, str], str | None]:
        """Resolve worker id and working directory."""
        repo_root = _safe_call(self._repo_root_getter)
        wid = _safe_call(self._worker_id_getter)

        env = os.environ.copy()
        if repo_root is not None and wid is not None:
            work_dir = repo_root / f"orchestration_files/.worker_{wid}_temp"
            env["PWD"] = str(work_dir)

        thread_name = f"worker-{wid}" if wid else None
        return env, thread_name

    @staticmethod
    def _build_command(config: JsonPath) -> list[str]:
        return ["pixi", "run", "-e", "worker", "python", "-u", "main.py", config]

    @staticmethod
    def _cleanup_stale_caches() -> None:
        try:
            for p in Path.cwd().glob("obl_point_data_*.pkl"):
                try:
                    p.unlink(missing_ok=True)
                except OSError as e:
                    logger.debug("Could not remove cache file %s: %s", p, e)
        except OSError as e:
            logger.warning(
                "Failed to scan/remove old 'obl_point_data_*.pkl' caches; proceeding anyway. Error: %s",
                e,
            )

    def _wait_for_process(
        self, process: subprocess.Popen, stop: threading.Event | None
    ) -> SimulationStatus:
        """Poll until process finishes, times out, or stop is requested.

        Returns FAILED (stop), TIMEOUT, or SUCCESS (process exited).
        """
        waited = 0.0
        while True:
            if stop is not None and stop.is_set():
                logger.warning("Stop requested; terminating OpenDarts subprocess.")
                self._terminate_process(process, graceful=False)
                return SimulationStatus.FAILED

            try:
                process.wait(timeout=_POLL_STEP)
                return SimulationStatus.SUCCESS
            except subprocess.TimeoutExpired:
                waited += _POLL_STEP
                if waited >= self._timeout_duration:
                    logger.warning(
                        "Subprocess timed out after %s seconds. Terminating.",
                        self._timeout_duration,
                    )
                    return SimulationStatus.TIMEOUT

    @staticmethod
    def _terminate_process(process: subprocess.Popen, graceful: bool) -> None:
        """Terminate or kill a running process."""
        if process.poll() is not None:
            return
        timeout = _GRACEFUL_TERMINATE_TIMEOUT if graceful else _STOP_TERMINATE_TIMEOUT
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Subprocess did not terminate gracefully. Killing.")
            process.kill()
            process.wait()

    @staticmethod
    def _join_io_threads(manager: ManagedSubprocess) -> None:
        for thread in (manager.stdout_thread, manager.stderr_thread):
            if thread and thread.is_alive():
                thread.join(timeout=_THREAD_JOIN_TIMEOUT)

    @staticmethod
    def _log_nonzero_returncode(returncode: int, manager: ManagedSubprocess) -> None:
        stdout_tail = _tail(manager.stdout_lines)
        stderr_tail = _tail(manager.stderr_lines)

        if returncode == -9:
            logger.error(
                "OpenDarts subprocess was killed (rc=-9). Likely OOM kill. "
                "Consider lowering worker_count, reducing model size, or limiting threads.\n"
                "Stdout tail:\n%s\nStderr tail:\n%s",
                stdout_tail,
                stderr_tail,
            )
            return

        logger.error(
            "OpenDarts subprocess failed rc=%s.\nStdout tail:\n%s\nStderr tail:\n%s",
            returncode,
            stdout_tail,
            stderr_tail,
        )

        if (
            "BlockingIOError" in stderr_tail
            and "h5py" in stderr_tail
            and "Unable to synchronously create file" in stderr_tail
        ):
            logger.error(
                "Detected HDF5 file locking error from h5py. "
                "Ensure each worker runs in an isolated directory and no other "
                "process holds the same HDF5 file open."
            )
            return SimulationStatus.EXCEPTION, user_cost_function_with_default_values
        except FileNotFoundError as e:
            logger.exception("Failed to prepare simulation workspace: %s", e)
            return SimulationStatus.EXCEPTION, user_cost_function_with_default_values
        except Exception as e:
            logger.exception("Error running simulation subprocess: %s", e)
            return SimulationStatus.EXCEPTION, user_cost_function_with_default_values

    def _parse_and_merge(
        self,
        manager: ManagedSubprocess,
        defaults: SimulationResults,
    ) -> tuple[SimulationStatus, SimulationResults]:
        """Parse broadcast results from stdout and merge with defaults."""
        full_stdout = "\n".join(manager.stdout_lines)
        broadcast_results = self._broadcast_results_parser(full_stdout)

        if not broadcast_results:
            logger.error(
                "Subprocess exited rc=0 but broadcast results are empty. "
                "The simulation likely crashed silently or produced no broadcast output.\n"
                "Stdout tail:\n%s\nStderr tail:\n%s",
                _tail(manager.stdout_lines),
                _tail(manager.stderr_lines),
            )
            return SimulationStatus.EXCEPTION, defaults

        merged = _update_user_cost_function_with_simulation_results(
            defaults, broadcast_results
        )
        return SimulationStatus.SUCCESS, merged

    @staticmethod
    def _default_managed_subprocess_factory(*a: Any, **k: Any) -> ManagedSubprocess:
        return ManagedSubprocess(*a, **k)


class ThreadRunner:
    """Lightweight runner that runs simulation inside a thread or in-process."""
    def _get_workspace_manager(
        self,
        runner_mode: str,
        repo_root: Path | None,
        worker_id: str | None,
    ):
        """Get the appropriate workspace context manager for the runner mode."""
        if runner_mode == "docker":
            template_dir_raw = os.environ.get("SIM_MODEL_DIR")
            if not template_dir_raw:
                logger.error("Docker runner mode requires SIM_MODEL_DIR to be set.")
                return None
            return docker_job_workspace(Path(template_dir_raw))

        # Thread mode
        if repo_root is None or worker_id is None:
            logger.debug(
                "Thread runner mode is missing repo_root or worker_id; "
                "running without an isolated worker workspace."
            )
            return static_workspace(None)
        work_dir = repo_root / f"orchestration_files/.worker_{worker_id}_temp"
        return static_workspace(work_dir)

    def _execute_in_workspace(
        self,
        config: JsonPath,
        defaults: SimulationResults,
        work_dir: Path | None,
        worker_id: str | None,
        runner_mode: str,
        stop: threading.Event | None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        """Execute the simulation within the prepared workspace."""
        command = _build_command(config, runner_mode)
        env = _build_env(work_dir, runner_mode)

        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)

        tw_logger = get_logger("threading-worker")
        manager = self._managed_subprocess_factory(
            command_args=command,
            stream_reader_func=stream_reader,
            logger_info_func=tw_logger.info,
            logger_error_func=tw_logger.error,
            env=env,
            cwd=work_dir,
            thread_name_prefix=f"worker-{worker_id}" if worker_id else None,
        )

        with manager as process:
            status = self._wait_for_process(process, manager, stop)
            if status is not None:
                return status, defaults

            if process.returncode != 0:
                _log_process_failure(process.returncode, manager)
                return SimulationStatus.FAILED, defaults

            return self._parse_results(manager, defaults)

    def _wait_for_process(
        self,
        process: subprocess.Popen,
        manager: ManagedSubprocess,
        stop: threading.Event | None,
    ) -> SimulationStatus | None:
        """Wait for process completion, handling stop signals and timeouts."""
        waited = 0.0
        poll_step = 0.25

        while True:
            # Check for stop signal
            if stop is not None and stop.is_set():
                logger.warning("Stop requested; terminating OpenDarts subprocess.")
                _terminate_process(process)
                return SimulationStatus.FAILED

            # Wait for process with timeout
            try:
                process.wait(timeout=poll_step)
                return None  # Process completed normally
            except subprocess.TimeoutExpired:
                waited += poll_step
                if waited >= self._timeout_duration:
                    logger.warning(
                        "Subprocess timed out after %s seconds. Terminating.",
                        self._timeout_duration,
                    )
                    _terminate_process(process)
                    _join_output_threads(manager)
                    return SimulationStatus.TIMEOUT

    def _parse_results(
        self, manager: ManagedSubprocess, defaults: SimulationResults
    ) -> tuple[SimulationStatus, SimulationResults]:
        """Parse simulation results from stdout."""
        parser = self._broadcast_results_parser
        if parser is None:
            raise RuntimeError(
                "SubprocessRunner requires a broadcast_results_parser to be provided"
            )
        full_stdout = "\n".join(manager.stdout_lines)
        broadcast_results = parser(full_stdout)
        merged = _merge_results(defaults, broadcast_results)
        return SimulationStatus.SUCCESS, merged



    def __init__(self, subprocess_runner: SubprocessRunner):
        self._subprocess_runner = subprocess_runner

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


    @staticmethod
    def _default_subprocess_factory(*args, **kwargs) -> ManagedSubprocess:
        """Default factory for creating ManagedSubprocess instances."""
        return ManagedSubprocess(*args, **kwargs)


def _build_command(config: JsonPath, runner_mode: str) -> list[str]:
    """Build the subprocess command based on runner mode."""
    if runner_mode == "docker":
        return [sys.executable, "-u", "main.py", config]
    return ["pixi", "run", "-e", "worker", "python", "-u", "main.py", config]


def _build_env(work_dir: Path | None, runner_mode: str) -> dict[str, str]:
    """Build environment variables for the subprocess."""
    env = os.environ.copy()
    if work_dir is not None:
        env["PWD"] = str(work_dir)
        if runner_mode == "docker":
            env["SIM_MODEL_TEMPLATE_DIR"] = os.environ.get("SIM_MODEL_DIR", "")
            env["SIM_MODEL_DIR"] = str(work_dir)
    return env


def _terminate_process(process: subprocess.Popen) -> None:
    """Gracefully terminate a process, killing if necessary."""
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("Subprocess did not terminate gracefully. Killing.")
        process.kill()
        process.wait()


def _join_output_threads(manager: ManagedSubprocess) -> None:
    """Join stdout/stderr reader threads."""
    if manager.stdout_thread and manager.stdout_thread.is_alive():
        manager.stdout_thread.join(timeout=2)
    if manager.stderr_thread and manager.stderr_thread.is_alive():
        manager.stderr_thread.join(timeout=2)


def _log_process_failure(returncode: int, manager: ManagedSubprocess) -> None:
    """Log detailed information about a process failure."""
    stderr_tail = "\n".join(manager.stderr_lines[-100:])
    stdout_tail = "\n".join(manager.stdout_lines[-100:])

    if returncode == -9:
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
            returncode,
            stdout_tail,
            stderr_tail,
        )
        if (
            "BlockingIOError" in stderr_tail
            and "h5py" in stderr_tail
            and "Unable to synchronously create file" in stderr_tail
        ):
            logger.error(
                "Detected HDF5 file locking error from h5py. The worker sets "
                "HDF5_USE_FILE_LOCKING=FALSE, but if the error persists, ensure "
                "each worker runs in an isolated directory and no other process "
                "holds the same HDF5 file open."
            )


def _merge_results(
    defaults: SimulationResults, broadcast_results: SimulationResults
) -> SimulationResults:
    """
    Merge broadcast results into defaults, validating key consistency.

    Raises:
        KeyError: If keys don't match between defaults and broadcast_results
    """
    default_keys = set(defaults.keys())
    result_keys = set(broadcast_results.keys())

    if default_keys != result_keys:
        missing = sorted(default_keys - result_keys)
        extra = sorted(result_keys - default_keys)
        raise KeyError(
            f"Broadcast results keys do not match user cost-function keys. "
            f"Missing in connector={missing}, extra in connector={extra}"
        )

    merged = dict(defaults)
    merged.update(broadcast_results)
    return merged
